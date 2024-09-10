import os
import sys
import numpy as np
import logging
import torch
from torch.utils.data import DistributedSampler
import pytorch_lightning as pl, pickle

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.getcwd()
sys.path.append(SCRIPT_DIR)

from read_emg import (
    EMGDataModule
)
from architecture import MONAConfig, MONA
from datetime import datetime
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform

from dataloaders import (
    LibrispeechDataset,
    EMGAndSpeechModule,
    cache_dataset,
    collate_gaddy_or_speech,
    BalancedBinPackingBatchSampler,
)

from functools import partial

##
DEBUG = False

if DEBUG:
    RUN_ID = "debug"
else:
    RUN_ID = "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.set_float32_matmul_precision("high")  # highest (32-bit) by default
torch.backends.cudnn.allow_tf32 = True  # should be True by default

per_index_cache = True  # read each index from disk separately

isotime = datetime.now().isoformat()

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 200
    limit_val_batches = 20  # will not run on_validation_epoch_end
    log_neptune = False
    n_epochs = 20
    precision = "bf16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

else:
    NUM_GPUS = 1
    grad_accum = 2  # might need if run on 1 GPU
    precision = "bf16-mixed"
    limit_train_batches = None
    limit_val_batches = None
    log_neptune = False
    n_epochs = 200
    num_sanity_val_steps = 0  # may prevent crashing of distributed training
    logger_level = logging.WARNING


if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""


# on my local machine
data_dir = "dataFolder"
librispeech_directory = "dataFolder"
lm_directory = os.getcwd()

normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

if __name__ == "__main__":

    # base_bzs = [1, 16, 32, 64]
    base_bz = 32 #TODO: Test this parameter of the speed [1,16,32,64]
    val_bz = 2  # terrible memory usage even at 8, I'm not sure why so bad...
    # gaddy used max_len = 128000, we double because of LibriSpeech
    # TODO: try 512000 and grad_accum=1 (prob OOM but might be faster!)
    # also almost def won't work for supTcon + dtw
    # max_len = 48000 # from best perf with 4 x V100
    max_len = 48000  #

    togglePhones = True
    learning_rate = 3e-4
    seqlen = 600
    white_noise_sd = 0
    constant_offset_sd = 0
    use_dtw = True
    use_crossCon = True
    use_supTcon = True
    audio_lambda = 1.0
    emg_lambda = 1.0
    weight_decay = 0.1
    latent_affine = True
    # Gaddy is 16% silent EMG, 84% vocalized EMG, and we use LibriSpeech for the rest
    # by utterance count, not by time
    frac_semg = 1588 / (5477 + 1588)
    frac_vocal = 1 - frac_semg
    frac_semg /= 2
    frac_vocal /= 2
    frac_librispeech = 0.5
    # TODO: should sweep librispeech ratios...
    # batch_class_proportions: np.ndarray = field(default_factory=lambda: np.array([frac_semg, frac_vocal, frac_librispeech]))
    batch_class_proportions = np.array([frac_semg, frac_vocal, frac_librispeech])
    latest_epoch = -1
    matmul_tf32 = True

    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32  # false by default


    # if ckpt_path != "":
    #     raise NotImplementedError("TODO: implement output_directory for ckpt_path")


    MANUAL_RESUME = False

    if MANUAL_RESUME:
        ckpt_path = "~/silent_speech/dataFolder/output_arthurcheckpoints/lightning_logs/"
        # Find the largest version in the checkpoints folder
        ckpt_path = os.path.expanduser(ckpt_path)

        last_verison = sorted(os.listdir(ckpt_path))[-1]
        last_verison = os.path.join(last_verison, "checkpoints")

        ckpt_path = os.path.join(ckpt_path, last_verison)
        # Find the .ckpt file in the checkpoints folder
        ckpt_path = os.path.expanduser(ckpt_path)
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])

    output_directory = os.path.join(data_dir, f"output_arthur{RUN_ID}")

    # needed for using CachedDataset
    emg_datamodule = EMGDataModule(
        data_dir,
        togglePhones,
        normalizers_file,
        max_len=max_len,
        collate_fn=collate_gaddy_or_speech,
        pin_memory=(not DEBUG),
        batch_size=val_bz,
    )

    emg_train = emg_datamodule.train

    mfcc_norm, emg_norm = pickle.load(open(normalizers_file, "rb"))

    if NUM_GPUS > 1:
        strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    else:
        strategy = "auto"

    devices = NUM_GPUS

    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        level=logger_level,
        format="%(message)s",
        force=True,
    )

    # logging.debug("DEBUG mode")
    # ##
    if NUM_GPUS > 1:
        num_workers = 0  # nccl backend doesn't support num_workers>0
        rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
        bz = base_bz * NUM_GPUS
        if rank_key not in os.environ:
            rank = 0
        else:
            rank = int(os.environ[rank_key])
        logging.info(f"SETTING CUDA DEVICE ON RANK: {rank}")

        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        TrainBatchSampler = partial(
            BalancedBinPackingBatchSampler,
            num_replicas=NUM_GPUS,
            # in emg_speech_dset_lengths we divide length by 8
            max_len=max_len // 8,
            always_include_class=[0],
        )
        ValSampler = lambda: DistributedSampler(
            emg_datamodule.val, shuffle=False, num_replicas=NUM_GPUS
        )
        TestSampler = lambda: DistributedSampler(
            emg_datamodule.test, shuffle=False, num_replicas=NUM_GPUS
        )
    else:
        # TrainBatchSampler = SizeAwareStratifiedBatchSampler
        TrainBatchSampler = partial(
            BalancedBinPackingBatchSampler,
            num_replicas=NUM_GPUS,
            # in emg_speech_dset_lengths we divide length by 8
            max_len=max_len // 8,
            always_include_class=[0],
        )
        num_workers = 10
        # num_workers = 0  # prob better now that we're caching
        bz = base_bz
        ValSampler = None
        TestSampler = None
        rank = 0

    if rank == 0:
        os.makedirs(output_directory, exist_ok=True)

    # must run cache_dataset_with_attrs_local_version.py first
    librispeech_train_cache = os.path.join(
        librispeech_directory, "librispeech-cache", "train_phoneme_cache"
    )
    librispeech_val_cache = os.path.join(
        librispeech_directory, "librispeech-cache", "val_phoneme_cache"
    )
    librispeech_test_cache = os.path.join(
        librispeech_directory, "librispeech-cache", "test_phoneme_cache"
    )

    speech_val = cache_dataset(
        librispeech_val_cache,
        LibrispeechDataset,
        per_index_cache,
        remove_attrs_before_save=["dataset"],
    )()
    speech_train = cache_dataset(
        librispeech_train_cache,
        LibrispeechDataset,
        per_index_cache,
        remove_attrs_before_save=["dataset"],
    )()
    speech_test = cache_dataset(
        librispeech_test_cache,
        LibrispeechDataset,
        per_index_cache,
        remove_attrs_before_save=["dataset"],
    )()

    datamodule = EMGAndSpeechModule(
        emg_datamodule.train,
        emg_datamodule.val,
        emg_datamodule.test,
        speech_train,
        speech_val,
        speech_test,
        bz=bz,
        val_bz=val_bz,
        num_replicas=NUM_GPUS,
        pin_memory=(not DEBUG),
        num_workers=num_workers,
        TrainBatchSampler=TrainBatchSampler,
        ValSampler=ValSampler,
        TestSampler=TestSampler,
        batch_class_proportions=batch_class_proportions,
    )

    steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum

    os.makedirs(output_directory, exist_ok=True)

    text_transform = TextTransform(togglePhones=togglePhones)
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1  # +1 for CTC blank token ( i think? )
    config = MONAConfig(
        steps_per_epoch=steps_per_epoch,
        lm_directory=lm_directory,
        num_outs=num_outs,
        precision=precision,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        audio_lambda=audio_lambda,
        emg_lambda=emg_lambda,
        # neural_input_features=datamodule.train.n_features,
        neural_input_features=1,
        seqlen=seqlen,
        max_len=max_len,
        batch_size=base_bz,
        white_noise_sd=white_noise_sd,
        constant_offset_sd=constant_offset_sd,
        num_train_epochs=n_epochs,
        togglePhones=togglePhones,
        use_dtw=use_dtw,
        use_crossCon=use_crossCon,
        use_supTcon=use_supTcon,
        batch_class_proportions=batch_class_proportions,
        # d_inner=8,
        # d_model=8,
        fixed_length=True,
        weight_decay=weight_decay,
        latent_affine=latent_affine,
    )
    configVM = MONAConfig(
        steps_per_epoch=steps_per_epoch,
        lm_directory=lm_directory,
        num_outs=num_outs,
        precision=precision,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        audio_lambda=audio_lambda,
        emg_lambda=emg_lambda,
        # neural_input_features=datamodule.train.n_features,
        neural_input_features=1,
        seqlen=seqlen,
        max_len=max_len,
        batch_size=base_bz,
        white_noise_sd=white_noise_sd,
        constant_offset_sd=constant_offset_sd,
        num_train_epochs=n_epochs,
        togglePhones=togglePhones,
        use_dtw=use_dtw,
        use_crossCon=use_crossCon,
        use_supTcon=use_supTcon,
        batch_class_proportions=batch_class_proportions,
        # d_inner=8,
        # d_model=8,
        fixed_length=True,
        weight_decay=weight_decay,
        latent_affine=latent_affine,
        num_heads=3,
        attn_layers=4,
        d_inner=1000,
    )

    model = MONA(configVM, text_transform, no_neural=True)

    ##
    logging.info("made model")

    callbacks = []

    monitor = "val/silent_emg_wer"
    save_top_k = 10

    neptune_logger = None

    trainer = pl.Trainer(
        max_epochs=configVM.num_train_epochs,
        devices='auto',
        accelerator="auto",
        accumulate_grad_batches=configVM.gradient_accumulation_steps,
        gradient_clip_val=1,  # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0
        logger=neptune_logger,
        default_root_dir=output_directory,
        callbacks=callbacks,
        precision=configVM.precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        sync_batchnorm=True,
        strategy=strategy,
        num_sanity_val_steps=0
    )

    num_threads = torch.get_num_threads()
    print(f"Number of threads: {num_threads}")

    ##
    logging.info("about to fit")
    print(f"Sanity check: {len(datamodule.train)} training samples")
    print(f"Sanity check: {len(datamodule.train_dataloader())} training batches")
    # epoch of 242 if only train...

    # Start to track memory:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()

    try:
        if MANUAL_RESUME:
            trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, datamodule=datamodule)
    except Exception as e:
        print(f"Error: {e}")
        print("Memory usage: {}".format(torch.cuda.memory_summary()))
    # End to track memory:
    print("Peak memory usage: {}".format(torch.cuda.max_memory_allocated()))

    torch.cuda.memory._dump_snapshot(f"f{RUN_ID}_snapshot.pickle")


