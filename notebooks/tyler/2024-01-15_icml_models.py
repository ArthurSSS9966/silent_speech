import os
import pytorch_lightning as pl, pickle
import sys
import numpy as np
import logging
import torch
from torch.utils.data import DistributedSampler

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import (
    EMGDataModule
)
from architecture import MONAConfig, MONA
import typer
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
DEBUG = True

if DEBUG:
    RUN_ID = "debug"
else:
    RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

torch.set_float32_matmul_precision("high")  # highest (32-bit) by default

torch.backends.cudnn.allow_tf32 = True  # should be True by default
run_id = ""
ckpt_path = ""

per_index_cache = True  # read each index from disk separately

from torchaudio.models.decoder import download_pretrained_files
download_pretrained_files("librispeech")

isotime = datetime.now().isoformat()

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2  # will not run on_validation_epoch_end
    # NUM_GPUS = 2
    # limit_train_batches = None
    # limit_val_batches = None
    log_neptune = False
    n_epochs = 2
    # n_epochs = 200
    # precision = "32"
    # precision = "16-mixed"
    precision = "bf16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""

# on my local machine
scratch_directory = "D:\\BlcRepo\\OtherCode\\Generative_Neuroscience\\silent_speech\\scratch"
librispeech_directory = "D:/Blcdata/hf_cache"
lm_directory = "D:\\BlcRepo\\OtherCode\\Generative_Neuroscience\\silent_speech"
SLURM_REQUEUE = False

data_dir = scratch_directory
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
base_bz = 24 * 4
val_bz = 2  # terrible memory usage even at 8, I'm not sure why so bad...
# gaddy used max_len = 128000, we double because of LibriSpeech
# TODO: try 512000 and grad_accum=1 (prob OOM but might be faster!)
# also almost def won't work for supTcon + dtw
# max_len = 48000 # from best perf with 4 x V100
max_len = 128000  #

##

app = typer.Typer()

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
batch_class_proportions = np.array([frac_semg, frac_vocal, frac_librispeech])
latest_epoch = -1
matmul_tf32 = True


@app.command()
def update_configs(
    constant_offset_sd_cli: float = typer.Option(0, "--constant-offset-sd"),
    white_noise_sd_cli: float = typer.Option(0, "--white-noise-sd"),
    learning_rate_cli: float = typer.Option(3e-4, "--learning-rate"),
    debug_cli: bool = typer.Option(False, "--debug/--no-debug"),
    phonemes_cli: bool = typer.Option(False, "--phonemes/--no-phonemes"),
    use_dtw_cli: bool = typer.Option(use_dtw, "--dtw/--no-dtw"),
    use_crossCon_cli: bool = typer.Option(use_crossCon, "--crossCon/--no-crossCon"),
    use_supTcon_cli: bool = typer.Option(use_supTcon, "--supTcon/--no-supTcon"),
    grad_accum_cli: int = typer.Option(grad_accum, "--grad-accum"),
    precision_cli: str = typer.Option(precision, "--precision"),
    logger_level_cli: str = typer.Option("WARNING", "--logger-level"),
    base_bz_cli: int = typer.Option(base_bz, "--base-bz"),
    val_bz_cli: int = typer.Option(val_bz, "--val-bz"),
    max_len_cli: int = typer.Option(max_len, "--max-len"),
    seqlen_cli: int = typer.Option(seqlen, "--seqlen"),
    n_epochs_cli: int = typer.Option(n_epochs, "--n-epochs"),
    run_id_cli: str = typer.Option(run_id, "--run-id"),
    ckpt_path_cli: str = typer.Option(ckpt_path, "--ckpt-path"),
    audio_lambda_cli: float = typer.Option(audio_lambda, "--audio-lambda"),
    emg_lambda_cli: float = typer.Option(emg_lambda, "--emg-lambda"),
    frac_semg_cli: float = typer.Option(frac_semg, "--frac-semg"),
    frac_vocal_cli: float = typer.Option(frac_vocal, "--frac-vocal"),
    frac_librispeech_cli: float = typer.Option(frac_librispeech, "--frac-librispeech"),
    weight_decay_cli: float = typer.Option(weight_decay, "--weight-decay"),
    matmul_tf32_cli: bool = typer.Option(matmul_tf32, "--matmul-tf32/--no-matmul-tf32"),
    latent_affine_cli: bool = typer.Option(
        latent_affine, "--latent-affine/--no-latent-affine"
    ),
    # devices_cli: str = typer.Option(devices, "--devices"),
):
    """Update configurations with command-line values."""
    global constant_offset_sd, white_noise_sd, DEBUG, grad_accum, matmul_tf32
    global precision, logger_level, base_bz, val_bz, max_len, seqlen, n_epochs
    global learning_rate, devices, togglePhones, use_dtw, use_crossCon, use_supTcon
    global audio_lambda, latent_affine, weight_decay, run_id, ckpt_path, latest_epoch
    global emg_lambda, frac_semg, frac_vocal, frac_librispeech, batch_class_proportions

    # devices = devices_cli
    # try:
    #     devices = int(devices) # eg "2" -> 2
    # except:
    #     pass
    use_dtw = use_dtw_cli
    use_crossCon = use_crossCon_cli
    use_supTcon = use_supTcon_cli
    togglePhones = phonemes_cli
    learning_rate = learning_rate_cli
    constant_offset_sd = constant_offset_sd_cli
    white_noise_sd = white_noise_sd_cli
    DEBUG = debug_cli
    run_id = run_id_cli
    grad_accum = grad_accum_cli
    precision = precision_cli
    logger_level = getattr(logging, logger_level_cli.upper())
    base_bz = base_bz_cli
    val_bz = val_bz_cli
    max_len = max_len_cli
    seqlen = seqlen_cli
    audio_lambda = audio_lambda_cli
    emg_lambda = emg_lambda_cli
    latent_affine = latent_affine_cli
    weight_decay = weight_decay_cli
    ckpt_path = ckpt_path_cli
    n_epochs = n_epochs_cli
    matmul_tf32 = matmul_tf32_cli

    if (
        frac_semg != frac_semg_cli
        or frac_vocal != frac_vocal_cli
        or frac_librispeech != frac_librispeech_cli
    ):
        batch_class_proportions = np.array(
            [frac_semg_cli, frac_vocal_cli, frac_librispeech_cli]
        )
        print(f"batch_class_proportions: {batch_class_proportions}")

    print("Updated configurations using command-line arguments.")


torch.backends.cuda.matmul.allow_tf32 = matmul_tf32  # false by default

# try:
# app()
# except SystemExit as e:
#     pass

if ckpt_path != "":
    raise NotImplementedError("TODO: implement output_directory for ckpt_path")


MANUAL_RESUME = False
output_directory = os.path.join(scratch_directory, f"output_arthur{RUN_ID}")

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

logging.debug("DEBUG mode")
##
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
    # num_workers=32
    num_workers = 0  # prob better now that we're caching
    bz = base_bz
    ValSampler = None
    TestSampler = None
    rank = 0

if rank == 0:
    os.makedirs(output_directory, exist_ok=True)


# must run 2023-07-17_cache_dataset_with_attrs_.py first
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

# speech_val = speech_val[:len(emg_datamodule.val)]
# speech_train = speech_train[:len(emg_datamodule.train)]
# speech_test = speech_test[:len(emg_datamodule.test)]

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

model = MONA(config, text_transform, no_neural=True)

##
logging.info("made model")

callbacks = []

monitor = "val/silent_emg_wer"
save_top_k = 10

neptune_logger = None

trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    accumulate_grad_batches=config.gradient_accumulation_steps,
    gradient_clip_val=1,  # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    sync_batchnorm=True,
    strategy=strategy,
    num_sanity_val_steps=0
)

##
logging.info("about to fit")
print(f"Sanity check: {len(datamodule.train)} training samples")
print(f"Sanity check: {len(datamodule.train_dataloader())} training batches")
# epoch of 242 if only train...
if MANUAL_RESUME or SLURM_REQUEUE:
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)


