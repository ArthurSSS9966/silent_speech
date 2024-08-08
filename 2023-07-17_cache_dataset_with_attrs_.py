import os, pickle, sys, torch

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from dataloaders import LibrispeechDataset, cache_dataset
from datasets import load_dataset

##
print("Loading datasets...")
librispeech_datasets = load_dataset("librispeech_asr", cache_dir="D:/Blcdata/hf_cache")

librispeech_train = torch.utils.data.ConcatDataset(
    [
        librispeech_datasets["train.clean.100"],
        librispeech_datasets["train.clean.360"],
        librispeech_datasets["train.other.500"],
    ]
)
librispeech_clean_val = librispeech_datasets["validation.clean"]
librispeech_clean_test = librispeech_datasets["test.clean"]

normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
mfcc_norm, emg_norm = pickle.load(open(normalizers_file, "rb"))
text_transform = TextTransform(togglePhones=False)

##
scratch_directory = "D:\\BlcRepo\\OtherCode\\Generative_Neuroscience\\silent_speech\\scratch"
librispeech_directory = "D:/Blcdata/hf_cache"

librispeech_train_cache = os.path.join(
    librispeech_directory, "librispeech-cache", "train_phoneme_cache"
)
librispeech_val_cache = os.path.join(
    librispeech_directory, "librispeech-cache", "val_phoneme_cache"
)
librispeech_test_cache = os.path.join(
    librispeech_directory, "librispeech-cache", "test_phoneme_cache"
)
alignment_dir = os.path.join(librispeech_directory, "librispeech-alignments")

##
alignment_dirs = [os.path.join(alignment_dir, d) for d in os.listdir(alignment_dir)]
per_index_cache = True

print("Caching Val...")

cached_speech_val = cache_dataset(
    librispeech_val_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
    # resolve https://github.com/guyhwilson/silent_speech/issues/5
    # other has leak for War of the Worlds
)(
    librispeech_clean_val,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "dev" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_val
print("Done")

print("Caching Train...")
cached_speech_train = cache_dataset(
    librispeech_train_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)(
    librispeech_train,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "train" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_train
print("Done")

print("Caching Test...")
cached_speech_test = cache_dataset(
    librispeech_test_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)(
    librispeech_clean_test,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "test" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_test
