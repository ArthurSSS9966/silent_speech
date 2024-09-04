#!/bin/bash
source venv/bin/activate

# Step 1: Install environment using conda
conda env update --file environment.yml --prune

conda activate silent_speech

# Step 2: Install required Python packages using pip
pip install jiwer torchaudio matplotlib scipy soundfile absl-py librosa numba unidecode praat-textgrids g2p_en einops opt_einsum hydra-core pytorch_lightning "neptune-client==0.16.18"

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Git Clone the EMG alignments file
git clone https://github.com/dgaddy/silent_speech_alignments.git



