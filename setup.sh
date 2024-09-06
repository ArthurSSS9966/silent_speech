# Step 1: Install environment using conda
conda env create --file environment.yml

conda activate silent_speech

pip install -r requirements.txt

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install required Python packages using pip
pip install jiwer matplotlib scipy soundfile absl-py librosa numba unidecode praat-textgrids g2p_en einops opt_einsum hydra-core pytorch_lightning "neptune-client==0.16.18"

# Git Clone the EMG alignments file
git clone https://github.com/dgaddy/silent_speech_alignments.git



