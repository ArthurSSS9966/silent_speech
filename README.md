## MONA LISA

This repository contains code for training Multimodal Orofacial Neural Audio (MONA) and Large Language
Model (LLM) Integrated Scoring Adjustment
(LISA). Together, MONA LISA sets a new state-of-the art for decoding silent speech, achieving 7.3% WER on validation data for open vocabulary.

[See the preprint on arxiv](https://arxiv.org/abs/2403.05583).

## Training Procedure
Download the lexicon.txt, tokens.txt, lm.binary using:
```
from torchaudio.models.decoder import download_pretrained_files
download_pretrained_files("librispeech-4-gram")
```
Then in the folder of 'librispeech-4-gram' you will find the files, copy and paste in the main folder
0) build the environment based on the below steps
1) run `cache_dataset_VMversion.py`
2) run `icml_models_VMversion.py`

[//]: # (3&#41; run `notebooks/tyler/2024-01-26_icml_pred.py`)

[//]: # (4&#41; run `notebooks/tyler/batch_beam_search.sh` &#40;`2024-01-26_icml_beams.py`&#41;)

[//]: # (5&#41; run `notebooks/tyler/2024-01-28_icml_figures.py`)

[//]: # (6&#41; run `notebooks/tyler/2024-01-31_icml_TEST.py`)

## Environment setup (In VM)

0. Make the data directory to store the data inside of the silent_speech folder:
```
cd silent_speech 
mkdir dataFolder
```
2. Run the `bash setup.sh`, it will install the necessary environment and download the data
3. Then you will need to download the [Gaddy 2020 dataset](https://doi.org/10.5281/zenodo.4064408) manually
4. Transfer the .tar.gz file to the VM using command:
`scp -i ~/.ssh/{key.perm} /path/to/file.tar.gz {user}@{localhost}:/path/to/destination`
5. Extract the .tar.gz file using command:
`tar -xzvf emg_data.tar.gz -C dataFolder`
6. Download the librispeech alignment data from [Montreal Force Alignment](https://drive.google.com/file/d/1OgfXbTYhgp8NW5fRTt_TXwViraOyVEyY/view) manually
7. Transfer the data using the same command
8. Extract the data using:
`unzip XXX.zip -d dataFolder`
9. Extract the text_alignments.tar.gz file inside the silent_speech_alignments folder to the dataFolder using:
```
cd silent_speech_alignments
tar -xzvf text_alignments.tar.gz -C dataFolder
```
10. Try to install dependencies. The requirements.txt may cause version issues.



## Explanation of model outputs for CTC loss
For each timestep, the network predicts probability of each of 38 characters ('abcdefghijklmnopqrstuvwxyz0123456789|_'), where `|` is word boundary, and `_` is the "blank token". The blank token is used to separate repeat letters like "ll" in hello: `[h,h,e,l,l,_,l,o]`. It can optionally be inserted elsewhere too, like `__hhhh_eeee_llll_lllooo___`

### Example prediction


**Target text**: after breakfast instead of working i decided to walk down towards the common

Example model prediction (argmax last dim) of shape `(1821, 38)`:

`______________________________________________________________a__f___tt__eerr|||b__rr_eaaakk___ff____aa____ss_tt___________________||____a_nd__|_ssttt___eaa_dd_||ooff||ww___o_rr_____kk_____ii___nngg________________________||_____a____t__||_______c______i___d_____eedd__________||tt___o__||_w_____a______l_kkk____________________||______o______w__t______________|||t____oowwwaarrrdddsss____||thhee_|||c_____o___mm__mm___oo_nn___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________`

Beam search gives, `after breakfast and stead of working at cided to walk owt towards the common`, which here is same as result from "best path decoding" (argmax), but in theory could be different since sums probability of multiple alignments and is therefore more accurate.

