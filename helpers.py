import re, os, numpy as np, torch
from tqdm import tqdm
import torch.nn as nn
from data_utils import TextTransform
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA
from torchaudio.models.decoder import ctc_decoder
import functools
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import jiwer, json, asyncio
import neptune.new as neptune
from pytorch_lightning.loggers import NeptuneLogger
from time import sleep
from openai import OpenAI, AsyncOpenAI
from typing import List
from data_utils import in_notebook
from aiocache.serializers import PickleSerializer
from aiocache import cached


def sentence_to_fn(sentence, directory, ext=".wav"):
    fn = re.sub(r"[^\w\s]", "", sentence)  # remove punctuation
    fn = fn.lower().replace(" ", "_")  # lowercase with underscores
    return os.path.join(directory, fn + ext)


def string_to_np_array(string):
    """
    Convert a string representation of a numpy array into an actual numpy array.
    """
    try:
        # Remove square brackets and split the string by spaces
        elements = string.strip("[]").split()
        # Convert each element to float and create a numpy array
        return np.array([float(element) for element in elements])
    except Exception as e:
        print(f"Error converting string to numpy array: {e}")
        return None


def load_npz_to_memory(npz_path, **kwargs):
    npz = np.load(npz_path, **kwargs)
    loaded_data = {k: npz[k] for k in npz}
    npz.close()
    return loaded_data


def load_model(ckpt_path, config):
    text_transform = TextTransform(togglePhones=config.togglePhones)
    model = MONA(config, text_transform, no_neural=True)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def get_emg_pred(model, dataloader, key="raw_emg"):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch[key], batch_first=True)
            X = X.cuda()
            pred = model.emg_forward(X)[0].cpu()
            predictions.append((batch, pred))
    return predictions


def get_audio_pred(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch["audio_features"], batch_first=True)
            X = X.cuda()
            pred = model.audio_forward(X)[0].cpu()
            predictions.append((batch, pred))
    return predictions


# Function to run the beam search
def run_beam_search(
    batch_pred,
    text_transform,
    k,
    lm_weight,
    beam_size,
    beam_threshold,
    use_lm,
    togglePhones,
    lexicon_file,
    lm_file,
):
    batch, pred = batch_pred

    if use_lm:
        lm = lm_file
    else:
        lm = None

    decoder = ctc_decoder(
        lexicon=lexicon_file,
        tokens=text_transform.chars + ["_"],
        lm=lm,
        blank_token="_",
        sil_token="|",
        nbest=k,
        lm_weight=lm_weight,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
    )

    beam_results = decoder(pred)
    all_trl_top_k = []
    all_trl_beam_scores = []
    all_sentences = []
    for i, (example, beam_result) in enumerate(zip(batch, beam_results)):
        # Filter out silences
        target_sentence = text_transform.clean_text(batch["text"][i])
        if len(target_sentence) > 0:
            trl_top_k = []
            trl_beam_scores = []
            for beam in beam_result:
                transcript = " ".join(beam.words).strip().lower()
                score = beam.score
                trl_top_k.append(transcript)
                trl_beam_scores.append(score)

            all_trl_top_k.append(np.array(trl_top_k))
            all_trl_beam_scores.append(np.array(trl_beam_scores))
            all_sentences.append(target_sentence)
        else:
            all_trl_top_k.append(np.array([]))
            all_trl_beam_scores.append(np.array([]))
            all_sentences.append(target_sentence)

    return all_trl_top_k, all_trl_beam_scores, all_sentences


def get_top_k(
    predictions,
    text_transform,
    k: int = 100,
    beam_size: int = 500,
    togglePhones: bool = False,
    use_lm: bool = True,
    beam_threshold: int = 100,
    lm_weight: float = 2,
    cpus: int = 8,
    lexicon_file: str = None,
    lm_file: str = None,
):
    # Define the function to be used with concurrent.futures
    func = partial(
        run_beam_search,
        text_transform=text_transform,
        k=k,
        lm_weight=lm_weight,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
        use_lm=use_lm,
        togglePhones=togglePhones,
        lexicon_file=lexicon_file,
        lm_file=lm_file,
    )

    # If cpus=0, run without multiprocessing
    if cpus == 0:
        beam_results = [func(pred) for pred in tqdm(predictions)]
    else:
        # Use concurrent.futures for running the beam search with a progress bar
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            beam_results = list(
                tqdm(executor.map(func, predictions), total=len(predictions))
            )

    # flatten batched tuples of (all_trl_top_k, all_trl_beam_scores, all_sentences)
    # Separate and flatten the results
    all_trl_top_k, all_trl_beam_scores, all_sentences = [], [], []
    for trl_top_k, trl_beam_scores, sentences in beam_results:
        all_trl_top_k.extend(trl_top_k)
        all_trl_beam_scores.extend(trl_beam_scores)
        all_sentences.extend(sentences)

    # Collecting results
    topk_dict = {
        "k": k,
        "beam_size": beam_size,
        "beam_threshold": beam_threshold,
        "sentences": np.array(all_sentences),
        "predictions": np.array(all_trl_top_k, dtype=object),  # ragged array
        "beam_scores": np.array(all_trl_beam_scores, dtype=object),
    }

    return topk_dict


def calc_wer(predictions, targets, text_transform):
    """Calculate WER from predictions and targets.

    predictions: list of strings
    targets: list of strings
    """
    if type(predictions) is np.ndarray:
        predictions = list(map(str, predictions))
    if type(targets) is np.ndarray:
        targets = list(map(str, targets))
    targets = list(map(text_transform.clean_text, targets))
    predictions = list(map(text_transform.clean_text, predictions))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    # print(targets, predictions)
    return jiwer.wer(targets, predictions)


@functools.cache
def get_neptune_run(run_id, mode="read-only", **neptune_kwargs):
    return NeptuneLogger(
        run=neptune.init_run(
            with_id=run_id,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            mode=mode,
            **neptune_kwargs,
        ),
        log_model_checkpoints=False,
    )


def get_best_ckpts(directory, n=1):
    # get all files ending in .ckpt in subdirectories of directory
    ckpt_paths = []
    metrics = []
    # extract wer from eg "silent_emg_wer=0.253.ckpt"
    r = re.compile("wer=(0\.\d+)")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                try:
                    metrics.append(float(r.findall(file)[0]))
                    ckpt_paths.append(os.path.join(root, file))
                except IndexError:
                    pass
    perm = np.argsort(metrics)
    return [ckpt_paths[i] for i in perm[:n]], [metrics[i] for i in perm[:n]]


def get_last_ckpt(directory):
    """Get the most recent checkpoint for e.g. resuming a run."""
    ckpt_paths = []
    epochs = []

    # Regular expression to extract the epoch number from the directory name
    r = re.compile(r"epoch=(\d+)-")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                dir_name = os.path.basename(root)
                match = r.search(dir_name)
                if match:
                    epoch = int(match.group(1))
                    epochs.append(epoch)
                    ckpt_paths.append(os.path.join(root, file))

    # Find the index of the checkpoint with the highest epoch number
    if epochs:  # Ensure list is not empty
        max_epoch_idx = np.argmax(epochs)
        return ckpt_paths[max_epoch_idx], epochs[max_epoch_idx]
    else:
        raise ValueError("No checkpoints found in directory.")


def nep_get(logger, key):
    val_promise = logger.experiment.get_attribute(key)
    if hasattr(val_promise, "fetch"):
        return val_promise.fetch()
    elif hasattr(val_promise, "fetch_values"):
        return val_promise.fetch_values()
    else:
        raise NotImplementedError("don't know how to fetch values")


def load_model_from_id(choose="best", **kwargs):
    assert choose in ["best", "last"]

    if choose == "best":
        run_id = kwargs.get("run_id", "000")

        neptune_logger = NeptuneLogger(
            run=neptune.init_run(
                with_id=run_id,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                mode="read-only",
                project="neuro/Gaddy",
            ),
            log_model_checkpoints=False,
        )
        output_directory = nep_get(neptune_logger, "output_directory")
        hparams = nep_get(neptune_logger, "training/hyperparams")
    if choose == "best":
        ckpt_paths, wers = get_best_ckpts(output_directory, n=1)
        wer = wers[0]
        ckpt_path = ckpt_paths[0]
        min_wer = nep_get(neptune_logger, "training/val/wer").value.min()
        assert wer <= min_wer + 1e-3, f"wer {wer} > min_wer {min_wer}"
        if not np.isclose(wer, min_wer, atol=1e-3):
            print(f"WARNING: found checkpoint wer {wer} < min_wer {min_wer}")
        else:
            print("found checkpoint with WER", wer)
    elif choose == "last":
        ckpt_path = os.path.join(output_directory, "finished-training_epoch=200.ckpt")
        if os.path.exists(ckpt_path):
            epoch = 199
        else:
            ckpt_path, epoch = get_last_ckpt(output_directory)
        assert (
            epoch == hparams["num_train_epochs"] - 1
        ), f"epoch {epoch} != {hparams['num_train_epochs'] -1}"
        print("found checkpoint with epoch", epoch)
    togglePhones = hparams["togglePhones"]
    assert togglePhones == False, "not implemented"
    if "use_supCon" in hparams:
        hparams["use_supTcon"] = hparams["use_supCon"]
        del hparams["use_supCon"]
    config = MONAConfig(**hparams)

    model = load_model(ckpt_path, config)
    return model, config, output_directory


def get_run_type(hparams):
    m256 = hparams["max_len"] == 256000
    d = hparams["use_dtw"]
    c = hparams["use_crossCon"]
    if "use_supCon" in hparams:
        s = hparams["use_supCon"]
    elif "use_supTcon" in hparams:
        s = hparams["use_supTcon"]
    else:
        raise ValueError("unknown run type")
    a = hparams["audio_lambda"] == 1
    if "emg_lambda" in hparams:
        e = hparams["emg_lambda"] == 1
    else:
        e = True
    b = string_to_np_array(hparams["batch_class_proportions"])
    l = b[2] > 0  # use librispeech
    # balanced audio/emg sampling
    bal = np.isclose(0.1835, b[0], atol=1e-3) and np.isclose(0.1835, b[2], atol=1e-3)
    if m256 and c and l and bal:
        return "crossCon (balanced) 256k"
    elif m256 and c and d:
        return "crossCon + DTW 256k"
    elif m256 and c and l:
        return "crossCon 256k"
    elif m256 and c and not l:
        return "crossCon (no Librispeech) 256k"
    elif d and c and s:
        return "crossCon + supTcon + DTW"
    elif c and s:
        return "crossCon + supTcon"
    elif c:
        return "crossCon"
    elif s and d:
        return "supTcon + DTW"
    elif s:
        return "supTcon"
    elif a and e and l:
        return "EMG & Audio"
    elif a and e:
        return "EMG & Audio (no Librispeech)"
    elif e and not l:
        return "EMG (no Librispeech)"
    elif e:
        return "EMG 256k"
    elif a:
        return "Audio"
    else:
        raise ValueError(f"unknown run type for {hparams}")


##### LISA #####
def create_rescore_msg(predictions):
    rescore_msg = "\n".join([p for p in predictions])
    return rescore_msg


def completion_coroutine(client, sys_msg, user_msg, model="gpt-3.5-turbo-16k-0613"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        seed=20240130,
    )


async def gather_completions(coroutines, n_jobs=3):
    msgs = []
    for i in tqdm(range(0, len(coroutines), n_jobs), desc="Processing completions"):
        batch = coroutines[i : i + n_jobs]
        responses = await asyncio.gather(*batch)
        msgs.extend([response.choices[0].message.content for response in responses])
    return msgs


def batch_completions(client, predictions, sys_msg, n_jobs=3, model="gpt-3.5-turbo-16k-0613"):
    coroutines = []
    for pred in predictions:
        rescore_msg = create_rescore_msg(pred)
        cc = completion_coroutine(client, sys_msg, rescore_msg, model=model)
        coroutines.append(cc)
        sleep(0.05)
    # run the asynchronous gathering function
    return asyncio.run(gather_completions(coroutines, n_jobs=n_jobs))


DIRECT_SYS_MSG = """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription only, without any introductory text."""


def cor_clean_transcripts(transcripts, text_transform):
    ret = []
    for transcript in transcripts:
        # split on 'TRANSCRIPT: '
        t = transcript.split("TRANSCRIPT: ")[-1]
        # remove leading and trailing whitespace
        t = t.strip()
        ret.append(t)
    ret = list(map(text_transform.clean_text, ret))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    ret = transformation(ret)
    return ret


def direct_LISA(client, preds, labels, model, text_transform, N=10):
    assert len(preds) == len(labels), f"{len(preds)=} {len(labels)=}"
    lisa_predictions = batch_completions(client,
        [s[:N] for s in preds], DIRECT_SYS_MSG, model=model, n_jobs=5
    )

    try:
        lisa_wer = calc_wer(
            cor_clean_transcripts(lisa_predictions, text_transform), labels, text_transform
        )
        print(f"WER with direct {N=} ({model}): {lisa_wer * 100:.2f}%")
    except Exception as e:
        print(f"Error calculating WER: {e}")

    return lisa_predictions


def save_finetuning_dset(preds, labels, save_path):
    dset = [(create_rescore_msg(p), l) for p, l in zip(preds, labels)]

    # Convert to JSONL format
    jsonl_data = []
    for user_msg, assistant_msg in dset:
        jsonl_data.append(
            {
                "messages": [
                    {"role": "system", "content": DIRECT_SYS_MSG},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
        )

    # Save as a JSONL file
    jsonl_path = save_path
    with open(jsonl_path, "w") as f:
        for entry in jsonl_data:
            json.dump(entry, f)
            f.write("\n")

    return jsonl_path


# @cached(ttl=24 * 60 * 60, serializer=PickleSerializer())  # Cache results for 24 hours
@cached(serializer=PickleSerializer())
def get_transcript(client, wav_file, cache_seed=None, **kwargs):
    # with open(wav_file, "rb") as audio_file:
    audio_file = open(wav_file, "rb")  # for async
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, **kwargs
    )
    return transcript


async def gather_transcripts(coroutines, n_jobs=3):
    msgs = []
    for i in tqdm(range(0, len(coroutines), n_jobs), desc="Processing transcripts"):
        batch = coroutines[i : i + n_jobs]
        responses = await asyncio.gather(*batch)
        msgs.extend([response.text for response in responses])
    return msgs


def batch_transcripts(async_client, wav_files, temperatures: List[float], n_jobs=3):
    coroutines = []
    for wav_file in wav_files:
        for i,t in enumerate(temperatures):
            coroutines.append(get_transcript(async_client, wav_file,
                            cache_seed=(i,t), temperature=t))
    # run the asynchronous gathering function
    predictions = asyncio.run(gather_transcripts(coroutines, n_jobs=n_jobs))
    # reshape the predictions into a list of lists
    predictions = np.array(predictions).reshape(len(wav_files), len(temperatures))
    return predictions
