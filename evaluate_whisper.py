from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm

batch_size = 16

whisper_asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
whisper_asr.model.config.max_length = 128

# missing Earnings22, AMI, voxpopuli under facebook
datasets = [
    ("librispeech_asr", "other", 2620),
    ("librispeech_asr", "clean", 2939),
    ("LIUM/tedlium", "release3", 1469),
    ("polinaeterna/voxpopuli", "en", 1842),
    ("speechcolab/gigaspeech", "l", 25619),
    ("kensho/spgispeech", "L", 39341),
    ("mozilla-foundation/common_voice_9_0", "en", 16335),
]


wer = evaluate.load("wer")
def is_valid(text):
    # TEDLIUM
    if text == "ignore_time_segment_in_scoring":
        return False
    if len(text) == 0:
        return False

    return True


def norm(text):
    return whisper_asr.tokenizer._normalize(text)


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    else:
        raise ValueError(f"Sample: {sample.keys()} has no transcript.")


def evaluate(batch):
    audios = [a["array"] for a in batch["audio"]]

    predictions = whisper_asr(audios)
    batch["ref"] = get_text(batch)
    batch["pred"] = [p["text"] for p in predictions]

    return batch


for name, config, length in datasets:
    print(f"Evaluating {name} - {config}")
    dataset = load_dataset(name, config, split="test", streaming=True, use_auth_token=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    preds = []
    refs = []

    result_set = dataset.map(evaluate, batched=True, batch_size=batch_size, remove_columns=dataset.features.keys())

    with tqdm(total=length) as progress_bar:
        for i, result in enumerate(iter(result_set)):
            ref = norm(result["ref"])
            pred = norm(result["pred"])

            if is_valid(ref):
                refs.append(ref)l
                preds.append(pred)

            progress_bar.update(1)

    wer_result = wer.compute(references=refs, predictions=preds)
    wer_result = round(100 * wer_result, 2)