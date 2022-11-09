import argparse

from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate

wer = evaluate.load("wer")


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


def main(args):
    batch_size = args.batch_size

    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )
    whisper_asr.model.config.max_length = 128

    dataset = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True, use_auth_token=True
    )

    # Only uncomment for debugging
    dataset = dataset.take(64)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    preds = []
    refs = []

    result_set = dataset.map(
        evaluate,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.features.keys(),
    )

    for i, result in enumerate(iter(result_set)):
        ref = norm(result["ref"])
        pred = norm(result["pred"])

        refs.append(ref)
        preds.append(pred)

    wer_result = wer.compute(references=refs, predictions=preds)
    wer_result = round(100 * wer_result, 2)

    print(wer_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'`  for Common Voice",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`"
    )

    parser.add_argument(
        "--log_outputs",
        action="store_true",
        help="If defined, write outputs to log file for analysis.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    args = parser.parse_args()

    main(args)
