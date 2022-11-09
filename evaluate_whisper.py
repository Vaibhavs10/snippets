import argparse
from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate

def norm(text):
    return whisper_asr.tokenizer._normalize(text)

def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""

    log_outputs = args.log_outputs
    dataset_id = "_".join(args.dataset.split("/") + [args.config, args.split])

    # load metric
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    # compute metrics
    wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"WER: {wer_result}\nCER: {cer_result}"
    print(result_str)

    with open(f"{dataset_id}_eval_results.txt", "w") as f:
        f.write(result_str)

    # log all results in text file. Possibly interesting for analysis
    if log_outputs is not None:
        pred_file = f"log_{dataset_id}_predictions.txt"
        target_file = f"log_{dataset_id}_targets.txt"

        with open(pred_file, "w") as p, open(target_file, "w") as t:

            # mapping function to write output
            def write_to_file(batch, i):
                p.write(f"{i}" + "\n")
                p.write(batch["prediction"] + "\n")
                t.write(f"{i}" + "\n")
                t.write(batch["target"] + "\n")

            result.map(write_to_file, with_indices=True)


def main(args):
    # load dataset
    dataset = load_dataset(args.dataset, args.config, streaming=True, split=args.split, use_auth_token=True)

    # for testing: only process the first two examples as a test
    # dataset = dataset.select(range(10))

    # load processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    sampling_rate = feature_extractor.sampling_rate

    # resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    if args.device is None:
        args.device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=args.model_id, device=args.device)

    # map function to decode audio
    def map_to_pred(batch):
        prediction = asr(
            batch["audio"]["array"]
        )

        batch["prediction"] = prediction["text"]
        batch["target"] = norm(batch["sentence"])
        return batch

    # run inference on all examples
    result = dataset.map(map_to_pred, remove_columns=dataset.column_names)

    # compute and log_results
    # do not change function below
    log_results(result, args)

    if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config of the dataset. *E.g.* `'en'`  for Common Voice"
    )
    parser.add_argument("--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`")

    parser.add_argument(
        "--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    args = parser.parse_args()

    main(args)