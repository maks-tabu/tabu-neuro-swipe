from transformers import (
    default_data_collator,
    Seq2SeqTrainer,
    DeiTImageProcessor,
    Seq2SeqTrainingArguments,
)
from datasets import load_metric
from src.model import TrOcrModel, TrOcrDataset
from src.constants import DATA_PATH
import yaml
import argparse
import polars as pl


def train(config):
    trocr_model = TrOcrModel(config["PRETRAINING_WEIGHT"], config["TOKENIZER"])

    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = trocr_model.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        labels_ids[labels_ids == -100] = trocr_model.tokenizer.pad_token_id
        label_str = trocr_model.tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True
        )

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    val_dataset = pl.read_parquet(DATA_PATH / "val.pa", columns=["path", "target"])
    val_dataset = TrOcrDataset(
        val_dataset, DeiTImageProcessor({}), trocr_model.tokenizer
    )

    for i, name_dataset in enumerate(config["TRAIN_DATASETS"]):
        if i == 0:
            train_dataset = pl.read_parquet(DATA_PATH / f"{name_dataset}.pa", columns=["path", "target"])
        else:
            train_dataset = pl.concat(
                [train_dataset, pl.read_parquet(DATA_PATH / f"{name_dataset}.pa", columns=["path", "target"])],
                how="vertical",
            )
    train_dataset = TrOcrDataset(
        train_dataset, DeiTImageProcessor({}), trocr_model.tokenizer
    )

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=config["EPOCH"],
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=config["LR"],
        fp16=True,
        output_dir=".",
        logging_steps=2,
        save_steps=10000,
        eval_steps=10000,
        save_total_limit=1,
    )

    trainer = Seq2SeqTrainer(
        model=trocr_model.model,
        tokenizer=trocr_model.tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trocr_model.save_model(config["NAME_WEIGHT"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stage", type=int, required=True)

    with open("src/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args, _ = parser.parse_known_args()
    train(config[f"train_part{args.n_stage}"])
