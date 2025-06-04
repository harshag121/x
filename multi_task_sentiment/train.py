"""CLI entry point for training multi-task sentiment model."""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)
import evaluate
from .data import get_datasets


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset = get_datasets(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./multi_task_sentiment/checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./multi_task_sentiment/logs",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./multi_task_sentiment/final_model")


if __name__ == "__main__":
    main()
