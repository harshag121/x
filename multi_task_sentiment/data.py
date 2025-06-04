"""Dataset loading utilities for multi-task sentiment analysis."""

from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def tokenize_function(examples, tokenizer: PreTrainedTokenizer):
    return tokenizer(examples["text"], truncation=True)


def get_datasets(tokenizer: PreTrainedTokenizer):
    """Load IMDB and Yelp Polarity datasets and tokenize them."""
    imdb = load_dataset("imdb")
    yelp = load_dataset("yelp_polarity")

    def prep(dataset, name):
        ds = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        ds = ds.rename_column("label", "labels")
        ds = ds.remove_columns(["text"])
        ds = ds.map(lambda x: {"task": name})
        return ds

    imdb = prep(imdb, "imdb")
    yelp = prep(yelp, "yelp")

    train_dataset = concatenate_datasets([imdb["train"], yelp["train"]])
    eval_dataset = concatenate_datasets([imdb["test"], yelp["test"]])

    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.shuffle(seed=42)

    return train_dataset, eval_dataset
