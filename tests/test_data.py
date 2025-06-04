from multi_task_sentiment.data import get_datasets
from transformers import AutoTokenizer


def test_dataset_loading():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train, eval_ds = get_datasets(tokenizer)
    assert len(train) > 0
    assert len(eval_ds) > 0
