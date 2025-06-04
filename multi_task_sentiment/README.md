# Multi-Task Sentiment Training

This package demonstrates training a single Transformer model on multiple
sentiment analysis datasets. It uses the IMDB and Yelp Polarity datasets
via the `datasets` library and fine-tunes `distilbert-base-uncased` across
both tasks.

## Installation

Create a virtual environment and install the package with its dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script from the project root:

```bash
python -m multi_task_sentiment.train
```

The script will download the datasets and start fine-tuning the model. Training
arguments can be modified in `multi_task_sentiment/train.py`.
