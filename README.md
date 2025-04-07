# Sentiment Analysis with BERT on Sentiment140

This project demonstrates how to fine-tune a BERT model (using `bert-base-uncased`) for sentiment classification on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The dataset contains 1.6 million tweets labeled as positive or negative, and the project leverages the Hugging Face Transformers library with PyTorch.

## Project Overview

The repository is organized into three main Python scripts:
- **`preprocess.py`**: Loads the CSV dataset, cleans the tweet texts (removes URLs, user mentions, etc.), splits the data into training and validation sets, and tokenizes the text using BERT's tokenizer.
- **`dataset.py`**: Contains the custom PyTorch `SentimentDataset` class that wraps tokenized data and labels.
- **`train.py`**: Integrates the preprocessing, creates DataLoaders, initializes the BERT model for sequence classification, and runs the training and evaluation loops. The final fine-tuned model is then saved for later use.

## Data File

Not uploaded. You may use it directly from Kagge:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kazanova/sentiment140")

print("Path to dataset files:", path)
```

You may also download it to your local device or a remote server. BUT IT CAN BE VERY LARGE!!!

## Requirements

Ensure you have Python installed (Python 3.7 or later is recommended) and install the required packages:

```bash
pip install kagglehub transformers torch scikit-learn pandas tqdm
```

## Usage

### Download the Dataset:

The script uses training.1600000.processed.noemoticon.csv in a folder called data. You may also use the online Kaggle platform

### Adjust Batch Size:

In the training script (train.py), you can modify the batch size by editing the DataLoader instantiation:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

For actual use, choose a batch size of 128 if your GPU memory permits. For example:

```python
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
```

### Handling Padding:

This project uses BERT's fast tokenizer, which handles padding tokens ([PAD]) automatically. Similar to models like Meta's LLaMA, it is important that the padding is correctly managed in the inputs:

The tokenizer returns input_ids and attention_mask, where padded tokens are set to a specific pad token (usually 0 for BERT).

These masks are used by the model to ignore padded tokens during attention calculations.

The code in preprocess.py includes:

```python
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
This ensures that all sequences are padded/truncated to a uniform length and that the model correctly processes them.
```

### Run the Training Script:

To run the complete training pipeline, open your terminal in the project directory and execute:

```bash
python train.py
```

If you use Python 3, change it to:

```bash
python3 train.py
```

If you are using GPU, you may have to do something like:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py
```

This will:
1. Download and load the dataset.
2. Preprocess and tokenize the data.
3. Create training and validation DataLoaders.
4. Fine-tune the BERT model on the Sentiment140 dataset.
5. Save the fine-tuned model in the sentiment140-bert-model folder.

## Notes
### Batch Size:
While the default example uses a batch size of 32, for production or more intensive training scenarios, a batch size of 128 can be used if GPU resources allow. Adjust the batch_size parameter in DataLoader accordingly.

### Padding Tokens:
Proper handling of padding is crucial. The fast tokenizer ensures that sequences are padded with [PAD] tokens and provides an attention mask to prevent these tokens from affecting the model's computations. This approach is consistent with best practices in modern transformer models, including META's LLaMA.

### Hardware Requirements:
Fine-tuning on a large dataset like Sentiment140 may require significant computational resources. Using a GPU is highly recommended for training efficiency.

### Customization:
Feel free to adjust hyperparameters such as learning rate, number of epochs, and gradient accumulation steps (if needed) in train.py to better suit your environment and training goals.

## License
This project is provided for educational purposes.

## Acknowledgments
Hugging Face Transformers
Sentiment140 Dataset on Kaggle
