# Sentiment Analysis using Bidirectional LSTM

This repository contains a sentiment analysis implementation using a Bidirectional Long Short-Term Memory (BiLSTM) neural network architecture. The model achieves 88% test accuracy on the dataset.

## Repository Structure

```
sentiment-analysis-bilstm/
├── dataset/
│   └── README.md          # Dataset description
├── notebook/
│   └── sentiment_analysis.ipynb  # Implementation notebook
└── visualization/
    └── visualization.py   # Data visualization scripts
```

## Approach

The sentiment analysis task is implemented using the following approach:

### 1. Data Preprocessing

- **Data Cleaning**:

  - Removing stop words (common words that don't add much meaning)
  - Removing punctuation marks
  - Removing HTML tags and special characters
  - Text normalization (lowercasing, etc.)

- **Tokenization**:

  - Converting text into tokens/words
  - Creating a vocabulary from the dataset

- **Padding**:
  - Ensuring all sequences have the same length for batch processing

### 2. Word Embedding

- Utilized GloVe (Global Vectors for Word Representation) embedding model
- Pre-trained word vectors are used to represent words in a meaningful vector space
- Words with similar meanings have similar vector representations

### 3. Model Architecture

- **Bidirectional LSTM** neural network:
  - Processes sequences in both forward and backward directions
  - Captures context from both past and future tokens
  - Effective for understanding sentiment context in sentences
  - Better understanding of long-range dependencies compared to traditional LSTMs

### 4. Training and Evaluation

- Model was trained on the provided dataset
- Achieved **88% accuracy** on the test set
- Evaluation metrics include accuracy, precision, recall, and F1-score

## Requirements

```
tensorflow>=2.0.0
numpy
pandas
matplotlib
scikit-learn
nltk
gensim
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/username/sentiment-analysis-bilstm.git
cd sentiment-analysis-bilstm
```

2. For visualizations:

```bash
python visualization/visualization.py
```

3. Run the Google Colab notebook:

```bash
!git clone https://github.com/roshan-rs-git/_IMDB_dataset_sentiment_Bidirectional_LSTM.ipynb
```

## Results

The bidirectional LSTM model achieves 88% accuracy on the sentiment analysis task. Detailed performance metrics and visualizations can be found in the notebook and visualization directories.

## Future Improvements

- Experiment with different word embedding techniques
- Try attention mechanisms to improve model performance
- Implement hyperparameter tuning
- Test on different datasets to evaluate generalization

## License

[MIT License](LICENSE)

## Acknowledgements

- GloVe embedding model by Stanford NLP
- TensorFlow and Keras for deep learning implementation
