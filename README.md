# Voice Classification Project

This project implements voice classification using Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Transformer models. The goal is to classify audio data based on various voice features.

## Project Structure

```
voice-classification-project
├── data
│   ├── raw                # Contains raw audio files for voice classification
│   └── processed          # Stores processed audio data ready for model training
├── models
│   ├── cnn_model.py       # Implements CNN model for voice classification
│   ├── lstm_model.py      # Implements LSTM model for voice classification
│   └── transformer_model.py # Implements Transformer model for voice classification
├── notebooks
│   └── exploratory_data_analysis.ipynb # Jupyter notebook for exploratory data analysis
├── src
│   ├── streamlit_app.py       # Streamlit app for real-time audio testing
│   ├── data_preprocessing.py  # Functions for preprocessing audio data
│   ├── feature_extraction.py  # Functions for extracting features from audio data
│   ├── train.py               # Responsible for training the models
│   └── evaluate.py            # Functions for evaluating the trained models
├── requirements.txt           # Lists the dependencies required for the project
└── README.md                  # Documentation for the project
```

## Requirements
- Python 3.10

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd voice-classification-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Place your raw audio files in the `data/raw` directory.
- Run the preprocessing script to prepare the data:
  ```
  python src/data_preprocessing.py
  ```
- Extract features from the processed audio data:
  ```
  python src/feature_extraction.py
  ```
- Train the models:
  ```
  python src/train.py
  ```
- Evaluate the models:
  ```
  python src/evaluate.py
  ```

## Additional Information

Refer to the Jupyter notebook in the `notebooks` directory for exploratory data analysis and insights on the dataset. Each model implementation in the `models` directory contains methods for building and training the respective models.