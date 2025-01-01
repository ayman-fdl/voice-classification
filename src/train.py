from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from src.data_preprocessing import load_data, split_data
from src.feature_extraction import extract_features

def train_models():
    # Load and preprocess data
    audio_data, labels = load_data('data/processed')
    X_train, X_test, y_train, y_test = split_data(audio_data, labels)

    # Feature extraction
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Initialize models
    cnn_model = CNNModel()
    lstm_model = LSTMModel()
    transformer_model = TransformerModel()

    # Build and train CNN model
    cnn_model.build_model()
    cnn_model.train(X_train_features, y_train)

    # Build and train LSTM model
    lstm_model.build_model()
    lstm_model.train(X_train_features, y_train)

    # Build and train Transformer model
    transformer_model.build_model()
    transformer_model.train(X_train_features, y_train)

if __name__ == "__main__":
    train_models()