def load_data(data_dir):
    import os
    import librosa
    import numpy as np

    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    audio_data = []

    for file in audio_files:
        file_path = os.path.join(data_dir, file)
        signal, sr = librosa.load(file_path, sr=None)
        audio_data.append((signal, sr, file))

    return audio_data

def normalize_data(audio_data):
    normalized_data = []
    
    for signal, sr, file in audio_data:
        normalized_signal = signal / np.max(np.abs(signal))
        normalized_data.append((normalized_signal, sr, file))
    
    return normalized_data

def split_data(audio_data, test_size=0.2):
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(audio_data, test_size=test_size, random_state=42)
    return train_data, test_data