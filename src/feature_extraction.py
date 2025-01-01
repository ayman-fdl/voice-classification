import numpy as np
import librosa
import os
import pandas as pd

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def save_features(features, labels, output_file):
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_file, index=False)