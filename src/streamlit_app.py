import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa

# Load your trained model
model = load_model('path_to_your_model.h5')

# Function to preprocess audio
def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs, axis=0)

# Streamlit app
st.title('Real-time Audio Classification')
st.write('Upload an audio file or use your microphone to test the model in real-time.')

# Upload audio file
uploaded_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    features = preprocess_audio(uploaded_file)
    prediction = model.predict(features)
    st.write(f'Prediction: {np.argmax(prediction)}')

# Real-time audio recording (requires additional setup)
if st.button('Record Audio'):
    st.write('Recording...')
    # Implement real-time audio recording and prediction here
    st.write('Recording complete.')