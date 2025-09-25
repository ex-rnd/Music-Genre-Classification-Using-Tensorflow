#!/usr/bin/env python3

import os
# 1) Kill the C++ backend logs (INFO, WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2) Silence absl (the Python wrapper that TensorFlow uses)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


import sys
import argparse
import logging
import numpy as np
import librosa

# 2. Import TensorFlow and drop its Python-side warnings  
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.image import resize

# 3. Default constants
DEFAULT_MODEL_PATH = r"C:\Users\Administrator\SoulfulSpixah\music-genre-classification\models\Trained_model.h5"
CLASSES = [
    'blues','classical','country','disco','hiphop',
    'jazz','metal','pop','reggae','rock'
]

def preprocess_file(filepath, target_shape=(150,150)):
    audio, sr = librosa.load(filepath, sr=None)
    desired = 4 * sr
    if len(audio) < desired:
        audio = np.pad(audio, (0, desired - len(audio)), mode='constant')
    else:
        audio = audio[:desired]
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel = resize(np.expand_dims(mel, -1), target_shape)
    return np.expand_dims(mel, axis=0)

def main():
    parser = argparse.ArgumentParser(
        description="Predict music genre from a WAV file."
    )
    parser.add_argument(
        '-f', '--file',
        required=True,
        help="Path to the input WAV file"
    )
    parser.add_argument(
        '-m', '--model',
        default=DEFAULT_MODEL_PATH,
        help="Path to the .h5 Keras model"
    )
    args = parser.parse_args()

    # Load model and run prediction
    model = tf.keras.models.load_model(args.model)
    X = preprocess_file(args.file)
    pred = model.predict(X)
    idx = np.argmax(pred, axis=1)[0]
    print(f"Predicted genre: {CLASSES[idx]} (confidence: {pred[0][idx]:.3f})")

if __name__ == "__main__":
    main()



# #
# python .\predict_g_classifier.py -f ..\data\audio\rock\rock.00000.wav -m ..\models\Trained_model.h5
# #    