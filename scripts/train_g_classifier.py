#!/usr/bin/env python3
# train_g_classifier.py

import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.image import resize
from sklearn.model_selection import train_test_split
import json

DATA_DIR = os.path.join('..', 'data', 'audio')
CLASSES = ['blues','classical','country','disco','hiphop',
           'jazz','metal','pop','reggae','rock']

def load_and_preprocess(data_dir, classes, target_shape=(150,150)):
    X, y = [], []
    for idx, genre in enumerate(classes):
        gd = os.path.join(data_dir, genre)
        for f in os.listdir(gd):
            if not f.endswith('.wav'): continue
            audio, sr = librosa.load(os.path.join(gd, f), sr=None)
            cd, od = 4, 2
            cs, osamp = cd*sr, od*sr
            n = int(np.ceil((len(audio)-cs)/(cs-osamp))) + 1
            for i in range(n):
                s = i*(cs-osamp); e = s+cs
                chunk = audio[s:e]
                mel = librosa.feature.melspectrogram(y=chunk, sr=sr)
                mel = resize(np.expand_dims(mel, -1), target_shape)
                X.append(mel); y.append(idx)
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    return X, y

def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        Conv2D(32,3,padding='same',activation='relu', input_shape=input_shape),
        Conv2D(32,3,activation='relu'),
        MaxPool2D(2,2),
        Conv2D(64,3,padding='same',activation='relu'),
        Conv2D(64,3,activation='relu'),
        MaxPool2D(2,2),
        Conv2D(128,3,padding='same',activation='relu'),
        Conv2D(128,3,activation='relu'),
        MaxPool2D(2,2),
        Dropout(0.3),
        Conv2D(256,3,padding='same',activation='relu'),
        Conv2D(256,3,activation='relu'),
        MaxPool2D(2,2),
        Conv2D(512,3,padding='same',activation='relu'),
        Conv2D(512,3,activation='relu'),
        MaxPool2D(2,2),
        Dropout(0.3),
        Flatten(),
        Dense(1200, activation='relu'),
        Dropout(0.45),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    X, y = load_and_preprocess(DATA_DIR, CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model(X_train[0].shape, len(CLASSES))
    history = model.fit(
        X_train, y_train,
        epochs=30, batch_size=32,
        validation_data=(X_test, y_test)
    )
    os.makedirs(os.path.join('..','models'), exist_ok=True)
    model.save(os.path.join('..','models','Trained_model.h5'))
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    main()