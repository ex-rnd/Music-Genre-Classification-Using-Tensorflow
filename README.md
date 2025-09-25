# 🎶 Music Genre Classification
A deep-learning project that classifies short audio clips into one of ten music genres using a CNN-based spectrogram model.

## ✳️ Visual Overview 

https://github.com/user-attachments/assets/78198f10-31bc-4046-b07e-18e835c2fb56


## 📹 Demo
First, see it in action:
1. Open a terminal.
2. Run the prediction script on a WAV file
```bash
python scripts/predict_g_classifier.py \
  -f data/audio/rock/rock.00000.wav \
  -m models/Trained_model.h5
```
3. Observe the predicted genre and confidence score


## 🔍 Project Overview

### Problem
- Automatically classify music clips into one of ten genres `(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)`.

### Key Components
-	Feature extraction: Mel spectrograms from 4-second audio chunks
-	Model: Deep CNN with stacked Conv2D → MaxPool → Dropout → Dense layers
- Training: 80/20 train/test split using 
- Evaluation: Accuracy metrics and training history saved to JSON


## 🛠️ Getting Started
1. Clone
```bash
git clone https://github.com/ex-rnd/Music-Genre-Classification-Using-Tensorflow.git
cd music-genre-classification
```

2. Environment
```bash
python3 -m venv specgen
source specgen/bin/activate   # Linux/macOS  
specgen\Scripts\activate      # Windows  
pip install -r requirements.txt
```

3. Data
- Organize WAV files under data/audio/<genre>/
- Each file should be named like rock.00000.wav, jazz.00001.wav, etc.
- Supported genres:
```python
CLASSES = ['blues','classical','country','disco','hiphop',
           'jazz','metal','pop','reggae','rock']
```

## ▶️ Usage
Run inference on any WAV file:
```bash
python scripts/predict_g_classifier.py \
  -f data/audio/rock/rock.00000.wav \
  -m models/Trained_model.h5
```
- -f/--file: path to input WAV
- -m/--model: path to saved .h5 model (defaults to models/Trained_model.h5)

## 📓 Interactive Notebook
All data loading, preprocessing, model training, and evaluation live in the Jupyter notebook:
1. 	Spectrogram generation
<img src="https://github.com/user-attachments/assets/sample-spectrogram.png" width="571" height="455" alt="Spectrogram Sample" />

2. 	Model architecture
<img src="https://github.com/user-attachments/assets/model-architecture.png" width="597" height="372" alt="Model Architecture" />

3. 	Training accuracy
<img src="https://github.com/user-attachments/assets/training-accuracy.png" width="547" height="413" alt="Training Accuracy" />

4. 	Training loss
<img src="https://github.com/user-attachments/assets/training-loss.png" width="547" height="418" alt="Training Loss" />



## 📐 Model Architecture
```md
Input (150 × 150 × 1 Mel Spectrogram)
    │
    ▼
  Conv2D (32, relu) → Conv2D (32, relu)
    │
    ▼
  MaxPool2D → Conv2D (64, relu) → Conv2D (64, relu)
    │
    ▼
  MaxPool2D → Dropout (0.3)
    │
    ▼
  Conv2D (128, relu) → Conv2D (128, relu)
    │
    ▼
  MaxPool2D → Conv2D (256, relu) → Conv2D (256, relu)
    │
    ▼
  MaxPool2D → Dropout (0.3)
    │
    ▼
  Conv2D (512, relu) → Conv2D (512, relu)
    │
    ▼
  MaxPool2D → Flatten → Dense (1200, relu)
    │
    ▼
  Dropout (0.45) → Dense (10, softmax)
    │
    ▼
  Output: 10 genre probabilities
```

## 💾 Data & Preprocessing
- Duration: Load or pad/truncate to 3 s (or 4 s)
- Sampling: librosa.load(sr=None)
- Feature: librosa.feature.melspectogram → resized to → (150, 150)
- Labels: Derived from folder name (e.g., data/audio/jazz/jazz.00001.wav → jazz)

## 📊 Training & Results
- Checkpoint: model.save('../models/Trained_model.h5')
- History: Saved to: training_history.json
- Final Performance:
- Train accuracy: ~99%
- Test accuracy: ~98–99%
- Prediction confidence: e.g., rock(confidence: 0.999)

## 🤝 Contributing
- Fork the repo
- Branch naming: feature/genre-augmentation or fix/model-loading
- Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

- Code style: Follow PEP 8, lint with flake8
- Tests: Add unit tests under tests/ directory
- Pull request:
- Describe your changes
- Link any related issues
- Ensure CI passes


### 🎧 Thank you for tuning in! Let’s keep the music flowing 🎉










