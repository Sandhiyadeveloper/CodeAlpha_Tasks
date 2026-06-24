# ==========================================
# Speech Emotion Recognition using TESS Dataset
# ==========================================

# Import Libraries
import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ==========================================
# 1. Feature Extraction
# ==========================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    # MFCC Features
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    # Chroma Features
    stft = np.abs(librosa.stft(audio))

    chroma = np.mean(
        librosa.feature.chroma_stft(
            S=stft,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Mel Spectrogram Features
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Combine Features
    features = np.hstack([
        mfcc,
        chroma,
        mel
    ])

    return features


# ==========================================
# 2. Load TESS Dataset
# ==========================================

dataset_path = "TESS Dataset"

features = []
emotions = []


for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(
                root,
                file
            )

            try:
                data = extract_features(file_path)

                # Emotion name is in file name
                emotion = file.split("_")[-1]
                emotion = emotion.replace(".wav", "")

                features.append(data)
                emotions.append(emotion)

            except Exception as e:
                print("Error:", file_path)


print("Total Audio Files:", len(features))


# ==========================================
# 3. Convert to Numpy Array
# ==========================================

X = np.array(features)
y = np.array(emotions)


print("Feature Shape:", X.shape)


# ==========================================
# 4. Encode Labels
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


print("Emotions:")
print(encoder.classes_)


# ==========================================
# 5. Feature Scaling
# ==========================================

scaler = StandardScaler()

X = scaler.fit_transform(X)


# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_categorical
)


# ==========================================
# 7. Build Deep Learning Model
# ==========================================

model = Sequential()


model.add(
    Dense(
        256,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(Dropout(0.3))


model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(Dropout(0.3))


model.add(
    Dense(
        64,
        activation="relu"
    )
)


model.add(
    Dense(
        y_train.shape[1],
        activation="softmax"
    )
)


# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==========================================
# 8. Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)


# ==========================================
# 9. Evaluate Model
# ==========================================

prediction = model.predict(X_test)


y_pred = np.argmax(
    prediction,
    axis=1
)


y_true = np.argmax(
    y_test,
    axis=1
)


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n============== RESULTS ==============")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )
)


# ==========================================
# 10. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(10,7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Speech Emotion Recognition Confusion Matrix")

plt.show()


# ==========================================
# 11. Save Model
# ==========================================

model.save(
    "speech_emotion_model.h5"
)

print("\nModel Saved Successfully!")