
# ASVSpoof2019 CNN training and evaluation pipeline with timing and caching

import os
import time
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import seaborn as sns
import librosa.display
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Parameters
DATASET_PATH = "LA/ASVspoof2019_LA_train/flac"
LABEL_FILE_PATH = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
TEST_DATASET_PATH = "./TestEvaluation"
PROTOCOL_FILE_PATH = "test_eval.txt"
MODEL_PATH = "audio_classifier.h5"
NUM_CLASSES = 2
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

os.makedirs("plots", exist_ok=True)

def load_labels(label_file_path):
    labels = {}
    with open(label_file_path, 'r') as label_file:
        for line in label_file.readlines():
            parts = line.strip().split()
            file_name = parts[1]
            label = 1 if parts[-1] == "bonafide" else 0
            labels[file_name] = label
    return labels

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.pad(mel, ((0, 0), (0, max(0, MAX_TIME_STEPS - mel.shape[1]))), mode='constant')[:, :MAX_TIME_STEPS]
    return mel

def load_data(dataset_path, labels, cache_prefix="train"):
    X_path = f"{cache_prefix}_X.npy"
    y_path = f"{cache_prefix}_y.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"📂 Loading cached data from {X_path}, {y_path}")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("🔄 Preprocessing audio files (no cache found)...")
        X, y = [], []
        for file_name, label in labels.items():
            file_path = os.path.join(dataset_path, file_name + ".flac")
            mel = preprocess_audio(file_path)
            X.append(mel)
            y.append(label)
        X = np.array(X)[..., np.newaxis]
        y = to_categorical(np.array(y), NUM_CLASSES)
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"✅ Cached preprocessed data to {X_path}, {y_path}")
    return X, y

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_and_save(fig, filename):
    fig.savefig(f"plots/{filename}")
    plt.close(fig)

def evaluate_model(y_true, y_pred, y_pred_prob):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spoof", "bonafide"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plot_and_save(plt.gcf(), "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plot_and_save(plt.gcf(), "roc_curve.png")

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print(f"📉 Equal Error Rate (EER): {eer:.4f}")
    with open("plots/eer.txt", "w") as f:
        f.write(f"EER: {eer:.4f}\n")

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plot_and_save(plt.gcf(), "precision_recall_curve.png")

    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, 'o-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plot_and_save(plt.gcf(), "calibration_curve.png")

def main():
    total_start = time.time()
    print("🚀 Starting training pipeline...")

    labels = load_labels(LABEL_FILE_PATH)
    start = time.time()
    X, y = load_data(DATASET_PATH, labels, cache_prefix="train")
    print(f"⏳ Training data loaded in {time.time() - start:.2f} seconds.")

    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    input_shape = (N_MELS, MAX_TIME_STEPS, 1)
    if not os.path.exists(MODEL_PATH):
        print("🔧 Training model...")
        model = build_model(input_shape)
        start = time.time()
        history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
        print(f"✅ Model training took {time.time() - start:.2f} seconds.")
        model.save(MODEL_PATH)

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plot_and_save(plt.gcf(), "loss_curve.png")
    else:
        print("✅ Model already exists. Loading...")
        model = load_model(MODEL_PATH)

    test_labels = load_labels(PROTOCOL_FILE_PATH)
    start = time.time()
    X_test, y_true = load_data(TEST_DATASET_PATH, test_labels, cache_prefix="test")
    print(f"⏳ Test data loaded in {time.time() - start:.2f} seconds.")

    start = time.time()
    y_pred_prob = model.predict(X_test)[:, 1]
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"🔍 Predictions completed in {time.time() - start:.2f} seconds.")

    evaluate_model(y_true.argmax(axis=1), y_pred, y_pred_prob)

    print(f"✅ Total pipeline finished in {time.time() - total_start:.2f} seconds.")

if __name__ == "__main__":
    main()