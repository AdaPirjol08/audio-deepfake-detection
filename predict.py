import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model

# Parameters (must match training)
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109
MODEL_PATH = "audio_classifier.h5"
TEST_DIR = "TestEvaluation"
PLOTS_DIR = "plots"

# Create plots folder if missing
os.makedirs(PLOTS_DIR, exist_ok=True)

def predict_audio(file_path, model_path=MODEL_PATH):
    # Load and process audio
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_padded = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])), mode='constant')[:, :MAX_TIME_STEPS]
    mel_input = mel_db_padded[np.newaxis, ..., np.newaxis]

    # Load model and predict
    model = load_model(model_path)
    prediction = model.predict(mel_input)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    label = "bonafide" if class_id == 1 else "spoof"

    # Print result
    print(f"\n🎤 Prediction: {label.upper()} (confidence: {confidence:.2f})")

    # Plot and save Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {os.path.basename(file_path)}')
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"mel_{os.path.basename(file_path).replace('.flac', '')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Mel spectrogram saved to: {plot_path}")

    return label, confidence

if __name__ == "__main__":
    print("\n📁 Files available in TestEvaluation/:")
    files = [f for f in os.listdir(TEST_DIR) if f.endswith(".flac")]
    for idx, f in enumerate(files):
        print(f"{idx + 1}. {f}")

    choice = int(input("\n🔍 Choose a file number to test: ")) - 1
    selected_file = files[choice]
    file_path = os.path.join(TEST_DIR, selected_file)

    print(f"\n🎧 Testing: {selected_file}")
    predict_audio(file_path)
