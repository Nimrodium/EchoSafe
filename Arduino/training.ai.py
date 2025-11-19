import os
import numpy as np
import pandas as pd
from scipy.fft import rfft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd

# CSV file paths
csv_files = ["sound_data_label0.csv", "sound_data_label1.csv"]  # noise, clap


# Number of FFT features per sample (keep small for Arduino)
NUM_FEATURES = 20


# -----------------------------
# 2. FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_features(df : dict[str,str]) -> np.ndarray:
    """Extract simple FFT-based features from mic_value column."""
    signal = df["mic_value"].values
    window_size = 256
    features = []
    for i in range(0, len(signal) - window_size, window_size):
        frame = signal[i : i + window_size]
        fft_vals = np.abs(rfft(frame))
        features.append(np.mean(fft_vals))
    features = features[:NUM_FEATURES]
    if len(features) < NUM_FEATURES:
        features += [0] * (NUM_FEATURES - len(features))
    return np.array(features)


# -----------------------------
# 3. LOAD AND PROCESS DATA
# -----------------------------
X = []
y = []

for file in csv_files:
    if not os.path.exists(file):
        print(f"⚠️ File not found: {file}")
        continue

    df = pd.read_csv(file)
    label = int(df["label"].iloc[0])  # 0=noise, 1=clap
    X.append(extract_features(df))
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples.")
print("Feature shape:", X.shape)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# 4. DEFINE & TRAIN THE MODEL
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(NUM_FEATURES,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🧠 Training model...")
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

# -----------------------------
# 5. EVALUATE & SAVE THE MODEL
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model accuracy: {acc*100:.2f}%")

# Save Keras model
model.save("sound_model.h5")
print("💾 Saved Keras model as sound_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("sound_model.tflite", "wb") as f:
    f.write(tflite_model)
print("💾 Saved TensorFlow Lite model as sound_model.tflite")

print("\n🎉 Training complete! You can now deploy 'sound_model.tflite' to Arduino.")



# number of samples to generate
NUM_SAMPLES = 2000

# ---- Noise data (label 0) ----
time = np.arange(NUM_SAMPLES) / 1000.0    # 1 kHz sampling
noise_values = np.random.normal(120, 5, NUM_SAMPLES).astype(int)

df_noise = pd.DataFrame({
    "time": time,
    "mic_value": noise_values,
    "label": np.zeros(NUM_SAMPLES, dtype=int)
})

df_noise.to_csv("sound_data_label0.csv", index=False)
print("Generated sound_data_label0.csv")


# ---- Clap data (label 1) ----
clap = []

for i in range(NUM_SAMPLES):
    if 300 < i < 350:             # main clap peak
        v = np.random.normal(400, 40)
    elif 350 <= i < 600:          # decay
        decay = np.exp(-(i-350)/80) * 250
        v = decay + np.random.normal(140, 10)
    else:                          # background
        v = np.random.normal(130, 6)

    clap.append(int(v))

df_clap = pd.DataFrame({
    "time": time,
    "mic_value": clap,
    "label": np.ones(NUM_SAMPLES, dtype=int)
})

df_clap.to_csv("sound_data_label1.csv", index=False)
print("Generated sound_data_label1.csv")
