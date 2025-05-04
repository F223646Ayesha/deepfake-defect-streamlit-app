import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎯 AI-Powered Prediction App</h1>", unsafe_allow_html=True)

# -------------------- MODEL TRAINING UTILS --------------------
@st.cache_resource
def train_audio_models():
    def generate_signal(label, sr=16000, duration=1.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        if label == 0:
            signal = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 600 * t)
            signal += 0.1 * np.random.randn(len(t))
        else:
            signal = np.sin(2 * np.pi * 300 * t + np.pi/4)
            signal += 0.5 * np.sin(2 * np.pi * 700 * t + np.pi/6)
            signal += 0.4 * np.random.randn(len(t))
        return signal / np.max(np.abs(signal)), label

    X_audio, y_audio = [], []
    for i in range(200):
        x, y = generate_signal(i % 2)
        mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=20)
        mfcc = librosa.util.fix_length(mfcc, size=100, axis=1)
        X_audio.append(mfcc.flatten())
        y_audio.append(y)

    return np.array(X_audio), np.array(y_audio)

@st.cache_resource
def train_text_models():
    sample_texts = [
        "App crashes on launch",
        "Update documentation for config file",
        "Improve error message for invalid input",
        "Fix regression in payment module",
        "Update library version to latest",
        "Add feature to export PDF",
        "Dependency upgrade breaks old API"
    ]
    label_vectors = [
        [1, 0, 1, 0, 0, 0, 0],  # blocker + bug
        [0, 0, 0, 1, 0, 0, 0],  # doc
        [0, 0, 1, 0, 0, 0, 0],  # bug
        [0, 1, 1, 0, 0, 0, 0],  # regression + bug
        [0, 0, 0, 0, 0, 0, 1],  # dep upgrade
        [0, 0, 0, 0, 1, 1, 0],  # enhancement + task
        [0, 0, 1, 0, 0, 0, 1],  # bug + dep upgrade
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sample_texts).toarray()
    y = np.array(label_vectors)
    return X, y, vectorizer

X_audio, y_audio = train_audio_models()
X_text, y_text, tfidf = train_text_models()

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid() if output_dim > 1 else nn.Softmax(dim=1)
        )
    def forward(self, x): return self.net(x)

# -------------------- SIDEBAR --------------------
st.sidebar.title("🔧 App Controls")
section = st.sidebar.radio("Choose Section", ["Audio Deepfake Detection", "Defect Prediction"])
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "SVM", "Perceptron", "DNN"])
st.sidebar.markdown("📌 Models:\n- LR & SVM: Probabilistic\n- Perceptron: Binary only\n- DNN: Trained on startup")

# -------------------- AUDIO SECTION --------------------
if section == "Audio Deepfake Detection":
    st.header("🎧 Upload Audio File")
    audio_file = st.file_uploader("Choose a `.wav` file", type=['wav'])

    if audio_file:
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = librosa.util.fix_length(mfcc, size=100, axis=1)
        audio_input = mfcc.flatten().reshape(1, -1)

        st.audio(audio_file, format="audio/wav")
        st.markdown("#### 📈 Waveform")

        show_full = st.checkbox("🔁 Show full waveform", value=False)

        fig, ax = plt.subplots(figsize=(8, 3))
        if show_full:
            ax.plot(np.arange(len(y)) / sr, y, color='dodgerblue', linewidth=1)
            ax.set_title("🔊 Full Audio Waveform (Time in Seconds)", fontsize=12)
            ax.set_xlabel("Time (s)", fontsize=10)
        else:
            ax.plot(np.arange(500) / sr, y[:500], color='dodgerblue', linewidth=1.2)
            ax.set_title("🔊 Audio Waveform (First 500 Samples)", fontsize=12)
            ax.set_xlabel("Time (s)", fontsize=10)

        ax.set_ylabel("Amplitude", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

        if model_choice == "DNN":
            model = SimpleDNN(X_audio.shape[1], 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(5):
                model.train()
                xb = torch.tensor(X_audio, dtype=torch.float32)
                yb = torch.tensor(y_audio, dtype=torch.long)
                out = model(xb)
                loss = loss_fn(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(audio_input, dtype=torch.float32))
                prob = pred.numpy().flatten()
        else:
            if model_choice == "SVM":
                clf = SVC(probability=True)
                clf.fit(X_audio, y_audio)
                prob = clf.predict_proba(audio_input)[0]
            elif model_choice == "Perceptron":
                clf = Perceptron()
                clf.fit(X_audio, y_audio)
                pred = clf.predict(audio_input)[0]
                prob = [1.0, 0.0] if pred == 0 else [0.0, 1.0]
            else:
                clf = LogisticRegression()
                clf.fit(X_audio, y_audio)
                prob = clf.predict_proba(audio_input)[0]

        label = "Bonafide" if np.argmax(prob) == 0 else "Deepfake"
        st.subheader(f"✅ Prediction: **{label}**")
        st.progress(float(prob[0]), text=f"Bonafide Confidence: {prob[0]:.2f}")
        st.progress(float(prob[1]), text=f"Deepfake Confidence: {prob[1]:.2f}")

# -------------------- TEXT SECTION --------------------
if section == "Defect Prediction":
    st.header("📝 Enter Bug Report")
    text_input = st.text_area("Describe the issue...", height=100)

    if text_input:
        text_vec = tfidf.transform([text_input]).toarray()
        defect_labels = ['blocker', 'regression', 'bug', 'documentation', 'enhancement', 'task', 'dependency upgrade']

        if model_choice == "DNN":
            model = SimpleDNN(X_text.shape[1], y_text.shape[1])
            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(10):
                model.train()
                xb = torch.tensor(X_text, dtype=torch.float32)
                yb = torch.tensor(y_text, dtype=torch.float32)
                outputs = model(xb)
                loss = loss_fn(outputs, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(text_vec, dtype=torch.float32)).numpy().flatten()
        else:
            if model_choice == "SVM":
                clf = OneVsRestClassifier(SVC(probability=True))
                clf.fit(X_text, y_text)
                pred = clf.predict_proba(text_vec)[0]
            elif model_choice == "Perceptron":
                clf = OneVsRestClassifier(Perceptron())
                clf.fit(X_text, y_text)
                pred = clf.predict(text_vec)[0]
            else:
                clf = OneVsRestClassifier(LogisticRegression())
                clf.fit(X_text, y_text)
                pred = clf.predict_proba(text_vec)[0]

        st.subheader("📋 Predicted Labels")
        for i, p in enumerate(pred):
            if p >= 0.5:
                with st.expander(f"✅ {defect_labels[i].title()} — Confidence: {p:.2f}"):
                    st.progress(float(p))

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("🧠 Created for Data Science A4 Assignment | Contact your instructor if the app crashes or freezes.")
