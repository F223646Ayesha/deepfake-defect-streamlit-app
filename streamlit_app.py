import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import streamlit as st
import librosa
import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# ------------------- Load Models -------------------
def load_models():
    return {
        "Deepfake_DNN": torch.load("models/deepfake_dnn.pt", map_location=torch.device("cpu")),
        "Defect_DNN": torch.load("models/defect_dnn.pt", map_location=torch.device("cpu")),
        "TFIDF": joblib.load("models/vectorizer.pkl"),
        "LR": joblib.load("models/lr_model.pkl"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "Perceptron": joblib.load("models/perceptron_model.pkl"),
        "Deepfake_LR": joblib.load("models/deepfake_lr_model.pkl"),
        "Deepfake_SVM": joblib.load("models/deepfake_svm_model.pkl"),
        "Deepfake_Perceptron": joblib.load("models/deepfake_perceptron_model.pkl"),
    }

# ------------------- DNN Loader for Deepfake -------------------
def load_dnn_for_deepfake(fixed_input_size, model_path):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(fixed_input_size, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2),
                torch.nn.Softmax(dim=1)
            )

        def forward(self, x):
            return self.layers(x)

    model = Net()
    model.load_state_dict(model_path)
    model.eval()
    return model

# ------------------- DNN Loader for Defect Prediction (final fix) -------------------
def load_dnn_for_defect(fixed_input_size, model_path):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(fixed_input_size, 256),  # net.0
                torch.nn.ReLU(),                         # net.1
                torch.nn.Linear(256, 128),               # net.3
                torch.nn.Linear(128, 7),                 # net.6
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = Net()
    model.load_state_dict(model_path)
    model.eval()
    return model


# ------------------- Deepfake Audio Prediction -------------------
def predict_audio(audio_file, classifier):
    models = load_models()
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc[:, :123].flatten()
    if mfcc.shape[0] < 1600:
        mfcc = np.pad(mfcc, (0, 1600 - mfcc.shape[0]), mode='constant')
    mfcc = mfcc.reshape(1, -1)

    if classifier == "DNN":
        model = load_dnn_for_deepfake(1600, models["Deepfake_DNN"])
        with torch.no_grad():
            output = model(torch.tensor(mfcc, dtype=torch.float32))
            pred = torch.argmax(output, dim=1).item()
            conf = output.numpy()[0][pred]
    else:
        clf = models[f"Deepfake_{classifier}"]
        proba = clf.predict_proba(mfcc)[0]
        pred = int(np.argmax(proba))
        conf = proba[pred]

    return ("Bonafide" if pred == 0 else "Deepfake"), conf

# ------------------- Defect Type Prediction -------------------
def predict_defect(text, classifier):
    models = load_models()
    tfidf = models["TFIDF"]
    vector = tfidf.transform([text])
    dense_vector = vector.toarray()

    expected_labels = ["blocker", "regression", "bug", "doc", "enhancement", "task", "dep_upgrade"]

    if classifier == "DNN":
        model = load_dnn_for_defect(dense_vector.shape[1], models["Defect_DNN"])
        with torch.no_grad():
            probs = model(torch.tensor(dense_vector, dtype=torch.float32)).numpy()[0]
    else:
        clf = models[classifier]
        raw = clf.predict_proba(dense_vector)

        # Handle list of outputs (e.g., OneVsRestClassifier)
        processed_probs = []
        for p in raw:
            if len(p.shape) == 2 and p.shape[1] > 1:
                processed_probs.append(p[:, 1])
            else:
                processed_probs.append(p)

        probs = np.array(processed_probs).T[0]  # final shape: (1, 7)

    # Ensure we got one score per label
    if len(probs) != len(expected_labels):
        raise ValueError(f"Expected {len(expected_labels)} probabilities, but got {len(probs)}: {probs}")

    threshold = 0.5
    predictions = [label for i, label in enumerate(expected_labels) if probs[i] >= threshold]
    conf_scores = {label: float(probs[i]) for i, label in enumerate(expected_labels)}

    return predictions, conf_scores


# ------------------- Streamlit UI -------------------
st.title("🧠 AI-Based Bug & Deepfake Predictor")

menu = st.sidebar.selectbox("Choose Task", ["Audio Deepfake Detection", "Defect Type Prediction"])
classifier = st.sidebar.selectbox("Choose Model", ["DNN", "SVM", "LogReg", "Perceptron"])

if menu == "Audio Deepfake Detection":
    st.subheader("🎧 Upload an audio file")
    audio_file = st.file_uploader("Upload .wav file", type=[".wav"])
    if audio_file:
        prediction, confidence = predict_audio(audio_file, classifier)
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")

elif menu == "Defect Type Prediction":
    st.subheader("📝 Enter bug report text")
    text = st.text_area("Bug Report Text")
    if st.button("Predict Defect Types") and text.strip():
        preds, confs = predict_defect(text, classifier)
        st.success(f"Predicted Labels: {', '.join(preds) if preds else 'None'}")
        st.write("Confidence Scores:")
        st.json(confs)
