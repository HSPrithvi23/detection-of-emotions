import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import sqlite3
import os

# ===================== PATH SETUP =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DB_PATH = os.path.join(BASE_DIR, "..", "database", "user_records.db")

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Emotion + Depression Detection", layout="wide")
st.title("🧠 Facial Emotion Recognition + Depression Risk Scoring")

# Sidebar
st.sidebar.header("Choose Function")
mode = st.sidebar.radio("Select Mode:", 
                        ["Real-time Emotion Detection", 
                         "PHQ-9 Depression Test",
                         "Custom Model Prediction (ML/DL)"])

# ======================================================
# 1️⃣ REAL-TIME EMOTION DETECTION
# ======================================================
if mode == "Real-time Emotion Detection":
    st.write("Press **Start Camera** to detect your facial emotion in real-time.")
    start = st.button("Start Camera")

    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop = st.button("Stop")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected.")
                break

            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f'Emotion: {emotion}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels='BGR')

            if stop:
                break
        cap.release()

# ======================================================
# 2️⃣ PHQ-9 DEPRESSION TEST
# ======================================================
elif mode == "PHQ-9 Depression Test":
    st.subheader("📝 PHQ-9 Questionnaire")
    st.write("Rate how often you experienced these problems in the **last 2 weeks**:")

    options = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
    questions = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble sleeping or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Poor appetite or overeating",
        "6. Feeling bad about yourself or feeling like a failure",
        "7. Trouble concentrating on things",
        "8. Moving/speaking slowly or being restless",
        "9. Thoughts of self-harm or self-injury"
    ]

    total_score = 0
    for q in questions:
        ans = st.radio(q, list(options.keys()), key=q)
        total_score += options[ans]

    if st.button("Calculate Depression Risk"):
        st.success(f"**Your PHQ-9 Score:** {total_score}/27")

        if total_score <= 4:
            risk = "Minimal depression"
            st.info(risk)
        elif total_score <= 9:
            risk = "Mild depression"
            st.warning(risk)
        elif total_score <= 14:
            risk = "Moderate depression"
            st.warning(risk)
        elif total_score <= 19:
            risk = "Moderately severe depression"
            st.error(risk)
        else:
            risk = "Severe depression — please seek professional help"
            st.error(risk)

        # 💾 Save record in SQLite
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS depression_logs(
                        user TEXT, score INTEGER, risk_level TEXT)""")
        cursor.execute("INSERT INTO depression_logs VALUES (?, ?, ?)", 
                       ("User1", total_score, risk))
        conn.commit()
        conn.close()

        # Optional chatbot suggestion
        if total_score >= 10:
            st.write("💬 **Chatbot Suggestion:** Try journaling, deep breathing, talking to a friend, or contacting a counselor.")

# ======================================================
# 3️⃣ CUSTOM MODEL PREDICTION (ML / DL)
# ======================================================
elif mode == "Custom Model Prediction (ML/DL)":
    st.subheader("🎯 Test Emotion Models (SVM vs Deep Learning)")
    choice = st.radio("Select Model:", ["SVM (Machine Learning)", "MobileNetV2 (Deep Learning)"])

    uploaded_file = st.file_uploader("Upload an image for emotion detection", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (48, 48))
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        if choice == "SVM (Machine Learning)":
            # Load SVM model and scaler
            svm_path = os.path.join(OUTPUTS_DIR, "svm_emotion_model.pkl")
            with open(svm_path, "rb") as f:
                data = pickle.load(f)

            svm = data["svm"]
            scaler = data["scaler"]
            class_names = data["classes"]

            # Extract features using MobileNet
            mobilenet = tf.keras.applications.MobileNetV2(
                include_top=False, input_shape=(48, 48, 3), pooling="avg", weights="imagenet"
            )
            img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_resized, axis=0))
            features = mobilenet.predict(img_input)
            X_scaled = scaler.transform(features)
            pred = svm.predict(X_scaled)[0]
            st.success(f"Predicted Emotion (SVM): **{class_names[pred]}**")

        else:
            # Load DL model
            model_path = os.path.join(OUTPUTS_DIR, "emotion_model_mobilenet.keras")
            model = tf.keras.models.load_model(model_path)
            img_norm = np.expand_dims(img_resized / 255.0, axis=0)
            preds = model.predict(img_norm)
            class_idx = np.argmax(preds)
            labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            st.success(f"Predicted Emotion (DL): **{labels[class_idx]}**")
