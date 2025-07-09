import streamlit as st
import pickle
import pandas as pd

# Load model, scaler, encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# App layout
st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2563eb;'>🧠 Social Media Emotion Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>
        Predict your <b>dominant emotion</b> based on your social media usage patterns.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --- Input Form ---
with st.form("emotion_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👤 Age", min_value=10, max_value=100, value=25)
        gender = st.selectbox("⚧️ Gender", ["Female", "Male", "Non-binary"])
        platform = st.selectbox("📱 Platform", [
            "Facebook", "Instagram", "LinkedIn", "Snapchat",
            "Telegram", "Twitter", "Whatsapp"
        ])
        daily_usage = st.slider("🕒 Daily Usage (minutes)", 0, 600, 120)
    with col2:
        posts = st.slider("📝 Posts Per Day", 0, 20, 3)
        likes = st.slider("❤️ Likes Received Per Day", 0, 500, 45)
        comments = st.slider("💬 Comments Per Day", 0, 100, 10)
        messages = st.slider("📨 Messages Sent Per Day", 0, 100, 12)

    submitted = st.form_submit_button("🎯 Predict Emotion")

# --- Preprocessing ---
def preprocess_input():
    features = [
        "Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day",
        "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day",
        "Gender_Female", "Gender_Male", "Gender_Non-binary",
        "Platform_Facebook", "Platform_Instagram", "Platform_LinkedIn", "Platform_Snapchat",
        "Platform_Telegram", "Platform_Twitter", "Platform_Whatsapp"
    ]
    data = dict.fromkeys(features, 0)
    data["Age"] = age
    data["Daily_Usage_Time (minutes)"] = daily_usage
    data["Posts_Per_Day"] = posts
    data["Likes_Received_Per_Day"] = likes
    data["Comments_Received_Per_Day"] = comments
    data["Messages_Sent_Per_Day"] = messages
    data[f"Gender_{gender}"] = 1
    data[f"Platform_{platform}"] = 1

    df = pd.DataFrame([[data[f] for f in features]], columns=features)
    numeric = [
        "Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day",
        "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"
    ]
    df[numeric] = scaler.transform(df[numeric])
    return df

# --- Prediction ---
if submitted:
    input_df = preprocess_input()
    probas = model.predict_proba(input_df)[0]
    pred_index = probas.argmax()
    confidence = round(probas[pred_index] * 100, 2)
    emotion = label_encoder.inverse_transform([pred_index])[0]

    st.markdown(
        f"""
        <div style='text-align: center; font-size: 26px; padding: 20px; border: 2px solid #2563eb; border-radius: 10px; background-color: #f0f8ff;'>
            🎯 Predicted Emotion: <b style='color:#2563eb'>{emotion}</b><br>
            🔒 Confidence Score: <b>{confidence}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )


