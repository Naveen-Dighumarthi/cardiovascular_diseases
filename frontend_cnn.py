import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model("C:\\Users\\91934\\Downloads\\cnn1_model.h5")  # Replace with your model path

# Class labels (must match the order used during training)
class_labels = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal']

# Function to preprocess the image
def preprocess_image(image):
    img = load_img(image, target_size=(256, 256))  # Resize to match model input
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit UI
st.title("Cardiovascular Disease Classification")
st.write("Upload an ECG image to classify the condition.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded ECG Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess and predict
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    diagnosis = class_labels[predicted_class]

    # Display results
    st.write("### Diagnosis:")
    st.write(f"**{diagnosis}**")
    st.write("### Prediction Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {prediction[0][i]:.2f}")
