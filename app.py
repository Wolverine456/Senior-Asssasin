import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to process the image and make predictions
def predict(image):
    img = Image.open(image)
    img = ImageOps.fit(img, (224, 224), method=Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    result = {}
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        result[label] = f'{score:.2f}'
    
    return result

# Streamlit UI for uploading an image
st.title("Image Recognition App")
st.write("Upload an image of an animal or plant for recognition")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Make predictions if an image is uploaded
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    predictions = predict(uploaded_image)
    st.write("Predictions:")
    for label, score in predictions.items():
        st.write(f"{label}: {score}")