import model
import cv2
import numpy as np
import streamlit as st

# Streamlit app
st.title("Object Detection using YOLO")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
  image=cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
  st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
  predicted_image=model.predict(image=image)
  st.image(predicted_image, channels="BGR", caption="Uploaded Image", use_column_width=True)