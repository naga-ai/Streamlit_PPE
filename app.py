
import streamlit as st
from PIL import Image
import cv2
import requests
import os
import numpy as np

from ultralytics import YOLO
import yolov5

# Function for inference
def yolov5_inference(image, model_path, image_size, conf_threshold, iou_threshold):
    # Loading Yolo V5 model
    model = yolov5.load(model_path, device="cpu")

    # Setting model configuration 
    model.conf = conf_threshold
    model.iou = iou_threshold

    # Inference
    results = model([image], size=image_size)

    # Cropping the predictions    
    crops = results.crop(save=False)
    img_crops = []
    for i in range(len(crops)):
        img_crops.append(crops[i]["im"][..., ::-1])
    return results.render()[0] #, img_crops

# Title
st.title("Identify violations of Personal Protective Equipment (PPE) protocols for improved safety")

# Input
image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
model_path = st.selectbox("Model", ["PPE_Safety_Y5.pt"], index=0)
image_size = st.slider("Image Size", min_value=320, max_value=1280, value=640, step=32)
conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
iou_threshold = st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

# Inference
if image and st.button("Run"):
    image = Image.open(image)
    image = np.array(image)
    output = yolov5_inference(image, model_path, image_size, conf_threshold, iou_threshold)
    st.image(output, caption="Output Image", use_column_width=True)
