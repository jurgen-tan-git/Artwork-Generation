import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import gc
import subprocess

t = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


def predict():
    bash_command = "python test.py --dataroot ./datasets/images  --name style_vangogh  --model test --no_dropout  --gpu_ids -1"
    result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)


st.title(body = "Van Gogh style transfer using CycleGAN.")

file = st.file_uploader(label="Upload your image",type=['.png','.jpg','jpeg'])

if not file:
    st.warning("Please upload an image")
    gc.collect()
    st.stop()
else:
    image = file.read()
    st.caption("Your image.")
    st.image(image, use_column_width=True)
    img = Image.open(io.BytesIO(image)).convert("RGB")
    pred_button = st.button("Generate")

if pred_button:
    with st.spinner("Generating. Please wait..."):
        gen_image = predict()
        del img
        st.caption("Generated image.")
        st.image(gen_image)
    del gen_image
    del image