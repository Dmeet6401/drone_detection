from io import StringIO
from pathlib import Path
import streamlit as st
import time
import cv2
import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

available=[]
st.header("DRONE DETECTION")
file_upload = st.file_uploader("choose file ", type = ["jpg", "png","jpeg"])
col1, col2= st.columns(2)
CLASSES = ['Drone']
with col1:

    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="..\\model\\drone_yolov5.pt")  # load scratch
    if file_upload is not None:
        img = Image.open(file_upload)
        result  = model(img)
        st.image(np.squeeze(result.render()))
        # st.write(img.pandas().xyxy[0])

        
with col2:
    
    try:
        drone_count = result.pandas().xyxy[0].name.value_counts()[CLASSES[0]]
        available.append(drone_count)
    except Exception as e:
        available.append(0)
    # try:
    #     banana_count = result.pandas().xyxy[0].name.value_counts()[CLASSES[1]]
    #     available.append(banana_count)
    # except Exception as e:
    #     available.append(0)
    # try:
    #     orange_count = result.pandas().xyxy[0].name.value_counts()[CLASSES[2]]
    #     available.append(orange_count)
    # except Exception as e:
    #     available.append(0)
    for index in range(len(CLASSES)):
        st.write(f""" {CLASSES[index]} detected""")
