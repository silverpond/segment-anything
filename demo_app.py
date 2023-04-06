import numpy as np
import os
import logging
from pathlib import Path
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import dumps
from shapely.affinity import scale

import onnxruntime

import matplotlib.pyplot as plt

def get_default_logger(name):
    # https://stackoverflow.com/questions/43109355/logging-setlevel-is-being-ignored
    logging.debug(f"Setting up logging for logger={name}")
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get("LOG_LEVEL", "INFO"))
    return logger

LOG = get_default_logger("OnnxSam")

def read_image(path, resize_wh = (1800, 1200)):
    image = Image.open(str(path))
    image = image.convert('RGB')

    if resize_wh is not None:
        image = image.resize(resize_wh, Image.BILINEAR)

    image = np.array(image, dtype=np.uint8)
    return image

from typing import Union
from typing_extensions import Annotated
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import io
import pickle
from segment_anything.demo_onnx_predictor import SamOnnxPredictor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SamOnnxPredictor()

@app.post("/predict")
def predict(file: Annotated[bytes, File()], point_x: int, point_y: int):
    LOG.info(type(file))
    image = Image.open(io.BytesIO(file))
    image = image.convert('RGB')
    image = np.array(image, dtype=np.uint8)

    input_points = np.array([[point_x, point_y]], dtype=np.float32)

    input_labels = np.ones(input_points.shape[0])
    wkt = predictor.predict(image, input_points, input_labels)
    #with open("mask.pkl", "wb") as f:
    #    pickle.dump(masks, f)

    #plt.figure(figsize=(10,10))
    #plt.imshow(image)
    #show_mask(masks, plt.gca())
    #show_points(input_points, input_labels, plt.gca())
    #plt.axis('off')
    #plt.show()
    print(wkt)
    
    return {"wkt": wkt}
