import numpy as np
from typing import List, Tuple, Dict, Union
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

def mask_to_wkt(mask: np.ndarray, scale_wh: np.ndarray) -> str:

    mask_as_uint = mask.astype(np.uint8) * 255

    # jank open cv version check to account for different cv2.findcontours
    # returned tuple size
    contours_tuple = cv2.findContours(
        mask_as_uint, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours_tuple) == 2:
        contours, _ = contours_tuple
    else:
        _, contours, _ = contours_tuple

    approx = [
        np.squeeze(
            cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
        ).reshape((-1, 2))
        for cnt in contours
    ]
    # it's not a valid polygon if it has less than 3 vertices
    approx = [coords for coords in approx if len(coords) > 2]

    polys = []
    for poly_points in approx:
        polys.append(Polygon(poly_points/scale_wh))
    mp = MultiPolygon(polys)

    wkt = dumps(mp)
    return wkt

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

def image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((1800, 1200), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)



def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

CKPT_PATH = Path(__file__).parent / ".." / "checkpoints" / "sam_vit_h_4b8939.pth"
ONNX_MODEL_PATH = Path(__file__).parent / ".." / "checkpoints" / "sam_onnx_quantized_vit_h.onnx"
MODEL_TYPE = "vit_h"

class SamOnnxPredictor():
    def __init__(self, onnx_model_path=ONNX_MODEL_PATH, model_type=MODEL_TYPE, checkpoint_path=CKPT_PATH):
        self.ort_session = onnxruntime.InferenceSession(str(onnx_model_path))
        
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)
        self.resize_wh = (1800, 1200)
        self.image_scale_factor = np.array([1.0, 1.0])

    def pre_process(self, image: np.ndarray, input_points: np.ndarray,
                    ) -> Tuple[np.ndarray, np.ndarray]:

        image_original_wh = image.shape[:2][::-1]

        image = cv2.resize(image, self.resize_wh, interpolation = cv2.INTER_AREA)

        scale_w = self.resize_wh[0] / image_original_wh[0]
        scale_h = self.resize_wh[1] / image_original_wh[1]
        # Used to upscale output mask points in post_process function
        self.image_scale_factor = np.array([scale_w, scale_h])

        input_points *= self.image_scale_factor

        return image, input_points

    def post_process(self, masks: np.ndarray) -> np.ndarray:
        masks = masks > self.predictor.model.mask_threshold
        return mask_to_wkt(masks, self.image_scale_factor)


    def predict(self, image: np.ndarray, input_points: np.ndarray,
                input_labels: np.ndarray) -> np.ndarray:

        input_points = input_points.astype(np.float32)

        pre_proc_image, input_points = self.pre_process(image, input_points)
        self.predictor.set_image(pre_proc_image)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        
        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, pre_proc_image.shape[:2]).astype(np.float32)
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(pre_proc_image.shape[:2], dtype=np.float32)
        }
        
        masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        masks = self.post_process(masks[0][0]) 
        return masks

        
if __name__ == "__main__":

    predictor = SamOnnxPredictor(ONNX_MODEL_PATH, MODEL_TYPE, CKPT_PATH)
    image = read_image("assets/eq_pole_image_id_9660254.jpg")
    input_points = np.array([[640, 640],
                            [876, 886],
                            [454, 592],
                            [864, 592],
                            [1344, 588],
                            ])
    input_labels = np.ones(input_points.shape[0])
    masks = predictor.predict(image, input_points, input_labels)
        
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.show()
