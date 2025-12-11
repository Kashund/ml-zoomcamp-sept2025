import json
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort


MODEL_PATH = "hair_classifier_empty.onnx"  # in Docker image; for local tests you can change it

def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def preprocess_pil_image(img: Image.Image, target_size=(200, 200)) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)

    img_np = np.array(img).astype("float32")
    img_np = img_np / 255.0

    img_chw = np.transpose(img_np, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_norm = (img_chw - mean) / std
    x = np.expand_dims(img_norm, 0)
    return x


# Create ONNX session once (cold start)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict_from_url(url: str) -> float:
    img = download_image(url)
    x = preprocess_pil_image(img).astype("float32")   # ensure float32 for ONNX
    pred = session.run([output_name], {input_name: x})[0]
    score = float(pred[0][0])
    return score


def lambda_handler(event, context):
    # API Gateway often passes JSON as a string in event["body"]
    if "body" in event and isinstance(event["body"], str):
        body = json.loads(event["body"])
    else:
        body = event

    url = body.get("url")
    if not url:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "url is required"})
        }

    score = predict_from_url(url)
    result = {"score": score}

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
