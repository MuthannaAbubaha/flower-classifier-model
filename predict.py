import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='Flower Classifier')

parser.add_argument('image_path', type=str)
parser.add_argument('model_path', type=str)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default=None)

args = parser.parse_args()

model = tf.keras.models.load_model(
    args.model_path,
    custom_objects={'KerasLayer': hub.KerasLayer},
    compile=False
)

class_names = None
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k):

    image = Image.open(image_path)
    image = process_image(np.asarray(image))
    image = np.expand_dims(image, axis=0)

    ps = model.predict(image)

    top_indices = np.argsort(ps[0])[-top_k:][::-1]
    probs = ps[0][top_indices]

    if class_names:
        labels = [class_names[str(i)] for i in top_indices]
    else:
        labels = [str(i) for i in top_indices]

    return probs, labels

probs, labels = predict(args.image_path, model, args.top_k)

print("\nTop Predictions:\n")

for label, prob in zip(labels, probs):
    print(f"{label} : {prob:.4f}")