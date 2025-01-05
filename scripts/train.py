import os
import sys
import torch


def train_model():
    # Ruta a YOLOv5
    yolov5_dir = "../yolov5"

    # Comando para ejecutar el entrenamiento con YOLOv5
    os.system(
        f"python {yolov5_dir}/train.py --img 640 --batch 16 --epochs 50 --data data/maquillaje.yaml --weights yolov5s.pt --cache")


if __name__ == "__main__":
    train_model()
