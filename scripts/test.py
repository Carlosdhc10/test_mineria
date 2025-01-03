import os
import torch


def test_model():
    # Ruta a YOLOv5
    yolov5_dir = "yolov5"

    # Carga el modelo entrenado (asegúrate de que la ruta sea correcta)
    model_path = "runs/train/exp/weights/best_model.pt"

    # Ruta de las imágenes a las que se les realizará la detección
    source_path = "data/images/test"  # Ajusta esta ruta a tu carpeta de imágenes de prueba

    # Comando para ejecutar la detección
    os.system(
        f"python {yolov5_dir}/detect.py --weights {model_path} --source {source_path} --conf 0.25 --save-txt --save-img")


if __name__ == "__main__":
    test_model()
