import os
from PIL import Image
import pandas as pd


def create_yolo_labels(csv_path, labels_dir, img_dir):
    os.makedirs(labels_dir, exist_ok=True)
    data = pd.read_csv(csv_path)

    for _, row in data.iterrows():
        img_name = row['file']  # Usamos 'file' según el CSV
        class_id = row['race']  # Usamos 'race' como clase en este caso

        # Ruta actualizada a las imágenes en 'fairface/train' o 'fairface/val'
        img_path = os.path.join(img_dir, img_name)

        try:
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Convertir coordenadas de bounding box si las tienes, sino pon valores predeterminados
            x_center = y_center = width = height = 0.5  # ejemplo de valores

            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {img_path}")


# Actualizamos la ruta de las imágenes para cada uno de los conjuntos de datos
create_yolo_labels('data/fitzpatrick-classification-by-ethnicity/fitz_undersampled_train_final.csv',
                   'data/labels/train',
                   'data/fitzpatrick-classification-by-ethnicity/fairface/fairface/train')

create_yolo_labels('data/fitzpatrick-classification-by-ethnicity/fitz_undersampled_test_final.csv',
                   'data/labels/val',
                   'data/fitzpatrick-classification-by-ethnicity/fairface/fairface/val')
