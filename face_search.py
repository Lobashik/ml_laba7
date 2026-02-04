import os
import cv2
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from deepface import DeepFace


conn = psycopg2.connect("dbname=face_db user=postgres password=password host=localhost")
cursor = conn.cursor()

cursor.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding VECTOR(128)
);
""")
conn.commit()


def build_embeddings_from_csv(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    data = []

    for idx, row in df.iterrows():
        filename = row['filename']
        name = row['name']

        path = os.path.join(images_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"Не удалось загрузить {path}, пропускаем...")
            continue

        try:
            embedding = DeepFace.represent(img_path=img, model_name="Facenet")[0]["embedding"]  # type: ignore
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            data.append((name, embedding_str))
            print(f"Эмбеддинг для {name} создан.")
        except Exception as e:
            print(f"Ошибка при создании эмбеддинга для {name}: {e}")
            continue

    if len(data) == 0:
        raise ValueError("Не удалось создать ни одного эмбеддинга!")

    execute_values(cursor, "INSERT INTO embeddings (name, embedding) VALUES %s", data)
    conn.commit()
    print(f"Вставлено {len(data)} эмбеддингов в базу данных.")


def find_face(image_data):
    """
    Поиск ближайшего лица в БД.
    
    Args:
        image_data: numpy array с изображением или путь к файлу
        
    Returns:
        dict: {'name': str, 'distance': float} или None если не найдено
    """
    try:
        embedding = DeepFace.represent(img_path=image_data, model_name="Facenet")[0]["embedding"]  # type: ignore
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        cursor.execute(
            "SELECT name, (embedding <-> %s::vector) as distance FROM embeddings ORDER BY distance LIMIT 1",
            (embedding_str,)
        )
        result = cursor.fetchone()
        
        if result:
            name, distance = result
            return {'name': name, 'distance': float(distance)}
        return None
    except Exception as e:
        print(f"Ошибка при поиске лица: {e}")
        return None


if __name__ == "__main__":
    build_embeddings_from_csv("hse_faces_miem/staff_photo.csv", "hse_faces_miem")
    cursor.close()
    conn.close()
