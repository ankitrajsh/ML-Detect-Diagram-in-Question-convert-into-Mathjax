import pandas as pd
import os
from PIL import Image

def preprocess_text_data(raw_text_file, processed_text_file):
    df = pd.read_csv(raw_text_file)
    # Example preprocessing: remove null values and reset index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(processed_text_file, index=False)

def preprocess_images(raw_image_dir, processed_image_dir):
    if not os.path.exists(processed_image_dir):
        os.makedirs(processed_image_dir)

    for category in os.listdir(raw_image_dir):
        category_path = os.path.join(raw_image_dir, category)
        if os.path.isdir(category_path):
            processed_category_path = os.path.join(processed_image_dir, category)
            if not os.path.exists(processed_category_path):
                os.makedirs(processed_category_path)

            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(image_path)
                    # Example preprocessing: resize image
                    image = image.resize((256, 256))
                    processed_image_path = os.path.join(processed_category_path, image_file)
                    image.save(processed_image_path)

if __name__ == "__main__":
    raw_text_file = '../data/raw/text_dataset.csv'
    processed_text_file = '../data/processed/processed_text_dataset.csv'
    raw_image_dir = '../data/raw/images/diagram'
    processed_image_dir = '../data/processed/processed_images'

    preprocess_text_data(raw_text_file, processed_text_file)
    preprocess_images(raw_image_dir, processed_image_dir)