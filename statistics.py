import io
import shutil
import random
import numpy as np
import pandas as pd
import json
import os
from PIL import Image


def dhash(img, hash_size=16):
    """Compute difference hash (dHash) for an image using numpy."""
    img = img.resize((hash_size + 1, hash_size), Image.LANCZOS).convert("L")
    pixels = np.array(img, dtype=np.float32)
    # Horizontal differences (hash_size * hash_size bits)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(diff.flatten().astype(np.uint8).astype(str))


def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


class DatasetSplitter:
    def __init__(self, dataset_path, test_frac):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(dataset_path)
        self.classes = [d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))
                        and d not in ("train", "val")]
        self.test_frac = test_frac

    def create_folders(self):
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        for phone in self.classes:
            os.makedirs(os.path.join(train_dir, phone), exist_ok=True)
            os.makedirs(os.path.join(val_dir, phone), exist_ok=True)

    def split_dataset(self, callback):
        random.seed(123)
        callback('{:^18} {:^18} {:^18}\n'.format('类别', '训练集数据个数', '测试集数据个数'))
        records = []

        for phone in self.classes:
            old_dir = os.path.join(self.dataset_path, phone)
            images_filename = os.listdir(old_dir)
            random.shuffle(images_filename)
            test_n = int(len(images_filename) * self.test_frac)
            test_images = images_filename[:test_n]
            train_images = images_filename[test_n:]

            for image in test_images:
                shutil.move(
                    os.path.join(self.dataset_path, phone, image),
                    os.path.join(self.dataset_path, 'val', phone, image),
                )
            for image in train_images:
                shutil.move(
                    os.path.join(self.dataset_path, phone, image),
                    os.path.join(self.dataset_path, 'train', phone, image),
                )
            callback('{:^18} {:^18} {:^18}\n'.format(phone, len(train_images), len(test_images)))
            records.append({'class': phone, 'trainset': len(train_images), 'testset': len(test_images)})

            # Remove class source dir if empty
            try:
                os.rmdir(os.path.join(self.dataset_path, phone))
            except OSError:
                pass

        self.df = pd.DataFrame(records)

    def save_statistics(self):
        if len(self.df):
            self.df['total'] = self.df['trainset'] + self.df['testset']
            self.df.to_csv(os.path.join(self.dataset_path, '数据量统计.csv'), index=False)

    def move_folders(self, path):
        src = os.path.join(path, 'phone split')
        dst = os.path.join(path, 'phone list')
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            for sub in ('train', 'val'):
                s = os.path.join(src, sub)
                if os.path.exists(s):
                    shutil.move(s, os.path.join(dst, sub))

    @staticmethod
    def convert_webp_to_jpg(webp_path):
        img = Image.open(webp_path)
        jpg_path = webp_path.replace(".webp", ".jpg")
        os.remove(webp_path)
        return jpg_path

    @staticmethod
    def convert_images_to_jpg(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.webp'):
                    DatasetSplitter.convert_webp_to_jpg(os.path.join(root, file))

    def remove_duplicates_and_convert_images(self, callback, directory=None, threshold=10):
        if directory is None:
            directory = self.dataset_path
        hashes = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.webp'):
                    img_path = self.convert_webp_to_jpg(os.path.join(root, file))
                elif file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(root, file)
                else:
                    continue

                try:
                    img = Image.open(img_path).convert('RGBA')
                    hash_ = dhash(img)
                    if any(hamming_distance(hash_, saved_hash) < threshold for saved_hash in hashes):
                        os.remove(img_path)
                        callback(f"已删除重复图像: {img_path}\n")
                    else:
                        hashes[hash_] = img_path
                        img.convert('RGB').save(img_path, 'JPEG')
                        callback(f"转化颜色格式成功：{img_path}\n")
                except IOError:
                    callback(f"无法读取图像: {img_path}, 跳过。\n")


if __name__ == '__main__':
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    splitter = DatasetSplitter(
        dataset_path=parameters['dataset_path'],
        test_frac=parameters['test_frac'],
    )

    buffer = io.StringIO()
    splitter.create_folders()
    splitter.split_dataset(buffer.write)
    splitter.remove_duplicates_and_convert_images(buffer.write)
    splitter.save_statistics()
    splitter.move_folders('./')
