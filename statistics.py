import io
import shutil
import random
import pandas as pd
import json
import os
import time
from PIL import Image

class DatasetSplitter:
    def __init__(self, dataset_path, test_frac):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_path.split('_')[0]
        self.classes = os.listdir(dataset_path)
        self.test_frac = test_frac
        self.df = pd.DataFrame()

    def create_folders(self):
        train_dir = os.path.join(self.dataset_path, 'train')
        val_dir = os.path.join(self.dataset_path, 'val')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        for phone in self.classes:
            train_phone_dir = os.path.join(train_dir, phone)
            val_phone_dir = os.path.join(val_dir, phone)
            if not os.path.exists(train_phone_dir):
                os.mkdir(train_phone_dir)
            if not os.path.exists(val_phone_dir):
                os.mkdir(val_phone_dir)

    def split_dataset(self, callback):
        random.seed(123)
        callback('{:^18} {:^18} {:^18}\n'.format('类别', '训练集数据个数', '测试集数据个数'))
        for phone in self.classes:
            old_dir = os.path.join(self.dataset_path, phone)
            images_filename = os.listdir(old_dir)
            random.shuffle(images_filename)
            testset_numer = int(len(images_filename) * self.test_frac)
            testset_images = images_filename[:testset_numer]
            trainset_images = images_filename[testset_numer:]
            for image in testset_images:
                old_img_path = os.path.join(self.dataset_path, phone, image)
                new_test_dir = os.path.join(self.dataset_path, 'val', phone)
                if not os.path.exists(new_test_dir):
                    os.makedirs(new_test_dir)
                new_test_path = os.path.join(new_test_dir, image)
                if os.path.isfile(old_img_path):  # 检查 old_img_path 是否指向一个文件
                    shutil.copy(old_img_path, new_test_path)
            for image in trainset_images:
                old_img_path = os.path.join(self.dataset_path, phone, image)
                new_train_dir = os.path.join(self.dataset_path, 'train', phone)
                if not os.path.exists(new_train_dir):
                    os.makedirs(new_train_dir)
                new_train_path = os.path.join(new_train_dir, image)
                if os.path.isfile(old_img_path):
                    shutil.copy(old_img_path, new_train_path)
            callback('{:^18} {:^18} {:^18}\n'.format(phone, len(trainset_images), len(testset_images)))
            self.df = self.df._append({'class': phone, 'trainset': len(trainset_images), 'testset': len(testset_images)},
                                     ignore_index=True)

    def save_statistics(self):
        self.df['total'] = self.df['trainset'] + self.df['testset']
        self.df.to_csv(os.path.join(self.dataset_path, '数据量统计.csv'), index=False)

    def move_folders(self, path):
        if os.path.exists(os.path.join(path, 'phone split')):
            os.makedirs(os.path.join(path, 'phone list'), exist_ok=True)
            shutil.move(os.path.join(path, 'phone split', 'train'), os.path.join(path, 'phone list', 'train'))
            shutil.move(os.path.join(path, 'phone split', 'val'), os.path.join(path, 'phone list', 'val'))

    # 差异哈希计算函数变成实例方法
    def dHash(self, img):
        img = img.resize((16, 16))
        gray = img.convert('L')
        hash_str = ''
        for i in range(15):
            for j in range(15):
                if gray.getpixel((j, i)) > gray.getpixel((j + 1, i)):
                    hash_str += '1'
                else:
                    hash_str += '0'
        for i in range(15):
            for j in range(15):
                if gray.getpixel((i, j)) > gray.getpixel((i, j + 1)):
                    hash_str += '1'
                else:
                    hash_str += '0'
        return hash_str

    # 汉明距离计算函数变成实例方法
    def hamming_distance(self, hash1, hash2):
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    # WEBP转JPEG函数变成实例方法
    def convert_webp_to_jpg(self, webp_path):
        img = Image.open(webp_path)
        jpg_path = webp_path.replace(".webp", ".jpg")
        os.remove(webp_path)
        return jpg_path

    # 去除重复图像并转换颜色格式的函数
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
                    hash_ = self.dHash(img)
                    if any(self.hamming_distance(hash_, saved_hash) < threshold for saved_hash in hashes):
                        os.remove(img_path)
                        callback(f"已删除重复图像: {img_path}\n")
                        # print(f"已删除重复图像: {img_path}", file=output)
                    else:
                        hashes[hash_] = img_path
                        img.convert('RGB').save(img_path, 'JPEG')
                        callback(f"转化颜色格式成功：{img_path}\n")
                        # print(f"转化颜色格式成功：{img_path}", file=output)
                except IOError:
                    callback(f"无法读取图像: {img_path}, 跳过。\n")
                    # print(f"无法读取图像: {img_path}, 跳过。", file=output)

    # 将目录下所有WEBP图像转换为JPEG格式的函数
    def convert_images_to_jpg(self, directory=None):
        if directory is None:
            directory = self.dataset_path
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.webp'):
                    self.convert_webp_to_jpg(os.path.join(root, file))

if __name__ == '__main__':
    # 从文件中读取参数
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    # 使用从parameters字典中获取的值
    dataset_path = parameters['dataset_path']
    test_frac = parameters['test_frac']

    classifiers = DatasetSplitter(dataset_path=dataset_path, test_frac=test_frac)

    update_output = io.StringIO()

    classifiers.create_folders()
    classifiers.split_dataset(update_output)
    classifiers.remove_duplicates_and_convert_images(update_output)
    classifiers.save_statistics()
    classifiers.move_folders('./')