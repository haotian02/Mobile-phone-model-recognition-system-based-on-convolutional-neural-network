import json
import io
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import wandb
import urllib.request
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import PIL.Image
import PIL.ExifTags
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from PIL import ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

class PhoneClassifier:
    def __init__(self, dataset_dir, model_path, batch_size, epochs, step_size, gamma, load_method, save):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device', self.device)
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.step_size = step_size
        self.gamma = gamma

        # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
        self.train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

        # 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
        self.test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

        # 数据集文件夹路径
        self.train_path = os.path.join(dataset_dir, 'train')
        self.test_path = os.path.join(dataset_dir, 'val')
        print('训练集路径', self.train_path)
        print('测试集路径', self.test_path)

        # 载入训练集
        self.train_dataset = datasets.ImageFolder(self.train_path, self.train_transform)
        # 载入测试集
        self.test_dataset = datasets.ImageFolder(self.test_path, self.test_transform)

        # 各类别名称
        self.class_names = self.train_dataset.classes
        self.n_class = len(self.class_names)
        # 映射关系：索引号 到 类别
        self.idx_to_labels = {y: x for x, y in self.train_dataset.class_to_idx.items()}

        # 保存为本地的 npy 文件
        np.save(os.path.join(save, 'idx_to_labels.npy'), self.idx_to_labels)
        np.save(os.path.join(save, 'labels_to_idx.npy'), self.train_dataset.class_to_idx)


        self.BATCH_SIZE = batch_size

        # 训练集的数据加载器
        self.train_loader = DataLoader(self.train_dataset,
                          batch_size=self.BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                         )

        # 测试集的数据加载器
        self.test_loader = DataLoader(self.test_dataset,
                         batch_size=self.BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                        )

        if load_method == 'load_model':
            self.model, self.optimizer = self.load_model(model_path, self.n_class)
        elif load_method == 'init_and_load_model':
            self.model, self.optimizer = self.init_and_load_model(model_path, self.n_class)
        elif load_method == 'download_and_load_model':
            self.model, self.optimizer = self.download_and_load_model(model_path, self.n_class)
        else:
            raise ValueError(f'Unknown load method: {load_method}')

        self.model, self.criterion, self.lr_scheduler, self.EPOCHS = self.setup_training()
        self.best_test_accuracy = 0

    def load_model(self, model_path, n_class):
        if 'resnet34.pth' in model_path:
            model = models.resnet34(pretrained=False)
        elif 'resnet50.pth' in model_path:
            model = models.resnet50(pretrained=False)
        else:
            print("Invalid model path")
            return None, None
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.fc = nn.Linear(model.fc.in_features, n_class)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def init_and_load_model(self, model_path, n_class):
        if 'resnet34.pth' in model_path:
            model = models.resnet34(pretrained=False)
        elif 'resnet50.pth' in model_path:
            model = models.resnet50(pretrained=False)
        else:
            print("Invalid model path")
            return None, None
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.fc = nn.Linear(model.fc.in_features, n_class)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def download_and_load_model(self, model_path, n_class):
        if 'resnet34.pth' in model_path:
            model = models.resnet34(pretrained=False)
        elif 'resnet50.pth' in model_path:
            model = models.resnet50(pretrained=False)
        else:
            print("Invalid model path")
            return None, None
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.fc = nn.Linear(model.fc.in_features, n_class)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def setup_training(self):
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        return self.model, self.criterion, self.lr_scheduler, self.epochs

    def apply_exif_orientation(self, image):
        try:
            exif = image._getexif()
        except AttributeError:
            exif = None

        if exif is None:
            return image

        exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif.items() if k in PIL.ExifTags.TAGS}
        orientation = exif.get('Orientation', None)

        if orientation == 1:
            return image
        elif orientation == 2:
            return PIL.ImageOps.mirror(image)
        elif orientation == 3:
            return image.transpose(PIL.Image.ROTATE_180)
        elif orientation == 4:
            return PIL.ImageOps.flip(image)
        elif orientation == 5:
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
        elif orientation == 6:
            return image.transpose(PIL.Image.ROTATE_270)
        elif orientation == 7:
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
        elif orientation == 8:
            return image.transpose(PIL.Image.ROTATE_90)
        else:
            return image

    def train_one_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        images = self.apply_exif_orientation(images)
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        log_train = {}
        log_train['epoch'] = self.epoch
        log_train['batch'] = self.batch_idx
        log_train['train_loss'] = loss
        log_train['train_accuracy'] = accuracy_score(labels, preds)
        log_train['train_precision'] = precision_score(labels, preds, average='macro')
        log_train['train_recall'] = recall_score(labels, preds, average='macro')
        log_train['train_f1-score'] = f1_score(labels, preds, average='macro')
        return log_train

    def evaluate_testset(self):
        loss_list = []
        labels_list = []
        preds_list = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                loss = self.criterion(outputs, labels)
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)
        log_test = {}
        log_test['epoch'] = self.epoch
        log_test['test_loss'] = np.mean(loss_list)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
        return log_test

    def train(self, callback):
        self.epoch = 0
        self.batch_idx = 0
        self.best_test_accuracy = 0

        # 训练日志-训练集
        df_train_log = pd.DataFrame()
        log_train = {}
        log_train['epoch'] = 0
        log_train['batch'] = 0
        images, labels = next(iter(self.train_loader))
        log_train.update(self.train_one_batch(images, labels))
        df_train_log = df_train_log._append(log_train, ignore_index=True)

        # 训练日志-测试集
        df_test_log = pd.DataFrame()
        log_test = {}
        log_test['epoch'] = 0
        log_test.update(self.evaluate_testset())
        df_test_log = df_test_log._append(log_test, ignore_index=True)

        # 创建wandb可视化项目
        # wandb.init(project='phone', name=time.strftime('%m%d%H%M%S'))

        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')

        for self.epoch in range(1, self.EPOCHS + 1):
            callback(f'Epoch {self.epoch}/{self.EPOCHS}\n')

            # 训练阶段
            self.model.train()
            for images, labels in tqdm(self.train_loader):  # 获得一个 batch 的数据和标注
                self.batch_idx += 1
                log_train = self.train_one_batch(images, labels)
                df_train_log = df_train_log._append(log_train, ignore_index=True)
                # wandb.log(log_train)

            self.lr_scheduler.step()

            # 测试阶段
            self.model.eval()
            log_test = self.evaluate_testset()
            df_test_log = df_test_log._append(log_test, ignore_index=True)
            # wandb.log(log_test)

            # 保存最新的最佳模型文件
            if log_test['test_accuracy'] > self.best_test_accuracy:
                # 删除旧的最佳模型文件(如有)
                old_best_checkpoint_path = f'checkpoint/best-{self.best_test_accuracy:.3f}.pth'
                if os.path.exists(old_best_checkpoint_path):
                    os.remove(old_best_checkpoint_path)

                self.best_test_accuracy = log_test['test_accuracy']
                new_best_checkpoint_path = f'checkpoint/best-{self.best_test_accuracy:.3f}.pth'
                # new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
                torch.save(self.model, new_best_checkpoint_path)
                callback(f'保存新的最佳模型 {new_best_checkpoint_path}\n')

        df_train_log.to_csv(os.path.join(self.dataset_dir, '训练日志-训练集.csv'), index=False)
        df_test_log.to_csv(os.path.join(self.dataset_dir, '训练日志-测试集.csv'), index=False)

        # 载入最佳模型作为当前模型
        self.model = torch.load(f'checkpoint/best-{self.best_test_accuracy:.3f}.pth')
        self.model.eval()
        callback(str(self.evaluate_testset()) + '\n')

if __name__ == '__main__':
    # 从文件中读取参数
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    # 创建一个 PhoneClassifier 对象
    classifier = PhoneClassifier(dataset_dir=parameters['dataset_dir'],
                                 model_path=parameters['model_path'],
                                 batch_size=parameters['batch_size'],
                                 epochs=parameters['epochs'],
                                 step_size=parameters['step_size'],
                                 gamma=parameters['gamma'],
                                 load_method=parameters['load_method'],
                                 save=parameters['dataset_dir']
                                 )
    update_output = io.StringIO()
    # 开始训练
    classifier.train(update_output)