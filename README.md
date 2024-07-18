# 基于卷积神经网络的手机型号识别系统

## 项目概述

在智能手机飞速发展的今天，手机型号识别对于二手手机回收业务尤为重要。传统的人工识别方法效率低、准确率不高。为了解决这些问题，本项目设计并实现了一个基于卷积神经网络（CNN）的手机型号识别系统，能够自动、快速、准确地识别手机型号，具有较高的实用价值和应用前景。

## 功能特点

1. **数据预处理**：通过网络爬虫技术从网页上下载智能手机图片，结合开源的手机型号图像数据集，使用感知哈希算法和明汉距离去除重复图像，并统一图像颜色格式，构建出一个高质量的手机型号数据集。
2. **模型构建与训练**：采用ResNet50残差网络结构，并添加注意力机制，提高模型对图像中重要特征和区域的关注度，增强识别精度。利用迁移学习加快模型的收敛速度，降低训练难度。
3. **用户界面设计**：使用Tkinter设计简洁美观的用户界面，方便用户进行手机型号识别操作。界面包括数据集划分、模型训练、手机型号识别等功能模块。

## 系统架构

本系统基于Python语言和PyTorch框架开发，包括以下主要模块：
- **数据预处理模块**：负责下载、清洗和处理手机图像数据集。
- **模型训练模块**：构建和训练卷积神经网络模型。
- **识别界面模块**：提供用户上传手机图片并识别手机型号的图形界面。

## 使用方法

1. **数据集准备**：运行数据预处理脚本，构建手机型号图像数据集。
2. **模型训练**：通过训练界面调节模型训练参数，训练卷积神经网络模型。
3. **手机型号识别**：在识别界面上传或拍照获取手机图片，系统将显示识别结果和准确率。

## 主要技术

- **卷积神经网络（CNN）**：使用ResNet50残差网络结构，结合注意力机制。
- **迁移学习**：提高模型训练效率和精度。
- **Tkinter**：开发用户操作界面。

## 目录结构

```plaintext
.
├── checkpoints
│   ├── Apple.pth
│   ├── Samsung.pth
│   ├── ... 
│   ├── Resnet34-CBAM-fc.pth
│   ├── Resnet34-CBAM-all.pth
│   ├── Resnet50-CBAM-fc.pth  
│   ├── Resnet50-CBAM-all.pth
│   └── ...
├── statistics.py
├── train1.py 
├── test1.py
├── ui.py
├── README.md
└── requirements.txt
```

## 手机型号

- Apple iPhone 11、Apple iPhone 11 Pro Max、Apple iphone 13、Apple iPhone 14 Pro Max、Apple iPhone 6s plus、Apple iPhone 7 plus、Apple iPhone 8 plus、Apple iPhone SE2、Apple iPhone XR、Apple iPhone XS Max
- Samsung note10、Samsung note20 ultra、Samsung note8、Samsung Samsung note9、Samsung s10、Samsung s8、Samsung s20、Samsung s20ultra、Samsung s21、Samsung s21 ultra、Samsung s22 ultra
- vivo s15 pro、vivo s16 pro、vivo x50 pro、vivo x60 pro、vivo x70 pro、vivo x80 pro、vivo x90 pro
- 一加 10 pro、一加 11、一加 7 pro、一加 8 pro、一加 9 pro、一加 ace 2v、一加 ace pro
- iqoo 10 pro、iqoo 8 pro、iqoo 9 pro
- oppo findx2 pro、oppo findx3 pro、oppo findx5 pro、oppo findx6 pro、oppo reno6 pro、oppo reno7 pro、oppo reno8 pro
- realme 11 pro、realme GT2 pro、realme GTneo3、realme GTneo5、realme Q3 pro
- 努比亚 z40 pro、努比亚 z40s pro、努比亚 z50 ultra、努比亚 z50s pro
- 小米 10 pro、小米 11 pro、小米 11 ultra、小米 12 pro、小米 12s ultra、小米 13 pro、小米 13 ultra、小米 6、小米 8、小米 9 pro
- 华为 mate20 pro、华为 mate30 pro、华为 mate40 pro、华为 mate50 pro、华为 nova10 pro、华为 nova11 pro、华为 nova8 pro、华为 nova9 pro、华为 p20 pro、华为 p30 pro、华为 p40 pro、华为 p50 pro、华为 p60 pro
- 红米 k20 pro、红米 k30 pro、红米 k40 pro、红米 k50 pro、红米 k60 pro、红米 note12 turbo、红米 note8 pro、红米 note9 pro
- 荣耀 100 pro、荣耀 50 pro、荣耀 60 pro、荣耀 70 pro、荣耀 80 pro、荣耀 90 pro、荣耀 magic4 pro、荣耀 magic5 pro
- 魅族 16 plus、魅族 16s pro、魅族 17 pro、魅族 18、魅族 18 pro、魅族 20 pro、魅族 pro7 plus

## 环境依赖

	•	Python 3.x
	•	PyTorch
	•	Tkinter
	•	其他依赖请参见requirements.txt

## 参考文献

	•	A. G. Biney and H. Sellahewa, “Analysis of smartphone model identification using digital images,” presented at the 20th IEEE International Conference on Image Processing, Melbourne, Australia, 2015. 
	•	韩红桂,甄琪,任柯燕等.基于孪生卷积神经网络的手机型号识别方法[J].北京工业大学学报,2021,47(02):112-119. 
	•	付勇刚.基于卷积神经网络的废旧手机型号识别与应用研究[D].合肥：合肥工业大学，2022. 

## 需要数据集的可以私我
