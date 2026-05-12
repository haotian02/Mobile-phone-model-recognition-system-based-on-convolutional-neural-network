# 基于卷积神经网络的手机型号识别系统

这是一个面向手机外观图像的型号识别项目。系统基于 PyTorch、ResNet 和 CBAM 注意力机制实现模型训练与推理，并提供一个本地 Web 管理页面，用于完成数据集处理、模型训练和图片识别等操作。

项目适合用于二手手机回收、手机型号辅助识别、图像分类课程设计和深度学习实践。

## 功能特点

- **手机型号识别**：输入手机图片后，输出置信度最高的 Top-K 型号预测结果。
- **实时摄像头检测**：支持调用本机摄像头进行实时画面识别。
- **模型训练**：支持基于 ImageFolder 结构的数据集训练 ResNet/CBAM 分类模型。
- **迁移学习**：可加载 ResNet34、ResNet50 或 CBAM 版本模型权重继续微调。
- **数据集处理**：支持按比例划分训练集和验证集，也支持图片去重与格式转换。
- **本地 Web 页面**：通过浏览器访问本地服务，完成账号、识别、训练和数据集处理操作。

## 技术栈

- Python
- PyTorch / TorchVision
- ResNet34 / ResNet50
- CBAM 注意力机制
- OpenCV
- Pillow
- scikit-learn
- pandas / numpy
- Python 标准库 HTTP Server

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── resnet_cbam.py          # ResNet + CBAM 模型结构
├── statistics.py           # 数据集划分、统计、去重与图片格式转换
├── train1.py               # 模型训练脚本
├── test1.py                # 图片与摄像头推理脚本
├── ui.py                   # 本地 Web 页面入口
├── SimHei.ttf              # 中文字体
├── parameters.json         # 运行时参数文件，本地运行后生成或更新
├── credentials.json        # 本地账号文件，本地运行后生成或更新
├── checkpoint/             # 模型权重目录
└── phone data/             # 数据集目录，本地按需准备
```

## 环境安装

建议使用 Python 3.9 或更高版本，并在虚拟环境中安装依赖。

```bash
pip install -r requirements.txt
```

如果使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch 版本。详见 PyTorch 官方安装页面。

## 数据集格式

训练脚本使用 `torchvision.datasets.ImageFolder` 读取数据。数据集目录需要包含 `train` 和 `val` 两个子目录，每个类别一个文件夹：

```text
phone dataset/
├── train/
│   ├── Apple iPhone 11/
│   ├── Huawei Mate 40 Pro/
│   └── ...
└── val/
    ├── Apple iPhone 11/
    ├── Huawei Mate 40 Pro/
    └── ...
```

如果原始数据还没有划分，可以通过 Web 页面中的「数据集处理」功能进行划分。

## 快速开始

启动本地 Web 页面：

```bash
python ui.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

如需更换端口，可以设置环境变量：

```bash
set PHONE_UI_PORT=8080
python ui.py
```

## 使用说明

### 1. 图片识别

进入「型号识别」页面后，可以选择预设厂商，也可以手动填写：

- 模型路径，例如 `./checkpoint/Resnet50-CBAM-all.pth`
- 标签路径，例如 `./phone data/phone dataset/idx_to_labels.npy`
- 图片路径，或直接上传图片

系统会返回置信度最高的若干个识别结果。

### 2. 模型训练

进入「模型训练」页面后，填写：

- 数据集目录
- 预训练模型路径
- Batch Size
- Epochs
- Step Size
- Gamma
- 加载方式

训练完成后会在数据集输出目录下保存：

- `idx_to_labels.npy`
- `labels_to_idx.npy`
- `train_log.csv`
- `val_log.csv`
- `checkpoint/best-*.pth`

### 3. 数据集处理

进入「数据集处理」页面后，可以执行：

- 按比例划分训练集和验证集
- 图片去重
- 图片颜色格式转换

## 命令行运行

也可以不使用 Web 页面，直接运行脚本。

训练模型：

```bash
python train1.py
```

图片识别：

```bash
python test1.py
```

这两个脚本会读取项目根目录下的 `parameters.json`。请确保其中包含相应字段，例如：

```json
{
  "dataset_dir": "./phone data/phone dataset",
  "model_path": "./resnet50.pth",
  "batch_size": 16,
  "epochs": 10,
  "step_size": 5,
  "gamma": 0.1,
  "load_method": "init_and_load_model",
  "save": "./phone data/phone dataset"
}
```

图片识别所需参数示例：

```json
{
  "idx_to_labels_path": "./phone data/phone dataset/idx_to_labels.npy",
  "model1_path": "./checkpoint/Resnet50-CBAM-all.pth",
  "image_path": "./image/demo.jpg"
}
```

## 支持的品牌示例

项目中预设了多个品牌模型路径，包括：

- 通用
- Apple
- Samsung
- vivo
- iqoo
- OPPO
- realme
- 红米
- 华为
- 荣耀
- 小米
- 一加
- 魅族
- 努比亚

实际可识别的型号取决于训练数据和对应模型权重。

## 注意事项

- 大体积数据集和模型权重可能不会完整放在 GitHub 仓库中，需要在本地准备。
- 运行摄像头识别前，请确认 OpenCV 能正常访问本机摄像头。
- Web 页面是本地服务，不建议直接暴露到公网。
- 训练时建议使用 GPU，否则 ResNet/CBAM 模型训练速度会较慢。
- `credentials.json` 仅用于本地页面的简单账号功能，不适合作为生产环境认证方案。

## 许可证

本项目主要用于学习、研究和课程设计场景。使用数据集和模型权重时，请遵守对应数据来源的授权要求。
