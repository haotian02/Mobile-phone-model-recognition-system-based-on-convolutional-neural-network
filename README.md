# 基于卷积神经网络的手机型号识别系统

这是一个面向手机外观图像的型号识别项目。系统基于 PyTorch、ResNet 和 CBAM 注意力机制实现模型训练与推理，并提供一个 iOS 风格本地 Web 管理页面，用于完成账号管理、数据集处理、模型训练和图片识别等操作。

项目适合用于二手手机回收、手机型号辅助识别、图像分类课程设计和深度学习实践。

## 功能特点

- **手机型号识别**：输入手机图片后，输出置信度最高的 Top-K 型号预测结果。
- **实时摄像头检测**：支持调用本机摄像头进行实时画面识别，叠加 FPS 和预测标签。
- **模型训练**：基于 ImageFolder 结构的数据集训练 ResNet/CBAM 分类模型，支持混合精度训练（AMP）、梯度裁剪、余弦退火学习率调度和早停机制。
- **迁移学习**：可加载 ResNet34、ResNet50 或 CBAM 版本预训练权重继续微调。
- **数据集处理**：支持按比例划分训练集/验证集、图片去重（dHash + 汉明距离）、WebP 转 JPEG 及统一 RGB 格式。
- **本地 Web 页面**：iOS 风格界面，通过浏览器访问本地服务，完成账号注册/登录、型号识别、模型训练和数据集处理操作。
- **实验追踪（可选）**：支持 wandb 记录训练指标和曲线图。

## 技术栈

- Python 3.9+
- PyTorch / TorchVision
- ResNet34 / ResNet50
- CBAM 注意力机制（Channel + Spatial Attention）
- OpenCV
- Pillow
- scikit-learn
- pandas / numpy
- wandb（可选）
- Python 标准库 HTTP Server（ThreadingHTTPServer）

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── resnet_cbam.py          # ResNet + CBAM 模型结构定义
├── statistics.py           # 数据集划分、统计、去重与图片格式转换
├── train1.py               # 模型训练脚本（PhoneClassifier）
├── test1.py                # 图片与摄像头推理脚本（VideoProcessor）
├── ui.py                   # 本地 Web 管理页面入口
├── SimHei.ttf              # 中文字体（用于推理标注和训练曲线）
├── parameters.json         # 运行时参数文件，由 Web 页面或手动创建
├── credentials.json        # 本地账号文件，由 Web 页面注册生成
├── checkpoint/             # 模型权重目录
├── uploads/                # Web 页面上传图片存放目录
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

如果原始数据还没有划分，可以通过 Web 页面中的「数据集处理」功能进行划分，也可以直接运行 `statistics.py`。

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
# Windows
set PHONE_UI_PORT=8080
python ui.py

# Linux / macOS
PHONE_UI_PORT=8080 python ui.py
```

首次使用需要注册账号（需提供用户名、密码和手机号），登录后即可使用全部功能。

## Web 页面功能

### 控制台
展示模型数量、预设厂商数量和训练状态概览，以及最近一次使用的参数。

### 型号识别
- 选择厂商预设（14 个品牌），自动填入对应的标签文件和模型文件路径
- 支持手动填写模型路径、标签路径和图片路径
- 支持上传本地图片
- 识别结果以置信度百分比展示

### 模型训练
支持配置以下参数：
- **数据集目录**：包含 `train/` 和 `val/` 子目录的路径
- **预训练模型路径**：`.pth` 权重文件路径
- **Batch Size**：批大小（默认 16）
- **Epochs**：训练轮数（默认 10）
- **Step Size / Gamma**：学习率调度参数（已保留，当前使用 CosineAnnealingLR）
- **加载方式**：
  - `init_and_load_model`：微调 CBAM 模型，仅替换分类头
  - `load_model`：微调普通 ResNet
  - `download_and_load_model`：微调 CBAM 全部层

训练在后台线程运行，日志实时输出到页面。训练完成后输出目录包含：

- `idx_to_labels.npy` — 索引到标签映射
- `labels_to_idx.npy` — 标签到索引映射
- `train_log.csv` — 每 batch 训练指标
- `val_log.csv` — 每 epoch 验证指标
- `training_curves.png` — 训练曲线图（Loss / Accuracy / Precision+Recall / F1）
- `checkpoint/best-*.pth` — 最佳验证准确率模型权重

### 数据集处理
- **划分数据集**：按比例将图片分为训练集和验证集（默认 20% 验证集）
- **去重与格式转换**：使用 dHash 算法检测并删除重复图片，统一转换为 RGB JPEG 格式

### 账号
查看当前用户、退出登录、修改密码。

## 命令行运行

也可以不使用 Web 页面，直接运行脚本。两个脚本均读取项目根目录下的 `parameters.json`。

### 训练模型

```bash
python train1.py
```

`parameters.json` 训练所需字段：

```json
{
  "dataset_dir": "./phone data/phone dataset",
  "model_path": "./resnet50.pth",
  "batch_size": 16,
  "epochs": 10,
  "step_size": 5,
  "gamma": 0.1,
  "load_method": "init_and_load_model",
  "save": "./phone data/phone dataset",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "patience": 5,
  "num_workers": 8
}
```

### 图片识别

```bash
python test1.py
```

`parameters.json` 推理所需字段：

```json
{
  "idx_to_labels_path": "./phone data/phone dataset/idx_to_labels.npy",
  "model1_path": "./checkpoint/Resnet50-CBAM-all.pth",
  "image_path": "./image/demo.jpg"
}
```

## 支持的品牌预设

Web 页面中预设了 14 个品牌路径：

| 品牌 | 标签/模型路径 |
|------|-------------|
| 通用 | `phone data/phone dataset/` + `checkpoint/Resnet50-CBAM-all.pth` |
| Apple | `phone data/phone name/Apple/` + `checkpoint/Apple.pth` |
| Samsung | `phone data/phone name/Samsung/` + `checkpoint/Samsung.pth` |
| vivo | `phone data/phone name/vivo/` + `checkpoint/vivo.pth` |
| iqoo | `phone data/phone name/iqoo/` + `checkpoint/iqoo.pth` |
| oppo | `phone data/phone name/oppo/` + `checkpoint/oppo.pth` |
| realme | `phone data/phone name/realme/` + `checkpoint/realme.pth` |
| 华为 | `phone data/phone name/华为/` + `checkpoint/华为.pth` |
| 荣耀 | `phone data/phone name/荣耀/` + `checkpoint/荣耀.pth` |
| 小米 | `phone data/phone name/小米/` + `checkpoint/小米.pth` |
| 红米 | `phone data/phone name/红米/` + `checkpoint/红米.pth` |
| 一加 | `phone data/phone name/一加/` + `checkpoint/一加.pth` |
| 魅族 | `phone data/phone name/魅族/` + `checkpoint/魅族.pth` |
| 努比亚 | `phone data/phone name/努比亚/` + `checkpoint/努比亚.pth` |

实际可识别的型号取决于训练数据和对应模型权重。

## 注意事项

- 大体积数据集和模型权重可能不会完整放在 GitHub 仓库中，需要在本地准备。
- 运行摄像头识别前，请确认 OpenCV 能正常访问本机摄像头。
- Web 页面是本地服务，默认绑定 `127.0.0.1`，不建议直接暴露到公网。
- 训练时建议使用 GPU，否则 ResNet/CBAM 模型训练速度会较慢。代码默认启用 CUDA。
- `credentials.json` 仅用于本地页面的简单账号功能，密码经 SHA-256 哈希存储，不适合作为生产环境认证方案。
- 训练使用混合精度（AMP）加速，梯度裁剪 max_norm=1.0，学习率调度为 CosineAnnealingLR。

## 许可证

本项目主要用于学习、研究和课程设计场景。使用数据集和模型权重时，请遵守对应数据来源的授权要求。
