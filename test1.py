import json
import os
import time
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
import resnet_cbam

class VideoProcessor:
    def __init__(self, model1_path, idx_to_labels_path, image_path, font_path='./SimHei.ttf'):
        self.image_path = image_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.font = ImageFont.truetype(font_path, 32)
        self.idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()

        self.model = torch.load(model1_path, map_location=self.device)
        self.model = self.model.eval().to(self.device)

        # 测试集图像预处理：缩放裁剪、转 Tensor、归一化
        self.test_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
                                                      mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                                 ])

        self.stop = False

    def process_frame(self, img):
        # 记录该帧开始处理的时间
        start_time = time.time()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        img_pil = Image.fromarray(img_rgb)  # array 转 PIL
        input_img = self.test_transform(img_pil).unsqueeze(0).to(self.device)  # 预处理
        pred_logits = self.model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

        top_n = torch.topk(pred_softmax, 5)  # 取置信度最大的 n 个结果
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
        confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度
        split_dataset_duration = time.time() - start_time
        # print("摄像头识别操作耗时：", split_dataset_duration, "秒")

        # 使用PIL绘制中文
        draw = ImageDraw.Draw(img_pil)
        # 在图像上写字
        for i in range(len(confs)):
            pred_class = self.idx_to_labels[pred_ids[i]]
            text = '{:<15} {:>.2f}%'.format(pred_class, confs[i]*100)
            # 文字坐标，中文字符串，字体，bgra颜色
            draw.text((50, 100 + 50 * i), text, font=self.font, fill=(255, 0, 0, 1))
        img = np.array(img_pil)  # PIL 转 array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR

        start_time = time.time()

        # 添加一个小的延迟
        time.sleep(0.001)
        end_time = time.time()
        # 检查end_time和start_time是否相等
        if end_time == start_time:
            FPS = 0
        else:
            FPS = 1 / (end_time - start_time)
            # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
        img = cv2.putText(img,
                          'FPS  ' + str(int(FPS)),
                          (50, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 2,
                          (0, 0, 255),
                          4,
                          cv2.LINE_AA)
        return img

    def run(self):
        # 获取摄像头，传入0表示获取系统默认摄像头
        cap = cv2.VideoCapture(0)
        # 打开cap
        cap.open(0)
        # 无限循环，直到break被触发
        while cap.isOpened() and not self.stop:
            # 获取画面
            success, frame = cap.read()
            if not success:
                print('Error')
                break
            frame = self.process_frame(frame)
            # 展示处理后的三通道图像
            cv2.imshow('phone', frame)
            if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出
                break
        # 关闭摄像头
        cap.release()
        # 关闭图像窗口
        cv2.destroyAllWindows()

    def capture_and_predict(self):
        # 获取摄像头，传入0表示获取系统默认摄像头
        cap = cv2.VideoCapture(0)
        # 打开cap
        cap.open(0)
        # 获取一帧图像
        success, frame = cap.read()
        if not success:
            print('Error')
            return
        # 关闭摄像头
        cap.release()
        # 将捕获的帧保存为临时图像文件
        temp_image_path = 'temp.jpg'
        cv2.imwrite(temp_image_path, frame)
        # 使用 predict_image 方法处理临时图像文件
        results = self.predict_image(temp_image_path)
        # 删除临时图像文件
        os.remove(temp_image_path)
        return results

    def predict_image(self, image_path):
        start_time = time.time()
        # 加载图像
        img_pil = Image.open(image_path)

        # 预处理图像
        input_img = self.test_transform(img_pil).unsqueeze(0).to(self.device)

        # 使用模型进行预测
        pred_logits = self.model(input_img)
        pred_softmax = F.softmax(pred_logits, dim=1)

        # 获取置信度最高的前5个结果
        top_n = torch.topk(pred_softmax, 5)
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()
        confs = top_n[0].cpu().detach().numpy().squeeze()

        split_dataset_duration = time.time() - start_time
        # print("图片识别操作耗时：", split_dataset_duration, "秒")

        # 将结果转换为文本
        results = []
        for i in range(len(confs)):
            pred_class = self.idx_to_labels[pred_ids[i]]
            results.append(f'{pred_class}: {confs[i]*100:.2f}%')
        return results

    def run_detection(self, mode, image_path, output):
        if mode == 'realtime':
            self.run()
        elif mode == 'image':
            if image_path is not None:
                results = self.predict_image(image_path)
                print(results, file=output)
            else:
                print("Image path is not provided.", file=output)
        else:
            print(f'Unknown mode: {mode}', file=output)

# 使用示例
if __name__ == '__main__':
    # 从文件中读取参数
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    # 使用从parameters字典中获取的值
    model1_path = parameters['model1_path']
    idx_to_labels_path = parameters['idx_to_labels_path']
    image_path = parameters['image_path']

    # 创建VideoProcessor实例
    classifiers = VideoProcessor(model1_path=model1_path, idx_to_labels_path=idx_to_labels_path, image_path=image_path)

    output = io.StringIO()
    classifiers.run_detection('realtime', image_path=None, output=output)  # 运行实时检测
    classifiers.run_detection('image', image_path=image_path, output=output)  # 运行图像检测