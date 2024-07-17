import io
import os
import statistics
import train1
import test1
import json
import hashlib
import time
from tkinter import *
from tkinter import Label, filedialog, Tk, ttk, Button, Canvas, BOTH, NW, YES, messagebox
from PIL import Image, ImageTk

class UI:
    def __init__(self, master):
        self.style = ttk.Style()
        self.video_processor = None
        self.master = master
        self.canvas = Canvas(master)
        self.canvas.pack(fill=BOTH, expand=YES)
        self.image = Image.open("./image/background.jpg")
        self.img_copy = self.image.copy()
        self.background = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.background, anchor=NW)
        self.canvas.bind("<Configure>", self.resize_image)

        # 创建一个打开新窗口的按钮
        self.new_window_button = Button(self.canvas, text="打开新窗口", command=self.create_new_window)

        # 创建一个账号输入框
        self.username_entry = Entry(self.canvas)
        self.username_entry.pack()

        # 创建一个密码输入框
        self.password_entry = Entry(self.canvas, show="*")
        self.password_entry.pack()

        # 创建一个手机号输入框
        self.phone_number_entry = Entry(self.canvas)
        self.phone_number_entry.pack()

        # 创建一个登录按钮
        self.login_button = Button(self.canvas, text="登录", command=self.login)
        self.login_button.pack()

        # 创建一个注册按钮
        self.register_button = Button(self.canvas, text="注册", command=self.register)
        self.register_button.pack()

        # 创造一个重置按钮
        self.reset_button = Button(self.canvas, text="重置", command=self.reset_password)
        self.reset_button.pack()

        # 创建标签
        self.username_label = Label(self.canvas, text="账号")
        self.password_label = Label(self.canvas, text="密码")
        self.phone_number_label = Label(self.canvas, text="手机号")

        # 创建一个字典来存储账号和密码
        if os.path.exists('credentials.json'):
            with open('credentials.json', 'r') as f:
                self.credentials = json.load(f)
        else:
            self.credentials = {}

    def hash_password(self, password):
        # 创建一个新的哈希对象
        hash_object = hashlib.sha256()
        # 哈希密码
        hash_object.update(password.encode('utf-8'))
        # 返回哈希的十六进制表示
        return hash_object.hexdigest()

    def login(self):
        username = self.username_entry.get()
        password = self.hash_password(self.password_entry.get())
        if username in self.credentials and self.credentials[username]['password'] == password:
            messagebox.showinfo("登录成功", "欢迎 {}!".format(username))
            self.create_new_window()
        else:
            messagebox.showerror("登录失败", "账号或密码错误")

    def register(self):
        username = self.username_entry.get()
        password = self.hash_password(self.password_entry.get())
        phone_number = self.phone_number_entry.get()

        if not phone_number.isdigit():
            messagebox.showerror("注册失败", "手机号码只能包含数字")
            return

        if not (username.isalnum() and password.isalnum()):
            messagebox.showerror("注册失败", "账号和密码只能包含字母和数字")
            return

        if username in self.credentials:
            messagebox.showerror("注册失败", "该用户名已被注册")
            return

        hashed_password = self.hash_password(password)
        hashed_phone_number = self.hash_password(phone_number)

        self.credentials[username] = {'password': hashed_password, 'phone_number': hashed_phone_number}
        with open('credentials.json', 'w') as f:
            json.dump(self.credentials, f)

        messagebox.showinfo("注册成功", "账号 {} 已成功注册!".format(username))

    def reset_password(self):
        username = self.username_entry.get()
        phone_number = self.hash_password(self.phone_number_entry.get())  # 对输入的手机号进行哈希加密
        if username in self.credentials and self.credentials[username]['phone_number'] == phone_number:
            new_password = self.password_entry.get()
            self.credentials[username]['password'] = self.hash_password(new_password)
            with open('credentials.json', 'w') as f:
                json.dump(self.credentials, f)
            messagebox.showinfo("密码重置成功", "账号 {} 的密码已成功重置!".format(username))
        else:
            messagebox.showerror("密码重置失败", "账号或手机号码错误")

    def resize_image(self, event):
        new_width = event.width
        new_height = event.height
        self.image = self.img_copy.resize((new_width, new_height), Image.LANCZOS)
        self.background = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.background, anchor=NW)
        # self.new_window_button.place(x=new_width * 0.1, y=new_height * 0.2)  # 更新打开新窗口按钮的位置
        self.login_button.place(x=new_width * 0.35, y=new_height * 0.6)
        self.register_button.place(x=new_width * 0.5, y=new_height * 0.6)
        self.username_entry.place(x=new_width * 0.4, y=new_height * 0.3)
        self.password_entry.place(x=new_width * 0.4, y=new_height * 0.4)
        self.phone_number_entry.place(x=new_width * 0.4, y=new_height * 0.5)
        self.reset_button.place(x=new_width * 0.43, y=new_height * 0.6)

        self.username_label.place(x=new_width * 0.35, y=new_height * 0.3)
        self.password_label.place(x=new_width * 0.35, y=new_height * 0.4)
        self.phone_number_label.place(x=new_width * 0.35, y=new_height * 0.5)

    def create_new_window(self):
        self.master.withdraw()
        self.new_window = Toplevel(self.master)
        self.new_window.title("主界面")
        self.new_window.geometry(f'{self.master.winfo_width()}x{self.master.winfo_height()}')  # 设置新窗口的大小与原窗口一致

        # 设置新窗口的背景
        self.new_image = Image.open("./image/background0.jpg")
        self.new_img_copy = self.new_image.copy()
        self.new_background = ImageTk.PhotoImage(self.new_image)
        self.new_canvas = Canvas(self.new_window)
        self.new_canvas.create_image(0, 0, image=self.new_background, anchor=NW)
        self.new_canvas.bind("<Configure>", self.resize_new_image)  # 绑定新窗口的背景图像的缩放函数
        self.new_canvas.pack(fill=BOTH, expand=YES)

        # 在新窗口关闭时重新显示主窗口，可以通过为新窗口绑定一个关闭事件来实现
        self.new_window.protocol("WM_DELETE_WINDOW", self.on_close_new_window)

        # 创建一个训练模型的按钮
        self.train_model_button = Button(self.new_window, text="训练手机型号模型", command=self.train_model)
        self.train_model_button_window = self.new_canvas.create_window(400, 150, anchor="nw", window=self.train_model_button)

        # 创建一个测试模型的按钮
        self.test_model_button = Button(self.new_window, text="手机型号识别", command=self.test_model)
        self.test_model_button_window = self.new_canvas.create_window(400, 200, anchor="nw", window=self.test_model_button)

        self.statistics_model_button = Button(self.new_window, text="分类手机型号数据集", command=self.statistics_model)
        self.statistics_model_button_window = self.new_canvas.create_window(400, 250, anchor="nw", window=self.statistics_model_button)

    def resize_new_image(self, event):
        new_width = event.width
        new_height = event.height
        self.new_image = self.new_img_copy.resize((new_width, new_height), Image.LANCZOS)
        self.new_background = ImageTk.PhotoImage(self.new_image)
        self.new_canvas.create_image(0, 0, image=self.new_background, anchor=NW)

    def on_close_new_window(self):
        # 关闭新窗口
        self.new_window.destroy()
        # 重新显示（原）根窗口
        self.master.deiconify()

    def train_model(self):
        # 首先检查new_window是否存在并隐藏
        if hasattr(self, 'create_new_window'):
            self.new_window.withdraw()

        self.train_window = Toplevel(self.master)
        self.train_window.title("训练模型")
        self.train_window.geometry(f'{self.master.winfo_width()}x{self.master.winfo_height()}')  # 设置新窗口的大小与原窗口一致

        # 设置新窗口的背景
        self.train_image = Image.open("./image/background0.jpg")
        self.train_img_copy = self.train_image.copy()
        self.train_background = ImageTk.PhotoImage(self.train_image)
        self.train_canvas = Canvas(self.train_window)
        self.train_canvas.create_image(0, 0, image=self.train_background, anchor=NW)
        self.train_canvas.bind("<Configure>", self.resize_train_image)  # 绑定新窗口的背景图像的缩放函数
        self.train_canvas.pack(fill=BOTH, expand=YES)

        self.train_window.protocol("WM_DELETE_WINDOW", self.on_close_train_window)

        # 创建输入框
        self.batch_size_entry = Entry(self.train_window)
        self.epochs_entry = Entry(self.train_window)
        self.step_size_entry = Entry(self.train_window)
        self.gamma_entry = Entry(self.train_window)

        # 创建文件选择框
        self.dataset_dir_button = Button(self.train_window, text="添加数据集", command=self.open_file)

        # 创建模型路径按钮
        self.model_path_button1 = Button(self.train_window, text="resnet34",
                                         command=lambda: self.select_model_path("resnet34.pth"))
        self.model_path_button2 = Button(self.train_window, text="resnet50",
                                         command=lambda: self.select_model_path("resnet50.pth"))

        # 创建加载方法按钮
        self.load_method_button1 = Button(self.train_window, text="微调全连接层",
                                          command=lambda: self.select_load_method("load_model"))
        self.load_method_button2 = Button(self.train_window, text="微调所有层",
                                          command=lambda: self.select_load_method("download_and_load_model"))

        # 创建输出文本框
        self.result_text = Text(self.train_window)

        # 创建训练按钮
        self.train_button = Button(self.train_window, text="训练", command=self.start_training)
        self.train_button_window = self.train_canvas.create_window(500, 600, anchor="nw", window=self.train_button)

        # 创建标签
        self.batch_size_label = Label(self.train_window, text="Batch Size")
        self.epochs_label = Label(self.train_window, text="epochs")
        self.step_size_label = Label(self.train_window, text="step size")
        self.gamma_label = Label(self.train_window, text="gamma")

        # 将标签添加到画布上
        self.train_canvas.create_window(80, 100, window=self.batch_size_label)
        self.train_canvas.create_window(80, 150, window=self.epochs_label)
        self.train_canvas.create_window(80, 200, window=self.step_size_label)
        self.train_canvas.create_window(80, 250, window=self.gamma_label)

        # 将组件添加到画布上
        self.train_canvas.create_window(200, 100, window=self.batch_size_entry)
        self.train_canvas.create_window(200, 150, window=self.epochs_entry)
        self.train_canvas.create_window(200, 200, window=self.step_size_entry)
        self.train_canvas.create_window(200, 250, window=self.gamma_entry)
        self.train_canvas.create_window(200, 300, window=self.dataset_dir_button)
        self.train_canvas.create_window(200, 350, window=self.model_path_button1)
        self.train_canvas.create_window(200, 400, window=self.model_path_button2)
        self.train_canvas.create_window(200, 450, window=self.load_method_button1)
        self.train_canvas.create_window(200, 500, window=self.load_method_button2)
        self.train_canvas.create_window(600, 200, window=self.result_text)
        self.train_canvas.create_window(600, 450, window=self.train_button)

    def resize_train_image(self, event):
        new_width = event.width
        new_height = event.height
        self.train_image = self.train_img_copy.resize((new_width, new_height), Image.LANCZOS)
        self.train_background = ImageTk.PhotoImage(self.train_image)
        self.train_canvas.create_image(0, 0, image=self.train_background, anchor=NW)

    def on_close_train_window(self):
        # 销毁train_window
        if hasattr(self, 'train_window'):
            self.train_window.destroy()
            delattr(self, 'train_window')  # 删除属性，防止后续错误

        # 检查new_window是否存在并重新显示
        if hasattr(self, 'new_window'):
            self.new_window.deiconify()

    def open_file(self):
        self.filename = filedialog.askdirectory()  # 获取文件夹路径并保存到实例变量中
        self.result_text.insert(END, f"选中的数据集路径是：{self.filename}\n")  # 将文件路径插入到文本框中

    def select_model_path(self, model_path):
        self.model_path = model_path
        self.result_text.insert(END, f"选中的模型路径是：{model_path}\n")

    def select_load_method(self, load_method):
        self.load_method = load_method

        # 根据load_method的值更改输出消息
        if load_method == "load_model":
            message = "选中的操作是：微调全连接层"
        elif load_method == "download_and_load_model":
            message = "选中的操作是：微调所有层"
        else:
            message = f"选中的操作是：{load_method}"  # 对于其他情况，直接显示load_method的值

        # 在结果文本框中插入相应的消息
        self.result_text.insert(END, f"{message}\n")

    def start_training(self):
        # 获取输入框和列表框的值
        batch_size = int(self.batch_size_entry.get())
        epochs = int(self.epochs_entry.get())
        step_size = float(self.step_size_entry.get())
        gamma = float(self.gamma_entry.get())
        dataset_dir = self.filename  # 使用在 open_file 方法中保存的文件路径
        model_path = self.model_path  # 使用保存的模型路径
        load_method = self.load_method

        # 显示输入的参数
        self.result_text.insert(END, f"batch_size={batch_size}\n")
        self.result_text.insert(END, f"epochs={epochs}\n")
        self.result_text.insert(END, f"step_size={step_size}\n")
        self.result_text.insert(END, f"gamma={gamma}\n")
        self.result_text.insert(END, f"dataset_dir={dataset_dir}\n")

        # 将参数保存到一个文件中
        parameters = {
            'batch_size': batch_size,
            'epochs': epochs,
            'step_size': step_size,
            'gamma': gamma,
            'dataset_dir': dataset_dir,
            'model_path': model_path,
            'load_method': load_method,
            'save': dataset_dir
        }
        with open('parameters.json', 'w') as f:
            json.dump(parameters, f)

        # 创建更新UI的回调函数
        def update_output(line):
            self.result_text.insert(END, line)
            self.result_text.see(END)  # 滚动到文本框底部
            self.master.update()  # 更新UI

        # 初始化并开始训练模型
        classifier = train1.PhoneClassifier(**parameters)
        classifier.train(update_output)  # 使用回调函数

        # 输出训练成功信息
        self.result_text.insert(END, "训练模型成功！\n")

    def resize_test_image(self, event):
        new_width = event.width
        new_height = event.height
        self.test_image = self.test_img_copy.resize((new_width, new_height), Image.LANCZOS)
        self.test_background = ImageTk.PhotoImage(self.test_image)
        self.test_canvas.create_image(0, 0, image=self.test_background, anchor=NW)
        self.test_canvas.config(width=new_width, height=new_height)

    def test_model(self):
        if hasattr(self, 'new_window'):
            self.new_window.withdraw()

        self.test_window = Toplevel(self.master)
        self.test_window.title("手机型号识别")
        self.test_window.geometry(f'{self.master.winfo_width()}x{self.master.winfo_height()}')  # 设置新窗口的大小与原窗口一致

        # 设置新窗口的背景
        self.test_image = Image.open("./image/background0.jpg")
        self.test_img_copy = self.test_image.copy()
        self.test_background = ImageTk.PhotoImage(self.test_image)
        self.test_canvas = Canvas(self.test_window)
        self.test_canvas.create_image(0, 0, image=self.test_background, anchor=NW)
        self.test_canvas.bind("<Configure>", self.resize_test_image)  # 绑定新窗口的背景图像的缩放函数
        self.test_canvas.pack(fill=BOTH, expand=YES)

        self.test_window.protocol("WM_DELETE_WINDOW", self.on_close_test_window)

        # 创建一个Label用于显示摄像头的输出
        self.camera_output = Label(self.test_window)
        self.camera_output.pack()

        # 创建一个可以折叠的选择框
        self.phone_brands_combobox = ttk.Combobox(self.test_window)
        # 添加手机品牌到选择框
        phone_brands = ["通用", "Apple", "iqoo", "oppo", "realme", "Samsung", "vivo", "红米", "华为", "魅族", "努比亚", "荣耀",
                        "小米", "一加"]
        self.phone_brands_combobox['values'] = phone_brands
        self.phone_brands_combobox.pack()
        self.phone_brands_combobox.bind("<<ComboboxSelected>>", self.update_paths)

        self.brand_paths = {
            "通用": ("./phone data/phone dataset/idx_to_labels.npy", "./checkpoint/Resnet50-CBAM-all.pth"),
            "Apple": ("./phone data/phone name/Apple/idx_to_labels.npy", "./checkpoint/Apple.pth"),
            "iqoo": ("./phone data/phone name/iqoo/idx_to_labels.npy", "./checkpoint/iqoo.pth"),
            "oppo": ("./phone data/phone name/oppo/idx_to_labels.npy", "./checkpoint/oppo.pth"),
            "realme": ("./phone data/phone name/realme/idx_to_labels.npy", "./checkpoint/realme.pth"),
            "Samsung": ("./phone data/phone name/Samsung/idx_to_labels.npy", "./checkpoint/Samsung.pth"),
            "vivo": ("./phone data/phone name/vivo/idx_to_labels.npy", "./checkpoint/vivo.pth"),
            "红米": ("./phone data/phone name/红米/idx_to_labels.npy", "./checkpoint/红米.pth"),
            "华为": ("./phone data/phone name/华为/idx_to_labels.npy", "./checkpoint/华为.pth"),
            "魅族": ("./phone data/phone name/魅族/idx_to_labels.npy", "./checkpoint/魅族.pth"),
            "努比亚": ("./phone data/phone name/努比亚/idx_to_labels.npy", "./checkpoint/努比亚.pth"),
            "荣耀": ("./phone data/phone name/荣耀/idx_to_labels.npy", "./checkpoint/荣耀.pth"),
            "小米": ("./phone data/phone name/小米/idx_to_labels.npy", "./checkpoint/小米.pth"),
            "一加": ("./phone data/phone name/一加/idx_to_labels.npy", "./checkpoint/一加.pth"),
        }

        # 创建三个文件选择按钮
        self.model1_path_button = Button(self.test_window, text="选择模型路径", command=self.select_model1_path)
        self.model1_path_button.pack()

        self.idx_to_labels_path_button = Button(self.test_window, text="选择标签路径", command=self.select_idx_to_labels_path)
        self.idx_to_labels_path_button.pack()

        self.image_path_button = Button(self.test_window, text="选择图像路径", command=self.select_image_path)
        self.image_path_button.pack()

        # 创建一个名为"测试"的按钮，按下后执行start_testing函数
        self.test_button = Button(self.test_window, text="摄像头检测", command=self.start_testing_realtime)
        self.test_button.pack()

        # 创建一个名为"检测"的按钮，按下后执行start_testing函数
        self.detect_button = Button(self.test_window, text="图片检测", command=self.start_testing_image)
        self.detect_button.pack()

        # 创建一个文本输出框
        self.output_text = Text(self.test_window)

        self.select_label = Label(self.test_window, text="选择厂商")

        self.test_canvas.create_window(500, 400, window=self.model1_path_button)  # 将文件选择按钮添加到画布上
        self.test_canvas.create_window(500, 450, window=self.idx_to_labels_path_button)  # 将文件选择按钮添加到画布上
        self.test_canvas.create_window(500, 500, window=self.image_path_button)  # 将文件选择按钮添加到画布上
        self.test_canvas.create_window(700, 400, window=self.test_button)  # 将测试按钮添加到画布上
        self.test_canvas.create_window(700, 500, window=self.detect_button)
        self.test_canvas.create_window(500, 200, window=self.output_text)
        self.test_canvas.create_window(240, 450, window=self.select_label)
        self.test_canvas.create_window(350, 450, window=self.phone_brands_combobox)  # 将列表选择框添加到画布上

    def on_close_test_window(self):
        if hasattr(self, 'test_window'):
            self.test_window.destroy()
            delattr(self, 'test_window')  # 删除属性，防止后续错误

            # 检查new_window是否存在并重新显示
        if hasattr(self, 'new_window'):
            self.new_window.deiconify()

    def select_model1_path(self):
        self.model1_path = filedialog.askopenfilename()
        self.output_text.insert(END, '选中的模型路径是：' + self.model1_path + '\n')

    def select_idx_to_labels_path(self):
        self.idx_to_labels_path = filedialog.askopenfilename()
        self.output_text.insert(END, '选中的模型标签路径是：' + self.idx_to_labels_path + '\n')

    def select_image_path(self):
        self.image_path = filedialog.askopenfilename()
        self.output_text.insert(END, '选中的图像路径是：' + self.image_path + '\n')

    def start_testing_realtime(self):
        self.start_testing('realtime', '')

    def start_testing_image(self):
        self.start_testing('image', self.image_path)

    def update_paths(self, event=None):
        selected_brand = self.phone_brands_combobox.get()
        self.idx_to_labels_path, self.model1_path = self.brand_paths.get(selected_brand, ("", ""))

        # 将路径写入parameters.json文件
        with open('parameters.json', 'w') as f:
            json.dump({
                'idx_to_labels_path': self.idx_to_labels_path,
                'model1_path': self.model1_path,
                'image_path': self.image_path
            }, f)

        self.output_text.insert(END, '选中的模型路径是：' + self.model1_path + '\n')
        self.output_text.insert(END, '选中的模型标签路径是：' + self.idx_to_labels_path + '\n')

    def start_testing(self, mode, image_path):
        idx_to_labels_path = self.idx_to_labels_path
        model1_path = self.model1_path
        image_path = self.image_path

        with open('parameters.json', 'w') as f:
            json.dump({
                'idx_to_labels_path': idx_to_labels_path,
                'model1_path': model1_path,
                'image_path': image_path
            }, f)

        output = io.StringIO()

        classifiers = test1.VideoProcessor(model1_path=model1_path, idx_to_labels_path=idx_to_labels_path, image_path=image_path)
        classifiers.run_detection(mode, image_path, output)

        # 将缓冲区的内容添加到输出文本框中
        output_content = output.getvalue()
        if output_content:  # 检查内容是否非空
            self.output_text.insert(END, "识别置信度由大到小前五的手机型号: \n" + output_content)

    def statistics_model(self):
        if hasattr(self, 'new_window'):
            self.new_window.withdraw()
        self.statistics_window = Toplevel(self.master)
        self.statistics_window.title("分类手机型号数据集")
        self.statistics_window.geometry(f'{self.master.winfo_width()}x{self.master.winfo_height()}')  # 设置新窗口的大小与原窗口一致

        # 设置新窗口的背景
        self.statistics_image = Image.open("./image/background0.jpg")
        self.statistics_img_copy = self.statistics_image.copy()
        self.statistics_background = ImageTk.PhotoImage(self.statistics_image)
        self.statistics_canvas = Canvas(self.statistics_window)
        self.statistics_canvas.create_image(0, 0, image=self.statistics_background, anchor=NW)
        self.statistics_canvas.bind("<Configure>", self.resize_statistics_image)  # 绑定新窗口的背景图像的缩放函数
        self.statistics_canvas.pack(fill=BOTH, expand=YES)

        self.statistics_window.protocol("WM_DELETE_WINDOW", self.on_close_statistics_window)

        self.test_frac_entry = Entry(self.statistics_window)

        self.put_text = Text(self.statistics_window)

        self.statistics_button = Button(self.statistics_window, text="分类", command=self.start_statistics)
        self.statistics_button.pack()

        self.statistics_dir_button = Button(self.statistics_window, text="添加数据集路径", command=self.statistics_path)

        self.test_frac_label = Label(self.statistics_window, text="划分测试集比例系数")

        self.convert_colors_button = Button(self.statistics_window, text="图像去重和转化颜色格式", command=self.convert_colors)
        self.convert_colors_button.pack()

        # 将标签添加到画布上
        self.statistics_canvas.create_window(350, 400, window=self.test_frac_label)

        self.statistics_canvas.create_window(500, 500, window=self.statistics_button)
        self.statistics_canvas.create_window(500, 450, window=self.statistics_dir_button)
        self.statistics_canvas.create_window(500, 200, window=self.put_text)
        self.statistics_canvas.create_window(500, 400, window=self.test_frac_entry)
        self.statistics_canvas.create_window(650, 400, window=self.convert_colors_button)

    def on_close_statistics_window(self):
        if hasattr(self, 'statistics_window'):
            self.statistics_window.destroy()
            delattr(self, 'statistics_window')  # 删除属性，防止后续错误

            # 检查new_window是否存在并重新显示
        if hasattr(self, 'new_window'):
            self.new_window.deiconify()

    def resize_statistics_image(self, event):
        new_width = event.width
        new_height = event.height
        self.statistics_image = self.new_img_copy.resize((new_width, new_height), Image.LANCZOS)
        self.statistics_background = ImageTk.PhotoImage(self.new_image)
        self.statistics_canvas.create_image(0, 0, image=self.new_background, anchor=NW)

    def convert_colors(self):
        # 定义一个回调函数，用于将输出直接添加到文本框中
        def update_output(message):
            self.put_text.insert(END, message)  # 将消息添加到文本框
            self.put_text.see(END)  # 滚动到文本框底部
            self.master.update()  # 更新 GUI 界面

        # 初始化 DatasetSplitter 对象
        self.dataset_splitter = statistics.DatasetSplitter(dataset_path=self.filepath, test_frac=None)

        # 使用 update_output 函数作为回调
        self.dataset_splitter.remove_duplicates_and_convert_images(callback=update_output, directory=self.filepath)

        # 输出处理成功的消息
        self.put_text.insert(END, "图像去重和转化颜色格式成功！\n")

    def statistics_path(self):
        self.filepath = filedialog.askdirectory()  # 获取文件夹路径并保存到实例变量中
        if self.filepath:  # 检查用户是否选择了路径
            self.put_text.insert(END, f"选中的数据集路径是：{self.filepath}\n")
        else:
            self.put_text.insert(END, "没有选择路径。\n")

    def start_statistics(self):
        test_frac = float(self.test_frac_entry.get())
        dataset_path = self.filepath

        self.put_text.insert(END, f"数据集中测试集的比值为：{test_frac}\n")
        self.put_text.insert(END, f"数据集路径：{dataset_path}\n")

        parameters = {
            'dataset_path': dataset_path,
            'test_frac': test_frac
        }
        with open('parameters.json', 'w') as f:
            json.dump(parameters, f)

        # 创建更新UI的回调函数
        def update_output(line):
            self.put_text.insert(END, line)
            self.put_text.see(END)  # 滚动到文本框底部
            self.master.update()  # 更新UI

        classifiers = statistics.DatasetSplitter(**parameters)
        classifiers.split_dataset(update_output)

        self.put_text.insert(END, "数据集分类成功！\n")

if __name__ == '__main__':
    root = Tk()
    root.geometry('1000x600')  # 设置窗口的初始大小为800x600
    app = UI(root)
    root.title("手机型号识别程序")
    root.mainloop()