import cgi
import hashlib
import html
import importlib.util
import io
import json
import os
import threading
import time
import traceback
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import test1
import train1


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
PARAMETERS_PATH = BASE_DIR / "parameters.json"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PHONE_UI_PORT", "8000"))


def load_local_statistics_module():
    module_path = BASE_DIR / "statistics.py"
    spec = importlib.util.spec_from_file_location("phone_dataset_statistics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset_statistics = load_local_statistics_module()


BRAND_PATHS = {
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


STATE = {
    "message": "",
    "user": None,
    "last_output": "",
    "last_image_path": "",
    "training": {"running": False, "output": "", "started_at": None},
}


def read_json(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hash_password(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_text(value):
    return html.escape(str(value or ""))


def safe_upload_name(filename):
    suffix = Path(filename or "").suffix.lower() or ".jpg"
    stem = Path(filename or "image").stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    return f"{stem}-{time.time_ns()}{suffix}"


def app_layout(title, body, active="home"):
    nav_items = [
        ("home", "/", "控制台"),
        ("detect", "/detect", "型号识别"),
        ("train", "/train", "模型训练"),
        ("dataset", "/dataset", "数据集处理"),
        ("account", "/account", "账号"),
    ]
    nav = "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in nav_items
    )
    message = f'<div class="notice">{safe_text(STATE["message"])}</div>' if STATE["message"] else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_text(title)} - 手机型号识别系统</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --primary: #176b87;
      --primary-strong: #0d5269;
      --accent: #d97706;
      --good: #0f766e;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: 244px 1fr;
    }}
    aside {{
      background: #10232d;
      color: white;
      padding: 28px 20px;
    }}
    .brand {{
      font-size: 22px;
      font-weight: 700;
      line-height: 1.25;
      margin-bottom: 28px;
    }}
    .status {{
      color: #b8c7d0;
      font-size: 13px;
      margin-bottom: 22px;
    }}
    nav a {{
      display: block;
      color: #d9e3e8;
      text-decoration: none;
      padding: 11px 12px;
      border-radius: 8px;
      margin-bottom: 6px;
      font-size: 15px;
    }}
    nav a.active, nav a:hover {{ background: #1f4758; color: white; }}
    main {{ padding: 30px; max-width: 1180px; width: 100%; }}
    h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: 0; }}
    .subhead {{ color: var(--muted); margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(16, 35, 45, 0.06);
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    label {{ display: block; font-weight: 600; margin: 14px 0 6px; }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: white;
    }}
    textarea {{ min-height: 180px; resize: vertical; }}
    button, .button {{
      display: inline-block;
      border: 0;
      border-radius: 8px;
      background: var(--primary);
      color: white;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      margin-top: 14px;
    }}
    button:hover, .button:hover {{ background: var(--primary-strong); }}
    .button.secondary {{ background: #475467; }}
    .button.warn {{ background: var(--accent); }}
    .notice {{
      border-left: 4px solid var(--primary);
      background: #e8f3f7;
      padding: 12px 14px;
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #e5e7eb;
      padding: 16px;
      border-radius: 8px;
      overflow: auto;
      min-height: 180px;
    }}
    .metric {{ font-size: 26px; font-weight: 800; margin-top: 8px; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 860px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ padding: 20px; }}
      main {{ padding: 20px; }}
      .span-4, .span-6, .span-8 {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">手机型号<br>识别系统</div>
      <div class="status">当前用户：{safe_text(STATE["user"] or "未登录")}</div>
      <nav>{nav}</nav>
    </aside>
    <main>{message}{body}</main>
  </div>
</body>
</html>"""


def dashboard_page():
    params = read_json(PARAMETERS_PATH, {})
    body = f"""
    <h1>控制台</h1>
    <p class="subhead">本地模型训练、图片识别和数据集整理入口。</p>
    <div class="grid">
      <section class="panel span-4"><div class="muted">模型数量</div><div class="metric">{len(list((BASE_DIR / "checkpoint").glob("*.pth"))) if (BASE_DIR / "checkpoint").exists() else 0}</div></section>
      <section class="panel span-4"><div class="muted">预设厂商</div><div class="metric">{len(BRAND_PATHS)}</div></section>
      <section class="panel span-4"><div class="muted">训练状态</div><div class="metric">{"运行中" if STATE["training"]["running"] else "空闲"}</div></section>
      <section class="panel span-12">
        <h2>最近参数</h2>
        <pre>{safe_text(json.dumps(params, ensure_ascii=False, indent=2))}</pre>
      </section>
    </div>"""
    return app_layout("控制台", body, "home")


def account_page():
    body = """
    <h1>账号</h1>
    <p class="subhead">账号信息保存在本机 credentials.json，仅用于本地 Web 页面访问。</p>
    <div class="grid">
      <form class="panel span-4" method="post" action="/login">
        <h2>登录</h2>
        <label>账号</label><input name="username" required>
        <label>密码</label><input name="password" type="password" required>
        <button>登录</button>
      </form>
      <form class="panel span-4" method="post" action="/register">
        <h2>注册</h2>
        <label>账号</label><input name="username" required>
        <label>密码</label><input name="password" type="password" required>
        <label>手机号</label><input name="phone_number" required>
        <button>注册</button>
      </form>
      <form class="panel span-4" method="post" action="/reset-password">
        <h2>重置密码</h2>
        <label>账号</label><input name="username" required>
        <label>手机号</label><input name="phone_number" required>
        <label>新密码</label><input name="password" type="password" required>
        <button>重置</button>
      </form>
    </div>"""
    return app_layout("账号", body, "account")


def detect_page():
    options = "".join(f'<option value="{safe_text(name)}">{safe_text(name)}</option>' for name in BRAND_PATHS)
    params = read_json(PARAMETERS_PATH, {})
    current_image_path = params.get("image_path") or STATE.get("last_image_path") or ""
    output = safe_text(STATE["last_output"])
    body = f"""
    <h1>型号识别</h1>
    <p class="subhead">选择预设厂商，或手动填写模型、标签和图片路径。</p>
    <div class="grid">
      <form class="panel span-6" method="post" action="/detect" enctype="multipart/form-data">
        <label>厂商预设</label><select name="brand">{options}</select>
        <label>模型路径</label><input name="model1_path" value="{safe_text(params.get("model1_path", ""))}">
        <label>标签路径</label><input name="idx_to_labels_path" value="{safe_text(params.get("idx_to_labels_path", ""))}">
        <label>图片路径</label><input id="image_path" name="image_path" value="{safe_text(current_image_path)}">
        <label>或上传图片</label><input type="file" name="image_file" accept="image/*">
        <button>开始识别</button>
      </form>
      <section class="panel span-6">
        <h2>识别结果</h2>
        <p class="muted">当前图片路径：{safe_text(current_image_path or "未选择")}</p>
        <pre>{output}</pre>
      </section>
    </div>"""
    return app_layout("型号识别", body, "detect")


def train_page():
    params = read_json(PARAMETERS_PATH, {})
    output = safe_text(STATE["training"]["output"])
    disabled = "disabled" if STATE["training"]["running"] else ""
    body = f"""
    <h1>模型训练</h1>
    <p class="subhead">数据集目录需要包含 train 和 val 子目录。</p>
    <div class="grid">
      <form class="panel span-4" method="post" action="/train">
        <label>数据集目录</label><input name="dataset_dir" value="{safe_text(params.get("dataset_dir", ""))}" required>
        <label>预训练模型路径</label><input name="model_path" value="{safe_text(params.get("model_path", "./resnet50.pth"))}" required>
        <label>Batch Size</label><input name="batch_size" type="number" value="{safe_text(params.get("batch_size", 16))}" min="1">
        <label>Epochs</label><input name="epochs" type="number" value="{safe_text(params.get("epochs", 10))}" min="1">
        <label>Step Size</label><input name="step_size" type="number" value="{safe_text(params.get("step_size", 5))}" min="1">
        <label>Gamma</label><input name="gamma" type="number" value="{safe_text(params.get("gamma", 0.1))}" step="0.01">
        <label>加载方式</label>
        <select name="load_method">
          <option value="init_and_load_model">微调 CBAM 模型</option>
          <option value="load_model">微调普通 ResNet</option>
          <option value="download_and_load_model">微调全部层</option>
        </select>
        <button {disabled}>开始训练</button>
      </form>
      <section class="panel span-8">
        <h2>训练日志</h2>
        <pre>{output}</pre>
        <a class="button secondary" href="/train">刷新日志</a>
      </section>
    </div>"""
    return app_layout("模型训练", body, "train")


def dataset_page():
    output = safe_text(STATE["last_output"])
    body = f"""
    <h1>数据集处理</h1>
    <p class="subhead">划分训练/验证集，或对图片去重并统一为 RGB/JPEG。</p>
    <div class="grid">
      <form class="panel span-6" method="post" action="/dataset/split">
        <h2>划分数据集</h2>
        <label>数据集目录</label><input name="dataset_path" required>
        <label>验证集比例</label><input name="test_frac" type="number" step="0.01" value="0.2" min="0.01" max="0.9">
        <button>开始划分</button>
      </form>
      <form class="panel span-6" method="post" action="/dataset/convert">
        <h2>图片去重与格式转换</h2>
        <label>图片目录</label><input name="dataset_path" required>
        <button class="warn">开始处理</button>
      </form>
      <section class="panel span-12">
        <h2>处理输出</h2>
        <pre>{output}</pre>
      </section>
    </div>"""
    return app_layout("数据集处理", body, "dataset")


class PhoneUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path).path
        pages = {
            "/": dashboard_page,
            "/account": account_page,
            "/detect": detect_page,
            "/train": train_page,
            "/dataset": dataset_page,
        }
        if route in pages:
            self.respond_html(pages[route]())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/login":
                self.handle_login()
            elif route == "/register":
                self.handle_register()
            elif route == "/reset-password":
                self.handle_reset_password()
            elif route == "/detect":
                self.handle_detect()
            elif route == "/train":
                self.handle_train()
            elif route == "/dataset/split":
                self.handle_dataset_split()
            elif route == "/dataset/convert":
                self.handle_dataset_convert()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception:
            STATE["message"] = "操作失败"
            STATE["last_output"] = traceback.format_exc()
            self.redirect("/")

    def parse_form(self):
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            data = {key: form.getvalue(key) for key in form.keys() if not getattr(form[key], "filename", None)}
            files = {key: form[key] for key in form.keys() if getattr(form[key], "filename", None)}
            return data, files

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return {key: values[0] for key, values in parse_qs(body).items()}, {}

    def handle_login(self):
        data, _ = self.parse_form()
        credentials = read_json(CREDENTIALS_PATH, {})
        username = data.get("username", "").strip()
        password_hash = hash_password(data.get("password", ""))
        if username in credentials and credentials[username]["password"] == password_hash:
            STATE["user"] = username
            STATE["message"] = f"登录成功，欢迎 {username}"
        else:
            STATE["message"] = "账号或密码错误"
        self.redirect("/account")

    def handle_register(self):
        data, _ = self.parse_form()
        username = data.get("username", "").strip()
        password = data.get("password", "")
        phone_number = data.get("phone_number", "").strip()
        credentials = read_json(CREDENTIALS_PATH, {})

        if not username.isalnum():
            STATE["message"] = "账号只能包含字母和数字"
        elif not phone_number.isdigit():
            STATE["message"] = "手机号只能包含数字"
        elif username in credentials:
            STATE["message"] = "该账号已注册"
        else:
            credentials[username] = {
                "password": hash_password(password),
                "phone_number": hash_password(phone_number),
            }
            write_json(CREDENTIALS_PATH, credentials)
            STATE["message"] = f"账号 {username} 注册成功"
        self.redirect("/account")

    def handle_reset_password(self):
        data, _ = self.parse_form()
        username = data.get("username", "").strip()
        phone_hash = hash_password(data.get("phone_number", "").strip())
        credentials = read_json(CREDENTIALS_PATH, {})
        if username in credentials and credentials[username]["phone_number"] == phone_hash:
            credentials[username]["password"] = hash_password(data.get("password", ""))
            write_json(CREDENTIALS_PATH, credentials)
            STATE["message"] = "密码已重置"
        else:
            STATE["message"] = "账号或手机号错误"
        self.redirect("/account")

    def handle_detect(self):
        data, files = self.parse_form()
        brand = data.get("brand") or "通用"
        idx_path, model_path = BRAND_PATHS.get(brand, ("", ""))
        model_path = data.get("model1_path") or model_path
        idx_path = data.get("idx_to_labels_path") or idx_path
        image_path = data.get("image_path") or ""

        uploaded = files.get("image_file")
        if uploaded is not None and uploaded.filename:
            UPLOAD_DIR.mkdir(exist_ok=True)
            image_path = str(UPLOAD_DIR / safe_upload_name(uploaded.filename))
            with open(image_path, "wb") as f:
                f.write(uploaded.file.read())

        params = {
            "idx_to_labels_path": idx_path,
            "model1_path": model_path,
            "image_path": image_path,
        }
        write_json(PARAMETERS_PATH, params)
        buffer = io.StringIO()
        processor = test1.VideoProcessor(model1_path=model_path, idx_to_labels_path=idx_path, image_path=image_path)
        processor.run_detection("image", image_path, buffer)
        STATE["last_output"] = buffer.getvalue()
        STATE["last_image_path"] = image_path
        STATE["message"] = f"识别完成，当前图片路径已更新为: {image_path}"
        self.redirect("/detect")

    def handle_train(self):
        if STATE["training"]["running"]:
            STATE["message"] = "训练任务正在运行"
            self.redirect("/train")
            return

        data, _ = self.parse_form()
        params = {
            "dataset_dir": data["dataset_dir"],
            "model_path": data["model_path"],
            "batch_size": int(data.get("batch_size") or 16),
            "epochs": int(data.get("epochs") or 10),
            "step_size": int(float(data.get("step_size") or 5)),
            "gamma": float(data.get("gamma") or 0.1),
            "load_method": data.get("load_method") or "init_and_load_model",
            "save": data["dataset_dir"],
        }
        write_json(PARAMETERS_PATH, params)
        STATE["training"] = {"running": True, "output": "训练已启动...\n", "started_at": time.time()}

        def run_training():
            try:
                def emit(line):
                    STATE["training"]["output"] += str(line)

                classifier = train1.PhoneClassifier(**params)
                classifier.train(emit)
                STATE["message"] = "训练完成"
            except Exception:
                STATE["training"]["output"] += "\n" + traceback.format_exc()
                STATE["message"] = "训练失败"
            finally:
                STATE["training"]["running"] = False

        threading.Thread(target=run_training, daemon=True).start()
        self.redirect("/train")

    def handle_dataset_split(self):
        data, _ = self.parse_form()
        dataset_path = data["dataset_path"]
        test_frac = float(data.get("test_frac") or 0.2)
        buffer = io.StringIO()
        splitter = dataset_statistics.DatasetSplitter(dataset_path=dataset_path, test_frac=test_frac)
        splitter.create_folders()
        splitter.split_dataset(buffer.write)
        splitter.save_statistics()
        STATE["last_output"] = buffer.getvalue()
        STATE["message"] = "数据集划分完成"
        self.redirect("/dataset")

    def handle_dataset_convert(self):
        data, _ = self.parse_form()
        dataset_path = data["dataset_path"]
        buffer = io.StringIO()
        splitter = dataset_statistics.DatasetSplitter(dataset_path=dataset_path, test_frac=0.2)
        splitter.remove_duplicates_and_convert_images(callback=buffer.write, directory=dataset_path)
        STATE["last_output"] = buffer.getvalue()
        STATE["message"] = "图片处理完成"
        self.redirect("/dataset")

    def respond_html(self, content):
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, location):
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, format, *args):
        return


def run_server(host=HOST, port=PORT):
    server = ThreadingHTTPServer((host, port), PhoneUIHandler)
    print(f"手机型号识别 Web 页面已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
