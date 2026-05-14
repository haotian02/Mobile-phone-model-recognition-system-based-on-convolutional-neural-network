import hashlib
import html
import io
import json
import os
import re
import threading
import time
import traceback
from email.parser import BytesParser
from email.policy import HTTP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import statistics as dataset_statistics
import test1
import train1


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
PARAMETERS_PATH = BASE_DIR / "parameters.json"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PHONE_UI_PORT", "8000"))
MAX_LOG_CHARS = 30000

# ── thread-safe state ───────────────────────────────────────────────
_state_lock = threading.Lock()

STATE = {
    "message": "",
    "user": None,
    "last_output": "",
    "last_image_path": "",
    "training": {"running": False, "output": "", "started_at": None},
}


def _set_state(**kwargs):
    with _state_lock:
        for k, v in kwargs.items():
            STATE[k] = v


def _get_state(key, default=""):
    with _state_lock:
        return STATE.get(key, default)


def _get_training():
    with _state_lock:
        return dict(STATE["training"])


def _set_training(**kwargs):
    with _state_lock:
        STATE["training"].update(kwargs)


# ── brand presets ───────────────────────────────────────────────────
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


# ── utilities ───────────────────────────────────────────────────────
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


class _UploadFile:
    """Mimics cgi.FieldStorage interface for uploaded files."""

    def __init__(self, filename, data):
        self.filename = filename
        self._file = io.BytesIO(data)

    @property
    def file(self):
        self._file.seek(0)
        return self._file


# ═════════════════════════════════════════════════════════════════════
#  iOS-style HTML/CSS templates
# ═════════════════════════════════════════════════════════════════════

def _page_html(title, body_html, show_nav=True, active_tab=""):
    """Render a full HTML page with optional iOS-style top navigation."""
    msg = _get_state("message")
    user = _get_state("user")

    notice = ""
    if msg:
        notice = f'<div class="toast" id="toast">{safe_text(msg)}</div><script>setTimeout(()=>document.getElementById("toast")?.remove(),3000)</script>'
        _set_state(message="")

    tabs_html = ""
    if show_nav and user:
        tabs = [
            ("home", "/", "控制台"),
            ("detect", "/detect", "型号识别"),
            ("train", "/train", "模型训练"),
            ("dataset", "/dataset", "数据集"),
            ("account", "/account", "账号"),
        ]
        tabs_html = '<nav class="tabs">' + "".join(
            f'<a class="tab{" active" if k == active_tab else ""}" href="{href}">{label}</a>'
            for k, href, label in tabs
        ) + "</nav>"

    user_html = ""
    if show_nav and user:
        user_html = f'<div class="nav-right">{safe_text(user)}<a href="/logout" class="logout-link">退出</a></div>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{safe_text(title)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --bg: #f2f2f7;
  --surface: #ffffff;
  --label: #1c1c1e;
  --secondary: #8e8e93;
  --tertiary: #c6c6c8;
  --blue: #007aff;
  --green: #34c759;
  --red: #ff3b30;
  --orange: #ff9500;
  --separator: #c6c6c8;
  --radius: 12px;
  --font: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', sans-serif;
}}
body {{
  font-family: var(--font);
  background: var(--bg);
  color: var(--label);
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}}
/* ── nav bar ── */
.navbar {{
  position: sticky; top: 0; z-index: 100;
  background: rgba(255,255,255,0.88);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 0.5px solid var(--tertiary);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; height: 52px;
}}
.nav-title {{ font-size: 17px; font-weight: 600; }}
.nav-right {{ font-size: 14px; color: var(--secondary); display: flex; align-items: center; gap: 12px; }}
.logout-link {{ color: var(--blue); text-decoration: none; font-size: 15px; }}
.logout-link:hover {{ opacity: 0.7; }}
/* ── tabs ── */
.tabs {{
  display: flex; gap: 0;
  background: rgba(255,255,255,0.88);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 0.5px solid var(--tertiary);
  padding: 0 16px; overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
.tab {{
  flex-shrink: 0;
  text-decoration: none; color: var(--secondary); font-size: 14px; font-weight: 500;
  padding: 12px 16px; border-bottom: 2px solid transparent; transition: 0.15s;
}}
.tab.active {{ color: var(--blue); border-bottom-color: var(--blue); }}
.tab:hover {{ color: var(--label); }}
/* ── layout ── */
.container {{ max-width: 720px; margin: 0 auto; padding: 24px 16px 48px; }}
/* ── cards ── */
.card {{
  background: var(--surface);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.card-title {{ font-size: 20px; font-weight: 700; margin-bottom: 16px; }}
.card-sub {{ font-size: 13px; color: var(--secondary); margin-bottom: 12px; }}
/* ── stats grid ── */
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.stat {{ background: var(--surface); border-radius: var(--radius); padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.stat-value {{ font-size: 28px; font-weight: 700; }}
.stat-label {{ font-size: 13px; color: var(--secondary); margin-top: 4px; }}
/* ── form ── */
label {{ display: block; font-size: 13px; font-weight: 600; color: var(--secondary); margin-bottom: 4px; margin-top: 14px; }}
input, select {{
  width: 100%; font: inherit; font-size: 16px; color: var(--label);
  background: var(--bg); border: none; border-radius: 10px;
  padding: 12px 14px; outline: none; appearance: none;
  -webkit-appearance: none;
}}
input:focus, select:focus {{ background: #e8e8ee; }}
select {{ background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%238e8e93' stroke-width='1.5' fill='none'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px; }}
textarea {{
  width: 100%; font: inherit; font-size: 14px; color: var(--label);
  background: var(--surface); border: 1px solid var(--tertiary); border-radius: 10px;
  padding: 12px 14px; outline: none; resize: vertical; min-height: 120px;
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
}}
/* ── buttons ── */
.btn {{
  display: inline-flex; align-items: center; justify-content: center;
  font: inherit; font-size: 16px; font-weight: 600;
  border: none; border-radius: 12px; padding: 14px 24px;
  cursor: pointer; transition: 0.15s; text-decoration: none;
  width: 100%; margin-top: 16px;
}}
.btn-primary {{ background: var(--blue); color: white; }}
.btn-primary:hover {{ opacity: 0.85; }}
.btn-primary:disabled {{ opacity: 0.4; cursor: not-allowed; }}
.btn-secondary {{ background: var(--bg); color: var(--blue); }}
.btn-secondary:hover {{ background: #e5e5ea; }}
.btn-destructive {{ background: var(--red); color: white; }}
.btn-destructive:hover {{ opacity: 0.85; }}
.btn-warning {{ background: var(--orange); color: white; }}
/* ── toast ── */
.toast {{
  position: fixed; top: 60px; left: 50%; transform: translateX(-50%);
  background: rgba(0,0,0,0.78); color: white; font-size: 14px; font-weight: 500;
  padding: 10px 22px; border-radius: 20px; z-index: 999;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.2s;
}}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateX(-50%) translateY(-8px); }} to {{ opacity: 1; transform: translateX(-50%) translateY(0); }} }}
/* ── pre / log ── */
pre.log {{
  background: #1c1c1e; color: #e5e5ea; font-size: 12px; line-height: 1.5;
  padding: 16px; border-radius: 10px; overflow: auto; min-height: 160px;
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; white-space: pre-wrap; word-break: break-word;
}}
/* ── misc ── */
.mt-8 {{ margin-top: 8px; }}
.mb-8 {{ margin-bottom: 8px; }}
.text-secondary {{ color: var(--secondary); }}
.text-sm {{ font-size: 13px; }}
.flex-row {{ display: flex; gap: 8px; }}
.flex-row .btn {{ flex: 1; }}
</style>
</head>
<body>
{notice}
{user_html and f'<header class="navbar"><span class="nav-title">手机型号识别</span>{user_html}</header>' or ''}
{tabs_html}
<div class="container">
{body_html}
</div>
</body>
</html>"""


def login_page(error=""):
    err_html = f'<p style="color:var(--red);font-size:14px;text-align:center;margin-bottom:12px;">{safe_text(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>登录 - 手机型号识别</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
:root {{
  --bg: #f2f2f7;
  --surface: #ffffff;
  --label: #1c1c1e;
  --secondary: #8e8e93;
  --blue: #007aff;
  --red: #ff3b30;
  --radius: 12px;
  --font: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', sans-serif;
}}
body {{
  font-family: var(--font);
  background: linear-gradient(145deg, #f2f2f7 0%, #e5e5ea 100%);
  color: var(--label);
  min-height: 100vh; display: flex; align-items: center; justify-content: center;
  -webkit-font-smoothing: antialiased; padding: 24px;
}}
.login-box {{
  background: rgba(255,255,255,0.75);
  backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
  border-radius: 20px; padding: 40px 32px 32px;
  width: 100%; max-width: 380px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.1);
}}
.icon {{ font-size: 48px; text-align: center; margin-bottom: 8px; }}
h1 {{ font-size: 26px; font-weight: 700; text-align: center; margin-bottom: 4px; }}
.sub {{ font-size: 14px; color: var(--secondary); text-align: center; margin-bottom: 28px; }}
label {{ display: block; font-size: 13px; font-weight: 600; color: var(--secondary); margin-bottom: 4px; margin-top: 12px; }}
input {{
  width: 100%; font: inherit; font-size: 16px; color: var(--label);
  background: rgba(242,242,247,0.8); border: 0.5px solid rgba(198,198,200,0.5);
  border-radius: 10px; padding: 14px 16px; outline: none;
  transition: 0.15s;
}}
input:focus {{ background: #e8e8ee; border-color: var(--blue); }}
.btn {{
  display: block; width: 100%; margin-top: 24px;
  font: inherit; font-size: 17px; font-weight: 600;
  background: var(--blue); color: white; border: none; border-radius: 12px;
  padding: 15px; cursor: pointer; transition: 0.15s;
}}
.btn:hover {{ opacity: 0.85; }}
.links {{ text-align: center; margin-top: 20px; font-size: 14px; }}
.links a {{ color: var(--blue); text-decoration: none; margin: 0 8px; }}
.links a:hover {{ text-decoration: underline; }}
.err {{ color: var(--red); font-size: 14px; text-align: center; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="login-box">
  <div class="icon">📱</div>
  <h1>手机型号识别</h1>
  <p class="sub">登录以使用系统</p>
  {err_html}
  <form method="post" action="/login">
    <label>账号</label>
    <input name="username" autocomplete="username" required autofocus>
    <label>密码</label>
    <input name="password" type="password" autocomplete="current-password" required>
    <button class="btn" type="submit">登录</button>
  </form>
  <div class="links">
    <a href="/register">注册账号</a>
    <span style="color:var(--secondary)">|</span>
    <a href="/reset-password">重置密码</a>
  </div>
</div>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════
#  Page builders (called from handler, may check auth)
# ═════════════════════════════════════════════════════════════════════

def _page_dashboard():
    params = read_json(PARAMETERS_PATH, {})
    ckpt_dir = BASE_DIR / "checkpoint"
    model_count = len(list(ckpt_dir.glob("*.pth"))) if ckpt_dir.exists() else 0
    training = _get_training()
    status = "运行中" if training["running"] else "空闲"
    status_color = "var(--green)" if not training["running"] else "var(--orange)"
    body = f"""
    <div class="stats">
      <div class="stat"><div class="stat-value">{model_count}</div><div class="stat-label">模型数量</div></div>
      <div class="stat"><div class="stat-value">{len(BRAND_PATHS)}</div><div class="stat-label">预设厂商</div></div>
      <div class="stat"><div class="stat-value" style="color:{status_color}">{status}</div><div class="stat-label">训练状态</div></div>
    </div>
    <div class="card" style="margin-top:4px;">
      <div class="card-title">最近参数</div>
      <pre class="log" style="min-height:80px;">{safe_text(json.dumps(params, ensure_ascii=False, indent=2))}</pre>
    </div>"""
    return _page_html("控制台", body, active_tab="home")


def _page_account():
    user = _get_state("user", "")
    body = f"""
    <div class="card">
      <div class="card-title">账号信息</div>
      <div style="margin-bottom:8px;"><span class="text-secondary">当前用户：</span>{safe_text(user)}</div>
      <form method="post" action="/logout" style="margin:0;">
        <button class="btn btn-destructive" type="submit">退出登录</button>
      </form>
    </div>
    <div class="card">
      <div class="card-title">修改密码</div>
      <form method="post" action="/reset-password">
        <label>原账号</label><input name="username" value="{safe_text(user)}" required>
        <label>手机号</label><input name="phone_number" required>
        <label>新密码</label><input name="password" type="password" required>
        <button class="btn btn-primary" type="submit">修改</button>
      </form>
    </div>"""
    return _page_html("账号", body, active_tab="account")


def _page_detect():
    options = "".join(
        f'<option value="{safe_text(name)}">{safe_text(name)}</option>' for name in BRAND_PATHS
    )
    params = read_json(PARAMETERS_PATH, {})
    current_image = params.get("image_path") or _get_state("last_image_path") or ""
    output = safe_text(_get_state("last_output"))
    body = f"""
    <div class="card">
      <div class="card-title">型号识别</div>
      <div class="card-sub">选择厂商预设或手动填写路径</div>
      <form method="post" action="/detect" enctype="multipart/form-data">
        <label>厂商预设</label>
        <select name="brand">{options}</select>
        <label>模型路径</label>
        <input name="model1_path" value="{safe_text(params.get("model1_path", ""))}" placeholder="留空使用预设">
        <label>标签路径</label>
        <input name="idx_to_labels_path" value="{safe_text(params.get("idx_to_labels_path", ""))}" placeholder="留空使用预设">
        <label>图片路径</label>
        <input name="image_path" value="{safe_text(current_image)}" placeholder="手动输入或上传图片">
        <label>上传图片</label>
        <input type="file" name="image_file" accept="image/*">
        <button class="btn btn-primary" type="submit">识别</button>
      </form>
    </div>
    <div class="card">
      <div class="card-title">识别结果</div>
      <div class="card-sub" style="word-break:break-all;">{safe_text(current_image or "未选择图片")}</div>
      <pre class="log">{output or "暂无结果"}</pre>
    </div>"""
    return _page_html("型号识别", body, active_tab="detect")


def _page_train():
    params = read_json(PARAMETERS_PATH, {})
    training = _get_training()
    output = safe_text(training["output"])
    running = training["running"]
    btn_disabled = "disabled" if running else ""
    body = f"""
    <div class="card">
      <div class="card-title">模型训练</div>
      <div class="card-sub">数据集目录需包含 train/ 和 val/ 子目录</div>
      <form method="post" action="/train">
        <label>数据集目录</label>
        <input name="dataset_dir" value="{safe_text(params.get("dataset_dir", ""))}" required>
        <label>预训练模型路径</label>
        <input name="model_path" value="{safe_text(params.get("model_path", "./resnet50.pth"))}" required>
        <div class="flex-row">
          <span style="flex:1"><label>Batch Size</label><input name="batch_size" type="number" value="{safe_text(params.get("batch_size", 16))}" min="1"></span>
          <span style="flex:1"><label>Epochs</label><input name="epochs" type="number" value="{safe_text(params.get("epochs", 10))}" min="1"></span>
        </div>
        <div class="flex-row">
          <span style="flex:1"><label>Step Size</label><input name="step_size" type="number" value="{safe_text(params.get("step_size", 5))}" min="1"></span>
          <span style="flex:1"><label>Gamma</label><input name="gamma" type="number" value="{safe_text(params.get("gamma", 0.1))}" step="0.01"></span>
        </div>
        <label>加载方式</label>
        <select name="load_method">
          <option value="init_and_load_model">微调 CBAM 模型</option>
          <option value="load_model">微调普通 ResNet</option>
          <option value="download_and_load_model">微调全部层</option>
        </select>
        <button class="btn btn-primary" {btn_disabled}>{'训练中...' if running else '开始训练'}</button>
      </form>
    </div>
    <div class="card">
      <div class="card-title">训练日志</div>
      <pre class="log">{output or "等待开始训练..."}</pre>
      <a href="/train" class="btn btn-secondary" style="text-align:center;margin-top:12px;">刷新</a>
    </div>"""
    return _page_html("模型训练", body, active_tab="train")


def _page_dataset():
    output = safe_text(_get_state("last_output"))
    body = f"""
    <div class="card">
      <div class="card-title">划分数据集</div>
      <div class="card-sub">将图片按比例分为训练集和验证集</div>
      <form method="post" action="/dataset/split">
        <label>数据集目录</label>
        <input name="dataset_path" required placeholder="图片所在目录路径">
        <label>验证集比例</label>
        <input name="test_frac" type="number" step="0.01" value="0.2" min="0.01" max="0.9">
        <button class="btn btn-primary" type="submit">划分</button>
      </form>
    </div>
    <div class="card">
      <div class="card-title">去重与格式转换</div>
      <div class="card-sub">去除重复图片，统一转为 RGB JPEG 格式</div>
      <form method="post" action="/dataset/convert">
        <label>图片目录</label>
        <input name="dataset_path" required placeholder="图片所在目录路径">
        <button class="btn btn-warning" type="submit">处理</button>
      </form>
    </div>
    <div class="card">
      <div class="card-title">处理日志</div>
      <pre class="log">{output or "暂无处理记录"}</pre>
    </div>"""
    return _page_html("数据集处理", body, active_tab="dataset")


def _page_register(error=""):
    err_html = f'<p style="color:var(--red);font-size:14px;text-align:center;margin-bottom:12px;">{safe_text(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>注册 - 手机型号识别</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
:root {{
  --bg: #f2f2f7; --surface: #fff; --label: #1c1c1e; --secondary: #8e8e93;
  --blue: #007aff; --red: #ff3b30; --radius: 12px;
  --font: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', sans-serif;
}}
body {{
  font-family: var(--font); background: linear-gradient(145deg,#f2f2f7,#e5e5ea);
  color: var(--label); min-height: 100vh; display: flex; align-items: center; justify-content: center;
  -webkit-font-smoothing: antialiased; padding: 24px;
}}
.box {{
  background: rgba(255,255,255,0.75); backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
  border-radius: 20px; padding: 32px; width: 100%; max-width: 380px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.1);
}}
h1 {{ font-size: 24px; font-weight: 700; text-align: center; margin-bottom: 20px; }}
label {{ display: block; font-size: 13px; font-weight: 600; color: var(--secondary); margin-bottom: 3px; margin-top: 12px; }}
input {{
  width: 100%; font: inherit; font-size: 16px; color: var(--label);
  background: rgba(242,242,247,0.8); border: 0.5px solid rgba(198,198,200,0.5);
  border-radius: 10px; padding: 14px 16px; outline: none;
}}
input:focus {{ background: #e8e8ee; border-color: var(--blue); }}
.btn {{ display: block; width: 100%; margin-top: 24px; font: inherit; font-size: 17px; font-weight: 600; background: var(--blue); color: white; border: none; border-radius: 12px; padding: 15px; cursor: pointer; }}
.btn:hover {{ opacity: 0.85; }}
.back {{ text-align: center; margin-top: 16px; font-size: 14px; }}
.back a {{ color: var(--blue); text-decoration: none; }}
.err {{ color: var(--red); font-size: 14px; text-align: center; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="box">
  <h1>注册账号</h1>
  {err_html}
  <form method="post" action="/register">
    <label>账号</label><input name="username" required autofocus>
    <label>密码</label><input name="password" type="password" required>
    <label>手机号</label><input name="phone_number" required>
    <button class="btn" type="submit">注册</button>
  </form>
  <div class="back"><a href="/login">返回登录</a></div>
</div>
</body>
</html>"""


def _page_reset_password(error=""):
    err_html = f'<p style="color:var(--red);font-size:14px;text-align:center;margin-bottom:12px;">{safe_text(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>重置密码 - 手机型号识别</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
:root {{
  --bg: #f2f2f7; --surface: #fff; --label: #1c1c1e; --secondary: #8e8e93;
  --blue: #007aff; --red: #ff3b30; --radius: 12px;
  --font: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', sans-serif;
}}
body {{
  font-family: var(--font); background: linear-gradient(145deg,#f2f2f7,#e5e5ea);
  color: var(--label); min-height: 100vh; display: flex; align-items: center; justify-content: center;
  -webkit-font-smoothing: antialiased; padding: 24px;
}}
.box {{
  background: rgba(255,255,255,0.75); backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
  border-radius: 20px; padding: 32px; width: 100%; max-width: 380px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.1);
}}
h1 {{ font-size: 24px; font-weight: 700; text-align: center; margin-bottom: 20px; }}
label {{ display: block; font-size: 13px; font-weight: 600; color: var(--secondary); margin-bottom: 3px; margin-top: 12px; }}
input {{
  width: 100%; font: inherit; font-size: 16px; color: var(--label);
  background: rgba(242,242,247,0.8); border: 0.5px solid rgba(198,198,200,0.5);
  border-radius: 10px; padding: 14px 16px; outline: none;
}}
input:focus {{ background: #e8e8ee; border-color: var(--blue); }}
.btn {{ display: block; width: 100%; margin-top: 24px; font: inherit; font-size: 17px; font-weight: 600; background: var(--blue); color: white; border: none; border-radius: 12px; padding: 15px; cursor: pointer; }}
.btn:hover {{ opacity: 0.85; }}
.back {{ text-align: center; margin-top: 16px; font-size: 14px; }}
.back a {{ color: var(--blue); text-decoration: none; }}
.err {{ color: var(--red); font-size: 14px; text-align: center; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="box">
  <h1>重置密码</h1>
  {err_html}
  <form method="post" action="/reset-password">
    <label>账号</label><input name="username" required autofocus>
    <label>手机号</label><input name="phone_number" required>
    <label>新密码</label><input name="password" type="password" required>
    <button class="btn" type="submit">重置</button>
  </form>
  <div class="back"><a href="/login">返回登录</a></div>
</div>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════
#  HTTP handler
# ═════════════════════════════════════════════════════════════════════

class PhoneUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler with iOS-style UI."""

    # ── helpers ──

    def _is_authenticated(self):
        with _state_lock:
            return STATE.get("user") is not None

    def _require_auth(self):
        """Redirect to login if not authenticated. Return True if OK."""
        if not self._is_authenticated():
            self.redirect("/login")
            return False
        return True

    # ── GET ──

    def do_GET(self):
        route = urlparse(self.path).path

        # Public pages (no auth required)
        if route == "/login":
            self.respond_html(login_page())
            return
        if route == "/register":
            self.respond_html(_page_register())
            return
        if route == "/reset-password":
            self.respond_html(_page_reset_password())
            return

        # Protected pages
        if not self._require_auth():
            return

        pages = {
            "/": _page_dashboard,
            "/account": _page_account,
            "/detect": _page_detect,
            "/train": _page_train,
            "/dataset": _page_dataset,
        }
        builder = pages.get(route)
        if builder:
            self.respond_html(builder())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    # ── POST ──

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/login":
                self.handle_login()
            elif route == "/register":
                self.handle_register()
            elif route == "/reset-password":
                self.handle_reset_password()
            elif route == "/logout":
                self.handle_logout()
            elif route == "/detect":
                if not self._require_auth():
                    return
                self.handle_detect()
            elif route == "/train":
                if not self._require_auth():
                    return
                self.handle_train()
            elif route == "/dataset/split":
                if not self._require_auth():
                    return
                self.handle_dataset_split()
            elif route == "/dataset/convert":
                if not self._require_auth():
                    return
                self.handle_dataset_convert()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception:
            _set_state(message="操作失败", last_output=traceback.format_exc())
            self.redirect("/")
        finally:
            _set_state(message="")  # clear after each POST

    # ── form parsing ──

    def parse_form(self):
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if content_type.startswith("multipart/form-data"):
            return self._parse_multipart(content_type, body)
        return {key: values[0] for key, values in parse_qs(body.decode("utf-8")).items()}, {}

    @staticmethod
    def _parse_multipart(content_type, body):
        msg = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {content_type}\r\n".encode() + body
        )
        data = {}
        files = {}
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            name = part.get_param("name", header="Content-Disposition")
            if not name:
                continue
            filename = part.get_filename()
            if filename:
                payload = part.get_content()
                if isinstance(payload, str):
                    payload = payload.encode()
                files[name] = _UploadFile(filename, payload)
            else:
                payload = part.get_content()
                data[name] = payload if isinstance(payload, str) else payload.decode("utf-8")
        return data, files

    # ── auth handlers ──

    def handle_login(self):
        data, _ = self.parse_form()
        credentials = read_json(CREDENTIALS_PATH, {})
        username = data.get("username", "").strip()
        password_hash = hash_password(data.get("password", ""))
        if username in credentials and credentials[username]["password"] == password_hash:
            _set_state(user=username, message=f"欢迎回来，{username}")
        else:
            _set_state(message="账号或密码错误")
            self.respond_html(login_page("账号或密码错误"))
            return
        self.redirect("/")

    def handle_register(self):
        data, _ = self.parse_form()
        username = data.get("username", "").strip()
        password = data.get("password", "")
        phone_number = data.get("phone_number", "").strip()
        credentials = read_json(CREDENTIALS_PATH, {})

        if not username.isalnum():
            self.respond_html(_page_register("账号只能包含字母和数字"))
            return
        if not phone_number.isdigit():
            self.respond_html(_page_register("手机号只能包含数字"))
            return
        if username in credentials:
            self.respond_html(_page_register("该账号已注册"))
            return

        credentials[username] = {
            "password": hash_password(password),
            "phone_number": hash_password(phone_number),
        }
        write_json(CREDENTIALS_PATH, credentials)
        _set_state(message=f"账号 {username} 注册成功，请登录")
        self.redirect("/login")

    def handle_reset_password(self):
        data, _ = self.parse_form()
        username = data.get("username", "").strip()
        phone_hash = hash_password(data.get("phone_number", "").strip())
        credentials = read_json(CREDENTIALS_PATH, {})
        if username in credentials and credentials[username]["phone_number"] == phone_hash:
            credentials[username]["password"] = hash_password(data.get("password", ""))
            write_json(CREDENTIALS_PATH, credentials)
            _set_state(message="密码已重置，请登录")
            self.redirect("/login")
        else:
            _set_state(message="账号或手机号错误")
            self.redirect("/reset-password")

    def handle_logout(self):
        _set_state(user=None, message="已退出登录")
        self.redirect("/login")

    # ── detect ──

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

        params = {"idx_to_labels_path": idx_path, "model1_path": model_path, "image_path": image_path}
        write_json(PARAMETERS_PATH, params)
        buffer = io.StringIO()
        processor = test1.VideoProcessor(
            model1_path=model_path, idx_to_labels_path=idx_path, image_path=image_path,
        )
        processor.run_detection("image", image_path, buffer)
        _set_state(last_output=buffer.getvalue(), last_image_path=image_path,
                   message="识别完成")
        self.redirect("/detect")

    # ── train ──

    def handle_train(self):
        with _state_lock:
            if STATE["training"]["running"]:
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

        with _state_lock:
            STATE["training"] = {"running": True, "output": "训练已启动...\n", "started_at": time.time()}

        def run_training():
            try:
                def emit(line):
                    with _state_lock:
                        STATE["training"]["output"] += str(line)
                        if len(STATE["training"]["output"]) > MAX_LOG_CHARS:
                            STATE["training"]["output"] = "...[截断]...\n" + STATE["training"]["output"][-MAX_LOG_CHARS:]

                classifier = train1.PhoneClassifier(**params)
                classifier.train(emit)
                _set_state(message="训练完成")
            except Exception:
                with _state_lock:
                    STATE["training"]["output"] += "\n" + traceback.format_exc()
                _set_state(message="训练失败")
            finally:
                _set_training(running=False)

        threading.Thread(target=run_training, daemon=True).start()
        self.redirect("/train")

    # ── dataset ──

    def handle_dataset_split(self):
        data, _ = self.parse_form()
        dataset_path = data["dataset_path"]
        test_frac = float(data.get("test_frac") or 0.2)
        buffer = io.StringIO()
        splitter = dataset_statistics.DatasetSplitter(dataset_path=dataset_path, test_frac=test_frac)
        splitter.create_folders()
        splitter.split_dataset(buffer.write)
        splitter.save_statistics()
        _set_state(last_output=buffer.getvalue(), message="数据集划分完成")
        self.redirect("/dataset")

    def handle_dataset_convert(self):
        data, _ = self.parse_form()
        dataset_path = data["dataset_path"]
        buffer = io.StringIO()
        splitter = dataset_statistics.DatasetSplitter(dataset_path=dataset_path, test_frac=0.2)
        splitter.remove_duplicates_and_convert_images(callback=buffer.write, directory=dataset_path)
        _set_state(last_output=buffer.getvalue(), message="图片处理完成")
        self.redirect("/dataset")

    # ── HTTP helpers ──

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


# ═════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════

def run_server(host=HOST, port=PORT):
    server = ThreadingHTTPServer((host, port), PhoneUIHandler)
    print(f"手机型号识别 Web 页面已启动: http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
