import argparse
import locale
import gradio as gr
import os
import subprocess
import json

#BEGIN argparse
parser = argparse.ArgumentParser(description='Puti Infer Web')
parser.add_argument('--public_web', type=bool, default=False, help='this web can be viewed by other computer')
parser.add_argument('--port', '-p', type=int, default=17860, help='port number of web ui')
parser.add_argument('--language', '-l', type=str, default="Auto", help='language of web ui')

args = parser.parse_args()
#END argparse

# Constants
def detect_rvc_path():
  return os.path.abspath(".")

RVC_PATH = detect_rvc_path()
VENV_PATH = os.path.join(RVC_PATH, ".venv", "python.exe")
MODEL_DIR = os.path.join(RVC_PATH, "assets", "weights")
LOG_FILE = "run_infer.log"
PREF_FILE = "run_infer_preferences.json"
RATE_MIN = 0.6
RATE_MAX = 0.75
RATE_STEP = 0.05
DEFAULT_RATE = 0.75

#slice size in mega bytes
SLICE_SIZE_MAX = 100
SLICE_SIZE_MIN = 5
SLICE_SIZE_STEP = 5
DEFAULT_SLICE_SIZE = 50
DEFAULT_SLICE_ENABLE = True

# BEGIN i18n
EN_US_JSON = {
  "TITLE": "English",
  "LOCAL": "en-US",
  "Language": "Language",
  "Source": "Source Folder",
  "Output": "Output Folder",
  "Model": "AI Model File",
  "Index": "Index File (Optional)",
  "Rate": "Index Rate",
  "Run": "Run",
  "Browse": "Browse",
  "Refresh": "Refresh",
  "Log": "Log",
  "Prefer": "Prefer",
  "Shortcut": "Shortcut",
  "Path": "Path",
  "OK": "OK",
  "Cancel": "Cancel",
  "Message Box": "Message Box",
  "Path Dialog": "Path Dialog",
  "Auto Slice File": "Auto Slice File",
  "Auto Slice Size": "Auto Slice Size",
  "OutOfMemoryError": "CUDA out of memory, file too big",
  "Please select folder": "Please Select Folder",
  "Select a File": "Select a File",
  "Command Output": "Command Output",
  "Command": "Command",
  "Index File": "Index File: {0}",
  "No Index File": "No Index File",
  "Select a WAV File": "Select a WAV File",
  "Select a Index File": "Select a Index File",
  "Select folder to save shortcut": "Select folder to save shortcut",
  "Command successfully executed": "Command successfully executed",
  "CommandDetail": "Will Run command with`n`nSource: {0}`n`nOutput: {1:}`n`n{2:}`n`n=> {3:}`n`nWill Freeze for a while running command, and another Dialog will shown up once finished.",
  "Source is required": "Source is required",
  "Program might be freeze for awhile to install neccessary tools": "Program might be freeze for awhile to install neccessary tools",
  "Infomation": "Infomation",
  "SelfFolder": "Self Folder",
}
ZH_TW_JSON = {
  "TITLE": "繁體中文",
  "LOCAL": "zh-TW",
  "Language": "語言",
  "Source": "來源目錄",
  "Output": "輸出目錄",
  "Model": "AI 模型",
  "Index": "Index 檔 (選填)",
  "Rate": "Index Rate",
  "Run": "執行",
  "Browse": "瀏覽",
  "Refresh": "更新",
  "Log": "執行記錄",
  "Prefer": "設定",
  "Shortcut": "建立捷徑",
  "Path": "路徑",
  "OK": "OK",
  "Cancel": "取消",
  "Message Box": "訊息",
  "Path Dialog": "選擇路徑",
  "Auto Slice File": "自動切割檔案",
  "Auto Slice Size": "自動切割大小",
  "OutOfMemoryError": "CUDA 記憶體不足，檔案太大",
  "Please select folder": "請選擇資料夾",
  "Select a File": "選擇檔案",
  "Command Output": "執行結果",
  "Command": "指令",
  "Index File": "Index 檔: {0}",
  "No Index File": "無 Index 檔",
  "Select a WAV File": "選擇 WAV 檔",
  "Select a Index File": "選擇 Index 檔",
  "Select folder to save shortcut": "選擇儲存捷徑的資料夾",
  "Command successfully executed": "指令執行成功",
  "CommandDetail": "將會執行指令`n`n來源檔: {0}`n`n輸出位置: {1}`n`n{2}`n`n=> {3}`n`n本程式將會凍結一段時間無反應，執行結束時會跳出另一個對話框。",
  "Source is required": "「來源 WAV 檔」為必填",
  "Program might be freeze for awhile to install neccessary tools": "程式可能會凍結一段時間，以安裝必要工具",
  "Infomation": "資訊",
  "SelfFolder": "安裝資料夾",
}

def load_language_list(language):
    if language == "en-US":
        language_list = EN_US_JSON
    elif language == "zh-TW":
        language_list = ZH_TW_JSON
    else:
        #only detect first 2 letter
        language = language[:2]
        if language == "zh":
            language_list = ZH_TW_JSON
        else:
            language_list = EN_US_JSON
    return language_list
def on_change_language(language):
    global currentI18n
    currentI18n = load_language_list(language)
    return currentI18n
def detect_language():
    #args.language if not None or not Auto else locale.getdefaultlocale()[0]
    if args.language is not None and args.language != "Auto":
        return args.language
    else:
        return locale.getdefaultlocale()[0]
def _(key, *args: object):
    global currentI18n
    '''
    Get i18n string

    key: string

    *args: object; must be same number defined in key

    return: string
    '''
    data = currentI18n.get(key)
    #check if existed
    if data is None:
        return key
    return data.format(*args)
# END i18n

# BEGIN preference
class Preference:
    def __init__(self):
        self.data = {
            "source": "",
            "output": "",
            "model": get_model_list()[0],
            "index": "",
            "rate": DEFAULT_RATE,
            "auto_slice": DEFAULT_SLICE_ENABLE,
            "slice_size": DEFAULT_SLICE_SIZE
        }
        self.load_preferences()

    def save_preferences(self):
        try:
            with open(PREF_FILE, "w") as f:
                json.dump(self.data, f)
        except Exception as e:
            log_message(f"Error saving preferences: {str(e)}", level="ERROR")

    def load_preferences(self):
        if os.path.exists(PREF_FILE):
            try:
                with open(PREF_FILE, "r") as f:
                    self.data.update(json.load(f))
            except Exception as e:
                log_message(f"Error loading preferences: {str(e)}", level="ERROR")

    def get_preference(self, key):
        return self.data.get(key)
    def update_preference(self, key, value):
        self.data[key] = value
        self.save_preferences()

    def get_source(self):
        return self.get_preference("source")
    def update_source(self, value):
        self.update_preference("source", value)

    def get_output(self):
        return self.get_preference("output")
    def update_output(self, value):
        self.update_preference("output", value)
    
    def get_model(self):
        return self.get_preference("model")
    def update_model(self, value):
        self.update_preference("model", value)

    def get_index(self):
        return self.get_preference("index")
    def update_index(self, value):
        self.update_preference("index", value)

    def get_rate(self):
        return self.get_preference("rate")
    def update_rate(self, value):
        self.update_preference("rate", value)

    def get_auto_slice(self):
        return self.get_preference("auto_slice")
    def update_auto_slice(self, value):
        self.update_preference("auto_slice", value)

    def get_slice_size(self):
        return self.get_preference("slice_size")
    def update_slice_size(self, value):
        self.update_preference("slice_size", value)
# END preference

# Helper functions
def log_message(message, level="INFO"):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{level}] {message}\n")

# BEGIN File dialog functions
def open_select_dir_dialog(preferred_dir: str = None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(initialdir=preferred_dir)

def open_select_file_dialog(extension: str = None, preferred_dir: str = None):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(initialdir=preferred_dir, filetypes=[(f"{extension} files", f"*.{extension}")])

#END file dialog functions

def get_model_list():
    if not os.path.exists(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            log_message(result.stderr, level="ERROR")
            return False, result.stderr
        log_message(result.stdout, level="INFO")
        return True, result.stdout
    except Exception as e:
        log_message(str(e), level="ERROR")
        return False, str(e)

#UI interaction
def refresh_model_list(current_value):
    '''
    https://discuss.huggingface.co/t/how-to-update-the-gr-dropdown-block-in-gradio-blocks/19231/2

    Please note that gr.Dropdown.update has been deprecated in version 4.x. You can directly use gr.Dropdown instead
    '''
    d = get_model_list()
    #if current_value not in d:
    if current_value not in d:
      current_value = d[0]
    #use this if 3.x.x
    return gr.Dropdown.update(choices=d, value=current_value)
    #use this if 4.x.x or latter
    #return gr.Dropdown(choices=d, value=current_value)

def update_success_status(success):
    return gr.update(visible=success), gr.update(visible=not success)

def infer_ui(source, output, model, index, rate, auto_slice, slice_size):
    #if not set auto_slice, set slice_size to very large number to disable slicing
    if not auto_slice:
        slice_size = 99999999

    process_command = [
        VENV_PATH, "run_infer.py",
        "--input_dir", source,
        "--output_dir", output,
        "--model", model,
        "--index_rate", str(rate),
        "--splice_size_mb", str(slice_size)
    ]
    if index:
        process_command.extend(["--index_path", index])
    log_message(f"Running command: {' '.join(process_command)}")
    try:
        result = subprocess.run(process_command, capture_output=True, text=True)
        if result.returncode != 0:
            log_message(result.stderr, level="ERROR")
            return False, result.stderr
        log_message(result.stdout, level="INFO")
        return True, result.stdout
    except Exception as e:
        log_message(str(e), level="ERROR")
        return False, str(e)

# Initialize preferences
preferences = Preference()
#init i18n
on_change_language(detect_language())

with gr.Blocks() as webapp:
    gr.Markdown("# Retrieval-based Voice Conversion")
    with gr.Row():
        source = gr.Textbox(label=_("Source"), interactive=True, value=preferences.get_source())
        source_browser = gr.Button(_("Browse"))
        source_browser.click(
            fn=lambda: open_select_dir_dialog(),
            inputs=[],
            outputs=source
        )
        source.change(preferences.update_source, inputs=[source], outputs=[])
    with gr.Row():
        output = gr.Textbox(label=_("Output"), interactive=True, value=preferences.get_output())
        output_browser = gr.Button(_("Browse"))
        output_browser.click(
            fn=lambda: open_select_dir_dialog(),
            inputs=[],
            outputs=output
        )
        output.change(preferences.update_output, inputs=[output], outputs=[])
    with gr.Row():
        model = gr.Dropdown(choices=get_model_list(), value=preferences.get_model(), label=_("Model"))
        refresh_model = gr.Button(_("Refresh"))
        refresh_model.click(
            refresh_model_list,
            inputs=model,
            outputs=[model]
        )
        model.change(preferences.update_model, inputs=[model], outputs=[])
    with gr.Row():
        index = gr.Textbox(label=_("Index"), interactive=True, value=preferences.get_index())
        browse_button = gr.Button(value=_("Browse"))
        browse_button.click(
            fn=lambda: open_select_file_dialog("index", RVC_PATH),
            inputs=[],
            outputs=index
        )
        index.change(preferences.update_index, inputs=[index], outputs=[])
        rate = gr.Slider(RATE_MIN, RATE_MAX, value=preferences.get_rate(), step=RATE_STEP, label=_("Index Rate"))
        rate.change(preferences.update_rate, inputs=[rate], outputs=[])
    with gr.Row():
        auto_slice = gr.Checkbox(label=_("Auto Slice"), value=preferences.get_auto_slice())
        auto_slice.change(preferences.update_auto_slice, inputs=[auto_slice], outputs=[])
        with gr.Row():
            slice_size = gr.Slider(label=_("Auto Slice Size") + " (MB)", value=preferences.get_slice_size(), maximum=SLICE_SIZE_MAX, minimum=SLICE_SIZE_MIN, step=SLICE_SIZE_STEP)
            slice_size.change(preferences.update_slice_size, inputs=[slice_size], outputs=[])
    with gr.Row():
        run_button = gr.Button(_("Run"))
    #show result
    with gr.Row():
        with gr.Column(scale=1):
            success_span = gr.HTML(label=_("Success"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 green;'>✔️</span>", visible=False)
            error_span = gr.HTML(label=_("Error"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 red;'>❌</span>", visible=False)
        is_success = gr.Checkbox(visible=False)
        with gr.Column(scale=4):
            result = gr.TextArea(label=_("Command Output"), value="", interactive=False)
        is_success.change(
            fn=update_success_status,
            inputs=[is_success],
            outputs=[success_span, error_span]
        )
    
    run_button.click(
        fn=infer_ui,
        inputs=[source, output, model, index, rate, auto_slice, slice_size],
        outputs=[is_success, result],
    )

webapp.launch(share=args.public_web, server_port=args.port, inbrowser=True)
