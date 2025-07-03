import argparse
import locale
import gradio as gr
import os
import subprocess
import json
import datetime

RVC_PATH = os.path.dirname(os.path.realpath(__file__))
VENV_PATH = os.path.join(RVC_PATH, ".venv", "python.exe")
if not os.path.exists(VENV_PATH):
    VENV_PATH = os.path.join(RVC_PATH, "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PATH):
    raise FileNotFoundError(f"Virtual environment python executable not found at {VENV_PATH}. Please create a virtual environment first.")

LOG_FILE = os.path.join(RVC_PATH, "run_infer.log")
log_file = None

def log_message(message, level="INFO"):
    global log_file
    if log_file is None:
        log_file = LOG_FILE
        print(f"[WARNING] using default log file: {log_file}")
    with open(log_file, "a", encoding='utf-8') as log_fd:
        log_fd.write(f"[{level}] {message}\n")

#BEGIN argparse
parser = argparse.ArgumentParser(description='Puti Infer Web')
parser.add_argument('--public_web', type=bool, default=False, help='this web can be viewed by other computer')
parser.add_argument('--port', '-p', type=int, default=17860, help='port number of web ui')
parser.add_argument('--language', '-l', type=str, default="Auto", help='language of web ui')

args = parser.parse_args()
#END argparse

# Constants
MODEL_DIR = os.path.join(RVC_PATH, "assets", "weights")
PREF_FILE = os.path.join(RVC_PATH, "run_infer_preferences.json")

#generate log file path base on today date
log_file = os.path.join(RVC_PATH, "infer_logs", f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
#create log folder if not existed
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# BEGIN i18n
EN_US_JSON = {
  "TITLE": "English",
  "LOCAL": "en-US",
  "Language": "Language",
  "Source": "Source Folder",
  "Output": "Output Folder",
  "Model": "AI Model File",
  "Index": "Index File (Optional)",
  "Run": "Run",
  "Browse": "Browse",
  "Refresh": "Refresh",
  "Log": "Log Path",
  "Prefer": "Prefer",
  "Shortcut": "Shortcut",
  "Path": "Path",
  "OK": "OK",
  "Cancel": "Cancel",
  "Message Box": "Message Box",
  "Path Dialog": "Path Dialog",
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
  "ProgramRunningPleaseDontClose": "Program is running, please don't close",
}
ZH_TW_JSON = {
  "TITLE": "繁體中文",
  "LOCAL": "zh-TW",
  "Language": "語言",
  "Source": "來源目錄",
  "Output": "輸出目錄",
  "Model": "AI 模型",
  "Index": "Index 檔 (選填)",
  "Run": "執行",
  "Browse": "瀏覽",
  "Refresh": "更新",
  "Log": "記錄位置",
  "Prefer": "設定",
  "Shortcut": "建立捷徑",
  "Path": "路徑",
  "OK": "OK",
  "Cancel": "取消",
  "Message Box": "訊息",
  "Path Dialog": "選擇路徑",
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
  "ProgramRunningPleaseDontClose": "程式正在執行中，請勿關閉",
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

# END preference

# Helper functions
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

def on_infer_button_click():
    """right after button clicked, change ui status

    Returns:
        status of changed ui elements
    """
    return gr.update(visible=False), gr.update(visible=False)

def handle_infer_ui_result(success):
    """
    Handle the result of the inference UI operation.
    
    :param success: Boolean indicating if the operation was successful.
    :param result: The output message from the operation.
    :return: Tuple of updates for success and error spans, and the result text area.
    """
    print(f'try update success as {success}, {not success}')
    return gr.update(visible=success), gr.update(visible=not success)

def infer_ui(source: str, output: str, model: str, index: str):
    global log_file
    process_command = [
        VENV_PATH, os.path.join(RVC_PATH, "run_infer.py"),
        "--input_dir", source,
        "--output_dir", output,
        "--model", model,
        "--log_file", log_file,
    ]
    if index:
        process_command.extend(["--index_path", index])
    try:
        print(f"Running command: {' '.join(process_command)}")
        log_message(f"Running command: {' '.join(process_command)}", level="INFO")

        result = subprocess.run(process_command, capture_output=True, text=True, cwd=RVC_PATH, encoding="utf-8")
        if result.returncode != 0:
            log_message(result.stdout + '\n' + result.stderr, level="ERROR")
            print(f"Command failed with error: {result.stderr}")
            return False, result.stdout
        else:
            log_message(result.stdout, level="INFO")
            print(f"Command executed successfully: {result.stdout}")
            return True, result.stdout
    except Exception as e:
        log_message(str(e), level="ERROR")
        print(f"Error running command: {str(e)}")
        return False, str(e)

# Initialize preferences
preferences = Preference()
#init i18n
on_change_language(detect_language())

with gr.Blocks() as webapp:
    gr.Markdown("# Retrieval-based Voice Conversion")
    with gr.Row():
        source = gr.Textbox(label=_("Source"), interactive=True, value=preferences.get_source()).style(show_copy_button=True)
        source_browser = gr.Button(_("Browse"))
        source_browser.click(
            fn=lambda: open_select_dir_dialog(),
            inputs=[],
            outputs=source
        )
        source.change(preferences.update_source, inputs=[source], outputs=[])
    with gr.Row():
        output = gr.Textbox(label=_("Output"), interactive=True, value=preferences.get_output()).style(show_copy_button=True)
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
            #default folder is RVC_PATH/assets/weights
            fn=lambda: open_select_file_dialog("index", RVC_PATH),
            inputs=[],
            outputs=index
        )
        index.change(preferences.update_index, inputs=[index], outputs=[])
    with gr.Row():
        run_button = gr.Button(_("Run"))
    #show result
    with gr.Row():
        with gr.Column(scale=1):
            success_span = gr.HTML(label=_("Success"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 green;'>✔️</span>", visible=False)
            error_span = gr.HTML(label=_("Error"), value="<span style='font-size: 5rem; color: transparent; text-shadow: 0 0 0 red;'>❌</span>", visible=False)
            full_log_path = gr.Textbox(label=_("Log"), value=log_file, interactive=False).style(show_copy_button=True)
        is_success = gr.Checkbox(visible=False)
        with gr.Column(scale=4):
            result = gr.TextArea(label=_("Command Output"), value="", interactive=False)
    
    run_button.click(
        fn=on_infer_button_click,
        inputs=[],
        outputs=[success_span, error_span]
    ).then(
        fn=infer_ui,
        inputs=[source, output, model, index],
        outputs=[is_success, result],
    ).then(
        fn=handle_infer_ui_result,
        inputs=[is_success],
        outputs=[success_span, error_span]
    )
print(currentI18n["ProgramRunningPleaseDontClose"])
webapp.launch(share=args.public_web, server_port=args.port, inbrowser=True)
