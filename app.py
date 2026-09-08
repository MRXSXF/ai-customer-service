# AI 智能客服:基于 Streamlit 与 Ollama 本地大模型的多轮对话应用
# 会话存档于 sessions/,模型与提示词预设存于 presets.json

import datetime
import json
import os
import re

import streamlit as st
from openai import OpenAI

WORKDIR = "sessions"
PRESET_FILE = "presets.json"
MODEL = "deepseek-r1:8b"
ENDPOINT = "http://localhost:11434/v1"
FAKE_KEY = "ollama"
TITLE_CAP = 20

# 生成时间戳作为默认会话名
def now_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 把文本整理成 Windows 合法文件名
def safe_title(text):
    flat = re.sub(r"\s+", " ", str(text)).strip()
    return re.sub(r'[\\/:*?"<>|]', "-", flat)

# 会话名对应到 JSON 存档路径
def path_of(session):
    return os.path.join(WORKDIR, safe_title(session) + ".json")

# 用首条用户消息命名新会话,重名追加时间后缀
def derive_title(messages):
    for item in messages:
        if item["role"] == "user":
            name = safe_title(item["content"])[:TITLE_CAP]
            if not name:
                break
            if os.path.exists(path_of(name)):
                name = f"{name}_{datetime.datetime.now().strftime('%H%M%S')}"
            return name
    return now_stamp()

# 把当前会话写入存档
def persist():
    os.makedirs(WORKDIR, exist_ok=True)
    payload = {
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages,
    }
    with open(path_of(st.session_state.current_session), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

# 列出全部已保存会话,新的在前
def known_sessions():
    if not os.path.isdir(WORKDIR):
        return []
    return sorted(
        (f[:-5] for f in os.listdir(WORKDIR) if f.endswith(".json")),
        reverse=True,
    )

# 加载指定会话到当前界面
def restore(name):
    target = path_of(name)
    if not os.path.exists(target):
        st.warning(f"未找到会话:{name}")
        return
    try:
        with open(target, encoding="utf-8") as fh:
            blob = json.load(fh)
        st.session_state.current_session = name
        st.session_state.messages = blob.get("messages", [])
    except Exception as err:
        st.error(f"会话读取失败:{err}")

# 删除指定会话的存档
def drop_session(name):
    target = path_of(name)
    if os.path.exists(target):
        os.remove(target)

# 读取预设,文件缺失或损坏时回落默认值
def load_presets():
    seed = {"models": [MODEL], "prompts": ["你是一个AI智能客服"]}
    if not os.path.exists(PRESET_FILE):
        return seed
    try:
        with open(PRESET_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("models", "prompts"):
            if not isinstance(data.get(key), list) or not data[key]:
                data[key] = seed[key]
        return data
    except Exception:
        return seed

# 保存预设列表到文件
def save_presets(data):
    with open(PRESET_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

# 向预设追加一项(空值/重复忽略)
def add_preset(kind, value, data):
    value = str(value).strip()
    if value and value not in data[kind]:
        data[kind].append(value)
        save_presets(data)

# 从预设移除一项(至少保留一个)
def remove_preset(kind, value, data):
    if len(data[kind]) > 1 and value in data[kind]:
        data[kind].remove(value)
        save_presets(data)

# 组装请求并调用 Ollama,返回流式响应
def ask_model(model, system_prompt, history):
    client = OpenAI(api_key=FAKE_KEY, base_url=ENDPOINT)
    payload = [{"role": "system", "content": system_prompt}, *history]
    return client.chat.completions.create(
        model=model,
        messages=payload,
        stream=True,
    )

# 把流式响应拆成文本片段逐个产出
def piece_text(stream):
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

st.set_page_config(
    page_title="AI智能客服",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session" not in st.session_state:
    st.session_state.current_session = now_stamp()

presets = load_presets()
model_options = presets["models"]
prompt_options = presets["prompts"]

current_model = st.session_state.get("active_model", model_options[0])
if current_model not in model_options:
    current_model = model_options[0]
current_prompt = st.session_state.get("active_prompt", prompt_options[0])
if current_prompt not in prompt_options:
    current_prompt = prompt_options[0]

with st.sidebar:
    st.header("AI 控制面板")

    st.subheader("模型")
    st.session_state.active_model = st.selectbox(
        "选择模型", model_options, index=model_options.index(current_model)
    )
    with st.expander("管理模型"):
        new_model = st.text_input("新模型名称", key="new_model", placeholder="如 qwen2.5:7b")
        if st.button("保存新模型"):
            add_preset("models", new_model, presets)
            st.rerun()
        if len(model_options) > 1:
            if st.button("删除当前模型"):
                remove_preset("models", st.session_state.active_model, presets)
                st.session_state.active_model = load_presets()["models"][0]
                st.rerun()

    st.divider()

    st.subheader("系统提示词")
    st.session_state.active_prompt = st.selectbox(
        "选择提示词", prompt_options, index=prompt_options.index(current_prompt)
    )
    with st.expander("管理提示词"):
        new_prompt = st.text_area("新提示词", key="new_prompt", height=90)
        if st.button("保存新提示词"):
            add_preset("prompts", new_prompt, presets)
            st.rerun()
        if len(prompt_options) > 1:
            if st.button("删除当前提示词"):
                remove_preset("prompts", st.session_state.active_prompt, presets)
                st.session_state.active_prompt = load_presets()["prompts"][0]
                st.rerun()

    st.divider()

    if st.button("新建会话", width="stretch"):
        if st.session_state.messages:
            persist()
        st.session_state.current_session = now_stamp()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("会话历史")
    sessions = known_sessions()
    if not sessions:
        st.caption("暂无历史会话,开始聊天后自动保存")

    for name in sessions:
        is_current = name == st.session_state.current_session
        col_a, col_b = st.columns([4, 1])
        with col_a:
            if st.button(
                name,
                width="stretch",
                icon="🗂",
                key=f"load_{name}",
                type="primary" if is_current else "secondary",
            ):
                restore(name)
                st.rerun()
        with col_b:
            if st.button(
                "",
                width="stretch",
                icon="❌",
                key=f"del_{name}",
            ):
                drop_session(name)
                if st.session_state.current_session == name:
                    st.session_state.current_session = now_stamp()
                    st.session_state.messages = []
                st.rerun()

st.title("AI智能客服")
st.caption(st.session_state.active_model)

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])

prompt = st.chat_input("请问有什么可以帮您?")
if prompt:
    fresh = not st.session_state.messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        stream = ask_model(
            st.session_state.active_model,
            st.session_state.active_prompt,
            st.session_state.messages,
        )
        with st.chat_message("assistant"):
            answer = st.write_stream(piece_text(stream))
    except Exception as err:
        st.session_state.messages.pop()
        st.error(f"请求失败,已撤销本条消息:{err}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if fresh:
        st.session_state.current_session = derive_title(st.session_state.messages)

    persist()
