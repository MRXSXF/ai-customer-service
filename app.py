import datetime
import json
import os
import re

import streamlit as st
from openai import OpenAI

# ==================== ① 常量配置（改这里即可） ====================
SESSION_DIR = "sessions"                    # 会话保存目录
MODEL_NAME = "deepseek-r1:8b"               # Ollama 中的模型名
BASE_URL = "http://localhost:11434/v1"      # Ollama 本地服务地址
API_KEY = "ollama"                          # Ollama 要求的固定假 Key
SYSTEM_PROMPT = "你是一个AI智能客服"
NAME_MAX_LEN = 20                           # 用问题命名时最长字数

# ==================== ② 会话工具函数 ====================

def default_session_name():
    """默认会话名：时间戳（下划线格式，避免 Windows 非法字符）"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def clean_name(text):
    """把任意文字变成 Windows 合法文件名：转字符串、去非法字符"""
    text = re.sub(r"\s+", " ", str(text)).strip()      # 压缩换行/多余空格
    return re.sub(r'[\\/:*?"<>|]', "-", text)          # 去掉非法字符


def _session_path(session_name):
    """由会话名生成文件路径"""
    return os.path.join(SESSION_DIR, clean_name(session_name) + ".json")


def make_session_name(messages):
    """用第一条用户问题给会话起名；重名时加时间后缀；无消息退回时间戳"""
    for msg in messages:
        if msg["role"] == "user":
            name = clean_name(msg["content"])[:NAME_MAX_LEN]
            if not name:                                # 内容全是符号 → 换兜底
                break
            if os.path.exists(_session_path(name)):     # 防覆盖
                name = f"{name}_{datetime.datetime.now().strftime('%H%M%S')}"
            return name
    return default_session_name()


def save_session():
    """把当前会话（名字 + 消息）保存成 JSON 文件"""
    os.makedirs(SESSION_DIR, exist_ok=True)             # 目录不存在就创建
    data = {
        "current_session": st.session_state.current_session,
        "messages": st.session_state.messages,
    }
    with open(_session_path(st.session_state.current_session), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)   # 中文直显 + 缩进


def load_sessions():
    """列出所有已保存的会话名（倒序 = 新的在前）"""
    if not os.path.exists(SESSION_DIR):
        return []
    names = []
    for filename in os.listdir(SESSION_DIR):
        if filename.endswith(".json"):
            names.append(filename[:-5])
    return sorted(names, reverse=True)


def load_session(session_name):
    """把指定会话读入内存（以文件名为准，不信任文件内容）"""
    path = _session_path(session_name)
    if not os.path.exists(path):
        st.warning(f"会话不存在：{session_name}")
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.current_session = session_name          # ✅ 文件名是真相
        st.session_state.messages = data.get("messages", [])     # ✅ 防缺字段
    except Exception as e:
        st.error(f"加载会话失败：{e}")


def delete_session(session_name):
    """删除指定会话文件"""
    path = _session_path(session_name)
    if os.path.exists(path):
        os.remove(path)


# ==================== ③ 页面配置（必须是第一个 st 调用） ====================
st.set_page_config(
    page_title="AI智能客服",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ==================== ④ 会话状态初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session" not in st.session_state:
    st.session_state.current_session = default_session_name()

st.title("AI智能客服")
st.caption("多轮对话 · 流式输出 · 会话自动保存")

# ==================== ⑤ 渲染历史消息 ====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==================== ⑥ 侧边栏 ====================
with st.sidebar:
    st.header("AI 控制面板")

    # ---- 新建会话 ----
    if st.button("新建会话", width="stretch"):
        if st.session_state.messages:              # 旧会话有内容才保存（防空文件）
            save_session()
        st.session_state.current_session = default_session_name()   # 新身份
        st.session_state.messages = []                              # 清空对话
        st.rerun()

    # ---- 会话历史 ----
    st.subheader("会话历史")
    sessions_list = load_sessions()
    if not sessions_list:
        st.caption("暂无历史会话，开始聊天后自动保存")

    for session in sessions_list:
        is_current = (session == st.session_state.current_session)
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(session, width="stretch", icon="🗂",
                         key=f"load_{session}",
                         type="primary" if is_current else "secondary"):
                load_session(session)
                st.rerun()
        with col2:
            if st.button("", width="stretch", icon="❌", key=f"del_{session}"):
                delete_session(session)
                if st.session_state.current_session == session:    # 删的是当前会话？
                    st.session_state.current_session = default_session_name()
                    st.session_state.messages = []
                st.rerun()


# ==================== ⑦ 主对话 ====================
prompt = st.chat_input("请问有什么可以帮您？")

if prompt:
    is_new_chat = not st.session_state.messages       # 开聊前是空会话吗？

    # 1. 记录并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. 调用模型（流式）
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *st.session_state.messages,                # 展开历史消息
        ],
        stream=True,
    )

    # 3. 流式接收并逐字渲染（打字机效果）
    box = st.empty()
    full_response = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:   # 防末尾空块
            full_response += chunk.choices[0].delta.content
            box.chat_message("assistant").write(full_response)

    # 4. 保存完整回复到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 5. 新会话的第一句话 → 用问题给会话命名（之后不再改）
    if is_new_chat:
        st.session_state.current_session = make_session_name(st.session_state.messages)

    # 6. 自动保存
    save_session()