# AI 智能客服（AI Customer Service）

基于 **Streamlit + Ollama（DeepSeek-R1 8B）** 本地部署的 AI 客服应用，支持多轮对话、流式输出与会话持久化管理。

## 功能特性
- 多轮对话：将完整对话历史传给大模型，实现上下文接续
- 流式输出：打字机式逐字渲染，输入即显即答
- 会话管理：自动保存、历史列表、一键加载 / 删除、当前会话高亮，关闭应用后仍可恢复任意历史对话
- 本地化部署：基于 Ollama 本地运行模型，数据不出本机

## 技术栈
Python · Streamlit · Ollama（OpenAI 兼容 API）· JSON

## 快速开始
1. 安装并启动 [Ollama](https://ollama.com/)，拉取模型：`ollama pull deepseek-r1:8b`
2. 安装依赖：`pip install -r requirements.txt`
3. 启动应用：`streamlit run app.py`
4. 浏览器打开 http://localhost:8501

## 目录结构
- `app.py`：主程序
- `sessions/`：会话存档目录（运行时自动生成）

## 后续计划
- [ ] RAG 知识库（BGE + Chroma 向量检索）
- [ ] 意图识别与自动转人工