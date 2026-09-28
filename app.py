import os
import uuid
import streamlit as st
from getpass import getpass

# ============================================================
# 🤖 AGENTIC AI LEARNING ASSISTANT
# Streamlit + LangChain + Groq + One Custom Tool
# ============================================================

st.set_page_config(
    page_title="Agentic AI Learning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- CUSTOM CSS -------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef7f5 100%);
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #182848 0%, #355c9a 55%, #2b8a83 100%);
        box-shadow: 0 10px 30px rgba(28, 49, 91, .16);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { margin: .6rem 0 0 0; font-size: 1.05rem; opacity: .92; }
    .info-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 120px;
        box-shadow: 0 4px 14px rgba(30, 50, 80, .05);
    }
    .info-card h3 { margin-top: 0; color: #20345c; }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.6rem;
    }
    [data-testid="stSidebar"] {
        background: #f0f4fb;
        border-right: 1px solid #dce5f2;
    }
    .small-note { color: #64748b; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------- SESSION STATE -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    st.session_state.agent = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "ready" not in st.session_state:
    st.session_state.ready = False

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.markdown("## 🤖 Agentic AI Lab")
    page = st.radio(
        "Navigate",
        ["💬 AI Chat", "🧪 Tool Playground", "📘 Learn the Concepts", "🛠️ Setup & Help"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### 🔌 Groq Connection")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a key from https://console.groq.com/keys",
        placeholder="Enter your Groq API key",
    )
    model_name = st.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
        index=0,
        help="Choose a model available in your Groq account.",
    )
    if st.button("🚀 Connect & Initialize", use_container_width=True):
        if not api_key.strip():
            st.error("Please enter your Groq API key.")
        else:
            try:
                from langchain_groq import ChatGroq
                from langchain.tools import tool
                from langchain.agents import create_agent
                from langgraph.checkpoint.memory import InMemorySaver

                llm = ChatGroq(
                    model=model_name,
                    temperature=0,
                    api_key=api_key.strip(),
                )

                @tool
                def text_assistant(task: str, text: str) -> str:
                    """Summarize text or answer a question using the provided context.
                    task should be 'summarize' or 'question'; text contains the content."""
                    prompt = f"""
You are a helpful text assistant.
Task: {task}
Input:
{text}

Instructions:
- If task is summarize, provide a clear and concise summary.
- If task is question, answer based on the provided context.
- If the answer is not in the context, say that clearly.
- Use simple and understandable language.
"""
                    result = llm.invoke(prompt)
                    return result.content if isinstance(result.content, str) else str(result.content)

                agent = create_agent(
                    model=llm,
                    tools=[text_assistant],
                    checkpointer=InMemorySaver(),
                    system_prompt="""
You are an Agentic AI learning assistant.
Understand the user's request. Use the text_assistant tool for text summarization
or context-based question answering. Answer clearly and simply. For general
questions, answer directly when the tool is not needed.
""",
                )
                st.session_state.llm = llm
                st.session_state.agent = agent
                st.session_state.ready = True
                st.session_state.thread_id = str(uuid.uuid4())
                st.success("Agent initialized! You can start chatting.")
            except Exception as e:
                st.session_state.ready = False
                st.error(f"Could not initialize the agent: {e}")
    if st.session_state.ready:
        st.success("🟢 Agent is connected")
    else:
        st.info("⚪ Connect your Groq key to start.")
    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption("Your API key is used for this session and is not displayed.")

# ------------------------- HEADER -------------------------
st.markdown("""
<div class="hero">
    <h1>😊 Agentic AI Learning Assistant 📝</h1>
    <p>Learn how an AI agent uses Groq, one custom tool, and conversation memory.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- CHAT PAGE -------------------------
if page == "💬 AI Chat":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><h3>⚡ Groq LLM</h3><p>Processes your prompts and generates responses.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><h3>🧰 One custom tool</h3><p>Summarizes text or answers questions from supplied context.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><h3>🧠 Memory</h3><p>Conversation state is retained within the same thread.</p></div>', unsafe_allow_html=True)

    st.markdown("### 💬 Chat with your agent")
    st.caption("Try: “Explain Agentic AI” or “Summarize this text: ...”")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("Ask a question or paste text to summarize...")
    if prompt:
        if not st.session_state.ready or st.session_state.agent is None:
            st.warning("Enter your Groq API key in the sidebar and click Connect & Initialize.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    try:
                        result = st.session_state.agent.invoke(
                            {"messages": [{"role": "user", "content": prompt}]},
                            config={"configurable": {"thread_id": st.session_state.thread_id}},
                        )
                        answer = result["messages"][-1].content
                        if not isinstance(answer, str):
                            answer = str(answer)
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Request failed: {e}")

# ------------------------- TOOL PLAYGROUND -------------------------
elif page == "🧪 Tool Playground":
    st.markdown("## 🧪 One-tool playground")
    st.write("This demonstrates the single `text_assistant` tool from the notebook.")
    task = st.radio("Choose a task", ["Summarize", "Question answering"], horizontal=True)
    context = st.text_area(
        "Enter text or context",
        height=220,
        placeholder="Paste an article, notes, or a paragraph here...",
    )
    question = ""
    if task == "Question answering":
        question = st.text_input("Your question", placeholder="What is the main idea?")
    if st.button("▶️ Run tool", type="primary"):
        if not st.session_state.ready or st.session_state.llm is None:
            st.warning("Connect your Groq API key in the sidebar first.")
        elif not context.strip():
            st.warning("Please enter some text or context.")
        elif task == "Question answering" and not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                task_value = "summarize" if task == "Summarize" else "question"
                tool_input = context if task_value == "summarize" else f"Context:\n{context}\n\nQuestion: {question}"
                with st.spinner("Running the tool..."):
                    output = st.session_state.llm.invoke(
                        f"""Task: {task_value}\nInput:\n{tool_input}\n\nSummarize clearly or answer using only the supplied context. If missing, say so."""
                    )
                st.markdown("### ✅ Tool output")
                st.success(output.content if isinstance(output.content, str) else str(output.content))
            except Exception as e:
                st.error(f"Tool execution failed: {e}")

# ------------------------- LEARNING PAGE -------------------------
elif page == "📘 Learn the Concepts":
    st.markdown("## 📘 Learn the project step by step")
    with st.expander("1. What is Agentic AI?", expanded=True):
        st.write("An Agentic AI system receives a request, reasons about the task, and may use tools to complete it. In this project, the agent can answer directly or call one custom text tool.")
    with st.expander("2. What is Groq?"):
        st.write("Groq provides hosted language models that this app accesses through LangChain's ChatGroq integration.")
    with st.expander("3. What is a custom tool?"):
        st.write("A tool is a Python function made available to the agent. Here, `text_assistant` accepts a task and text, then returns a summary or a context-based answer.")
        st.code("""@tool
def text_assistant(task: str, text: str) -> str:
    \"\"\"Summarize text or answer using supplied context.\"\"\"
    response = llm.invoke(prompt)
    return response.content""", language="python")
    with st.expander("4. What does create_agent() do?"):
        st.write("It builds an agent from a model, a list of tools, and instructions. The agent decides whether a tool is needed.")
        st.code("""agent = create_agent(
    model=llm,
    tools=[text_assistant],
    checkpointer=InMemorySaver(),
    system_prompt="You are an Agentic AI assistant."
)""", language="python")
    with st.expander("5. What is conversation memory?"):
        st.write("The checkpointer stores conversation state. A consistent `thread_id` lets successive requests continue the same conversation.")
        st.code('config = {"configurable": {"thread_id": "chat_session_1"}}', language="python")
    with st.expander("6. Project workflow"):
        st.code("User prompt → Groq agent → (optional) text_assistant tool → response → saved thread state", language="text")
    st.markdown("### 🎯 Practice tasks")
    st.markdown("""
    1. Ask the agent to explain Artificial Intelligence.
    2. Paste a paragraph and request a short summary.
    3. Ask a question that can be answered from your pasted context.
    4. Ask a follow-up question in the same chat to observe conversation continuity.
    5. Start a new conversation and compare the behavior.
    """)

# ------------------------- SETUP PAGE -------------------------
elif page == "🛠️ Setup & Help":
    st.markdown("## 🛠️ Setup & Help")
    st.markdown("### Install dependencies")
    st.code("pip install -U streamlit langchain langchain-groq langgraph", language="bash")
    st.markdown("### Run the app")
    st.code("streamlit run app.py", language="bash")
    st.markdown("### Requirements")
    st.code("""streamlit
langchain
langchain-groq
langgraph""", language="text")
    st.markdown("### Common issues")
    st.markdown("""
    - **Model not found:** choose a model currently available in your Groq account.
    - **Authentication error:** check that your API key is valid and active.
    - **Rate limit / quota error:** wait until the quota window resets or check your Groq account limits.
    - **Package import error:** run the install command, then restart the Streamlit app.
    """)
    st.warning("Do not hard-code your API key in app.py or commit it to a public repository.")

st.divider()
st.markdown('<p class="small-note">Built for learning • Streamlit + LangChain + Groq • One custom tool</p>', unsafe_allow_html=True)
