import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Poor Man's ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp {
        background-color: #343541 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #202123 !important;
        width: 260px !important;
    }

    .chat-message {
        padding: 15px 20px;
        margin: 0;
        display: flex;
        align-items: start;
    }
    
    .user { background-color: #343541; }
    .assistant { background-color: #444654; }
    
    .message-content {
        color: #ECECF1 !important;
        margin-left: 12px;
        margin-right: 40px;
        font-size: 15px;
        line-height: 1.4;
    }

    pre {
        background-color: #1e1e1e !important;
        border-radius: 5px;
        padding: 10px;
        white-space: pre-wrap;
        color: #ECECF1 !important;
    }

    code {
        color: #ECECF1 !important;
    }

    .input-container {
        position: fixed;
        bottom: 0;
        left: 260px;
        right: 0;
        padding: 20px;
        background: #343541;
        border-top: 1px solid #565869;
    }

    .stTextInput input {
        background-color: #40414F !important;
        border-radius: 10px !important;
        border: 1px solid #565869 !important;
        padding: 12px !important;
        color: #ECECF1 !important;
        font-size: 16px !important;
    }

    button {
        color: #ECECF1 !important;
        background-color: #202123 !important;
        border: 1px solid #4a4b53 !important;
    }

    button:hover {
        border-color: #ECECF1 !important;
        background-color: #2A2B32 !important;
    }

    .model-indicator {
        position: fixed;
        bottom: 80px;
        left: 280px;
        padding: 5px 10px;
        background-color: #202123;
        border-radius: 5px;
        color: #ECECF1;
        font-size: 12px;
        z-index: 1000;
    }

    .bottom-space { height: 140px; }
    
    .stMarkdown div { color: #ECECF1 !important; }

    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_model" not in st.session_state:
    st.session_state.current_model = "llama2"
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Models
MODELS = {
    "llama2": {
        "name": "Llama 2",
        "icon": "🦙",
        "description": "General purpose chat model"
    },
    "qwen2.5-coder:3b": {
        "name": "Qwen Coder",
        "icon": "👨‍💻",
        "description": "Specialized in coding tasks"
    }
}

# Sidebar
with st.sidebar:
    if st.button("+ New chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Model")
    for model_id, model_info in MODELS.items():
        button_style = "background-color: #2A2B32;" if model_id == st.session_state.current_model else ""
        if st.button(
            f"{model_info['icon']} {model_info['name']}",
            key=f"model_{model_id}",
            help=model_info['description']
        ):
            st.session_state.current_model = model_id
            st.rerun()

    if st.session_state.conversations:
        st.markdown("### History")
        for idx, conv in enumerate(st.session_state.conversations):
            title = conv.get('title', f'Chat {idx + 1}')
            if st.button(f"💬 {title[:30]}...", key=f"conv_{idx}"):
                st.session_state.messages = conv['messages']
                st.rerun()

# Show current model indicator
current_model = MODELS[st.session_state.current_model]
st.markdown(f"""
<div class="model-indicator">
    {current_model['icon']} Using {current_model['name']}
</div>
""", unsafe_allow_html=True)

# Main chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 20])
            with col1:
                st.button("👍", key=f"good_{len(st.session_state.messages)}")
                st.button("👎", key=f"bad_{len(st.session_state.messages)}")

# Chat input
if prompt := st.chat_input("Message...", key="chat_input"):
    # Prevent message repetition
    if not st.session_state.messages or prompt != st.session_state.messages[-1].get('content', ''):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": st.session_state.current_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                response_data = json.loads(response.text.strip().split('\n')[0])
                if 'response' in response_data:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data['response']
                    })
                    
                    # Save new conversations
                    if len(st.session_state.messages) == 2:
                        st.session_state.conversations.append({
                            "title": prompt[:30],
                            "messages": st.session_state.messages.copy()
                        })
                    
                    # Clear input and rerun
                    st.session_state.user_input = ""
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Bottom spacing
st.markdown('<div class="bottom-space"></div>', unsafe_allow_html=True)