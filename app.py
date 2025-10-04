import streamlit as st
import requests
import json
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Claude-inspired design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #f5f5f5;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e5;
        padding-top: 1rem;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    .main-header {
        padding: 1.5rem 0;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        max-width: 48rem;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .message-wrapper {
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.25rem;
        margin-left: 3rem;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.25rem;
        margin-right: 3rem;
    }
    
    .message-label {
        font-weight: 600;
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-content {
        color: #1f2937;
        line-height: 1.6;
        font-size: 0.9375rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #d1d5db;
        padding: 0.875rem 1.25rem;
        font-size: 0.9375rem;
        background-color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #9333ea;
        box-shadow: 0 0 0 3px rgba(147, 51, 234, 0.1);
    }
    
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        background-color: #ffffff;
        color: #374151;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f9fafb;
        border-color: #d1d5db;
    }
    
    .primary-button > button {
        background-color: #9333ea !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .primary-button > button:hover {
        background-color: #7e22ce !important;
    }
    
    .provider-card {
        background: #f9fafb;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .provider-card:hover {
        background: #f3f4f6;
        border-color: #9333ea;
    }
    
    .provider-card-active {
        background: #ede9fe;
        border-color: #9333ea;
    }
    
    .settings-section {
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .settings-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.75rem;
        letter-spacing: 0.05em;
    }
    
    pre {
        background-color: #1e1e1e;
        border-radius: 6px;
        padding: 1rem;
        overflow-x: auto;
    }
    
    code {
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
    }
    
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #6b7280;
        font-size: 0.875rem;
        padding: 0.75rem;
    }
    
    .dot-flashing {
        position: relative;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #9333ea;
        animation: dotFlashing 1s infinite linear;
    }
    
    @keyframes dotFlashing {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .error-message {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        color: #991b1b;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.75rem;
        color: #15803d;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
    
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        color: #6b7280;
    }
    
    .welcome-message h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .capability-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 2rem;
        max-width: 48rem;
        margin-left: auto;
        margin-right: auto;
    }
    
    .capability-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .capability-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .capability-card p {
        font-size: 0.875rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_provider" not in st.session_state:
    st.session_state.current_provider = "ollama"
if "current_model" not in st.session_state:
    st.session_state.current_model = ""
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "groq": "",
        "openai": "",
        "anthropic": ""
    }
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = "http://localhost:11434"
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "ollama_connected" not in st.session_state:
    st.session_state.ollama_connected = False

# Provider configurations
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "icon": "🦙",
        "models": [],  # Will be populated dynamically
        "requires_api_key": False
    },
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "requires_api_key": True
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🤖",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "requires_api_key": True
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🔮",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "requires_api_key": True
    }
}

def fetch_ollama_models():
    """Fetch available models from Ollama"""
    try:
        response = requests.get(
            f"{st.session_state.ollama_host}/api/tags",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            st.session_state.ollama_models = models
            st.session_state.ollama_connected = True
            
            # Set default model if not set
            if not st.session_state.current_model and models:
                st.session_state.current_model = models[0]
            
            return models
        else:
            st.session_state.ollama_connected = False
            return []
    except Exception as e:
        st.session_state.ollama_connected = False
        return []

def call_ollama(model, messages):
    """Call Ollama API"""
    try:
        # Convert messages to prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = requests.post(
            f"{st.session_state.ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', '')
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def call_groq(model, messages):
    """Call Groq API"""
    api_key = st.session_state.api_keys.get("groq", "")
    if not api_key:
        return "Error: Groq API key not set. Please add it in the sidebar."
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"Error: Groq API returned status code {response.status_code}"
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

def call_openai(model, messages):
    """Call OpenAI API"""
    api_key = st.session_state.api_keys.get("openai", "")
    if not api_key:
        return "Error: OpenAI API key not set. Please add it in the sidebar."
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"Error: OpenAI API returned status code {response.status_code}"
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def call_anthropic(model, messages):
    """Call Anthropic API"""
    api_key = st.session_state.api_keys.get("anthropic", "")
    if not api_key:
        return "Error: Anthropic API key not set. Please add it in the sidebar."
    
    try:
        # Convert messages format for Anthropic
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": formatted_messages,
                "max_tokens": 2048
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            return f"Error: Anthropic API returned status code {response.status_code}"
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"

def get_ai_response(provider, model, messages):
    """Route to appropriate API based on provider"""
    if provider == "ollama":
        return call_ollama(model, messages)
    elif provider == "groq":
        return call_groq(model, messages)
    elif provider == "openai":
        return call_openai(model, messages)
    elif provider == "anthropic":
        return call_anthropic(model, messages)
    else:
        return "Error: Unknown provider"

# Sidebar
with st.sidebar:
    st.markdown("### AI Provider")
    
    # Provider selection
    for provider_id, provider_info in PROVIDERS.items():
        is_active = provider_id == st.session_state.current_provider
        if st.button(
            f"{provider_info['icon']} {provider_info['name']}",
            key=f"provider_{provider_id}",
            use_container_width=True
        ):
            st.session_state.current_provider = provider_id
            if provider_id == "ollama":
                # Fetch models when switching to Ollama
                models = fetch_ollama_models()
                if models:
                    st.session_state.current_model = models[0]
            else:
                st.session_state.current_model = provider_info['models'][0]
            st.rerun()
    
    st.markdown("---")
    
    # Model selection
    current_provider_info = PROVIDERS[st.session_state.current_provider]
    
    # For Ollama, fetch models dynamically
    if st.session_state.current_provider == "ollama":
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Model")
        with col2:
            if st.button("🔄", key="refresh_models", help="Refresh model list"):
                fetch_ollama_models()
                st.rerun()
        
        available_models = st.session_state.ollama_models
        
        if not available_models:
            # Try to fetch models
            available_models = fetch_ollama_models()
        
        if available_models:
            if st.session_state.ollama_connected:
                st.markdown(f'<div class="success-message">✓ Connected - {len(available_models)} models found</div>', unsafe_allow_html=True)
            
            st.session_state.current_model = st.selectbox(
                "Select model",
                available_models,
                index=0 if st.session_state.current_model not in available_models 
                      else available_models.index(st.session_state.current_model),
                label_visibility="collapsed"
            )
        else:
            st.error("⚠️ No Ollama models found. Make sure Ollama is running.")
            st.info("Install models with: `ollama pull llama2`")
    else:
        st.markdown("### Model")
        st.session_state.current_model = st.selectbox(
            "Select model",
            current_provider_info['models'],
            index=0 if st.session_state.current_model not in current_provider_info['models'] 
                  else current_provider_info['models'].index(st.session_state.current_model),
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # API Keys section
    st.markdown("### API Configuration")
    
    if st.session_state.current_provider == "ollama":
        new_host = st.text_input(
            "Ollama Host",
            value=st.session_state.ollama_host,
            placeholder="http://localhost:11434"
        )
        if new_host != st.session_state.ollama_host:
            st.session_state.ollama_host = new_host
            # Re-fetch models when host changes
            fetch_ollama_models()
            st.rerun()
    
    if current_provider_info['requires_api_key']:
        api_key = st.text_input(
            f"{current_provider_info['name']} API Key",
            value=st.session_state.api_keys.get(st.session_state.current_provider, ""),
            type="password",
            placeholder="Enter your API key..."
        )
        if api_key:
            st.session_state.api_keys[st.session_state.current_provider] = api_key
    
    st.markdown("---")
    
    # Chat management
    if st.button("🗨️ New Chat", use_container_width=True):
        if len(st.session_state.messages) > 0:
            # Save current conversation
            title = st.session_state.messages[0]['content'][:40] if st.session_state.messages else "New Chat"
            st.session_state.conversations.append({
                "title": title,
                "messages": st.session_state.messages.copy(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        st.session_state.messages = []
        st.rerun()
    
    # Conversation history
    if st.session_state.conversations:
        st.markdown("### Chat History")
        for idx, conv in enumerate(reversed(st.session_state.conversations[-10:])):
            actual_idx = len(st.session_state.conversations) - idx - 1
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"💬 {conv['title'][:25]}...",
                    key=f"conv_{actual_idx}",
                    use_container_width=True
                ):
                    st.session_state.messages = conv['messages'].copy()
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{actual_idx}"):
                    st.session_state.conversations.pop(actual_idx)
                    st.rerun()

# Main content area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-message">
        <h1>How can I help you today?</h1>
        <p>Choose a provider and model from the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="capability-grid">', unsafe_allow_html=True)
    capabilities = [
        {"icon": "💬", "title": "Natural Conversation", "desc": "Chat naturally about any topic"},
        {"icon": "💻", "title": "Code Assistant", "desc": "Write and debug code"},
        {"icon": "📝", "title": "Writing Help", "desc": "Create and edit content"},
        {"icon": "🧠", "title": "Analysis", "desc": "Analyze data and solve problems"}
    ]
    
    cols = st.columns(len(capabilities))
    for col, cap in zip(cols, capabilities):
        with col:
            st.markdown(f"""
            <div class="capability-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{cap['icon']}</div>
                <h3>{cap['title']}</h3>
                <p>{cap['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="message-wrapper">
            <div class="user-message">
                <div class="message-label">
                    <span>👤</span>
                    <span>You</span>
                </div>
                <div class="message-content">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        provider_icon = PROVIDERS[st.session_state.current_provider]['icon']
        st.markdown(f"""
        <div class="message-wrapper">
            <div class="assistant-message">
                <div class="message-label">
                    <span>{provider_icon}</span>
                    <span>Assistant</span>
                </div>
                <div class="message-content">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Type your message here...", key="chat_input")

if prompt:
    # Check if model is selected
    if not st.session_state.current_model:
        st.error("Please select a model from the sidebar first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            # Get AI response
            response = get_ai_response(
                st.session_state.current_provider,
                st.session_state.current_model,
                st.session_state.messages
            )
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
