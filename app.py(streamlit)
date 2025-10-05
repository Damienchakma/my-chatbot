import streamlit as st
import requests
import json
from datetime import datetime
import sqlite3
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup
DB_PATH = "chat_history.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  provider TEXT,
                  model TEXT,
                  created_at TIMESTAMP,
                  updated_at TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id INTEGER,
                  role TEXT,
                  content TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS usage_stats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id INTEGER,
                  provider TEXT,
                  model TEXT,
                  input_tokens INTEGER,
                  output_tokens INTEGER,
                  estimated_cost REAL,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')
    
    conn.commit()
    conn.close()

init_database()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
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
if "lmstudio_host" not in st.session_state:
    st.session_state.lmstudio_host = "http://localhost:1234"
if "provider_models" not in st.session_state:
    st.session_state.provider_models = {}
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Enhanced dark mode CSS
def get_theme_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
            border-right: 1px solid #1e3a5f;
        }
        
        section[data-testid="stSidebar"] * {
            color: #e4e4e7 !important;
        }
        
        .user-message {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            border: 1px solid #3b82f6;
            border-radius: 16px;
            padding: 1.25rem;
            margin: 1rem 0 1rem 3rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
            border: 1px solid #4b5563;
            border-radius: 16px;
            padding: 1.25rem;
            margin: 1rem 3rem 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .message-label {
            font-weight: 600;
            font-size: 0.875rem;
            color: #93c5fd;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .message-content {
            color: #f3f4f6;
            line-height: 1.7;
            font-size: 0.9375rem;
        }
        
        .welcome-message {
            text-align: center;
            padding: 3rem 2rem;
        }
        
        .welcome-message h1 {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .welcome-message p {
            color: #9ca3af;
            font-size: 1.125rem;
        }
        
        .capability-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .capability-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
            border-color: #60a5fa;
        }
        
        .capability-card h3 {
            color: #f3f4f6;
            font-size: 1.125rem;
            margin: 0.5rem 0;
        }
        
        .capability-card p {
            color: #9ca3af;
            font-size: 0.875rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.625rem 1.25rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            transform: translateY(-2px);
        }
        
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stTextArea > div > div > textarea {
            background-color: #1e293b !important;
            color: #f3f4f6 !important;
            border: 1px solid #475569 !important;
            border-radius: 8px;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
        
        .success-message {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 0.75rem;
            color: #d1fae5;
            font-size: 0.875rem;
            margin: 0.5rem 0;
        }
        
        div[data-testid="stExpander"] {
            background-color: #1e293b;
            border: 1px solid #475569;
            border-radius: 8px;
        }
        
        .stMetric {
            background-color: #1e293b;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #475569;
        }
        
        code {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }
        
        pre {
            background-color: #0f172a !important;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
    """

st.markdown(get_theme_css(), unsafe_allow_html=True)

# Provider configurations
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "icon": "🦙",
        "requires_api_key": False,
        "fetch_models_func": "fetch_ollama_models"
    },
    "lmstudio": {
        "name": "LM Studio (Local)",
        "icon": "🎯",
        "requires_api_key": False,
        "fetch_models_func": "fetch_lmstudio_models"
    },
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "requires_api_key": True,
        "fetch_models_func": "fetch_groq_models"
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🤖",
        "requires_api_key": True,
        "fetch_models_func": "fetch_openai_models"
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🔮",
        "requires_api_key": True,
        "fetch_models_func": "fetch_anthropic_models"
    }
}

# Dynamic model fetching functions
def fetch_ollama_models():
    """Fetch available models from Ollama"""
    try:
        response = requests.get(f"{st.session_state.ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

def fetch_lmstudio_models():
    """Fetch available models from LM Studio"""
    try:
        response = requests.get(f"{st.session_state.lmstudio_host}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        return []
    except:
        return []

def fetch_groq_models():
    """Fetch available models from Groq"""
    api_key = st.session_state.api_keys.get("groq", "")
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return sorted([model['id'] for model in data.get('data', [])])
        return []
    except:
        return []

def fetch_openai_models():
    """Fetch available models from OpenAI"""
    api_key = st.session_state.api_keys.get("openai", "")
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            # Filter for chat models
            models = [m['id'] for m in data.get('data', []) if 'gpt' in m['id'].lower()]
            return sorted(models, reverse=True)
        return []
    except:
        return []

def fetch_anthropic_models():
    """Return Anthropic models (API doesn't provide model list endpoint)"""
    return [
        "claude-sonnet-4-5-20250929",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307"
    ]

def fetch_models_for_provider(provider):
    """Fetch models for a specific provider"""
    func_name = PROVIDERS[provider].get("fetch_models_func")
    if func_name:
        return globals()[func_name]()
    return []

# Database functions
def save_conversation_to_db(title, provider, model):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    c.execute('''INSERT INTO conversations (title, provider, model, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?)''', (title, provider, model, now, now))
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def save_message_to_db(conversation_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''', (conversation_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def save_usage_stats(conversation_id, provider, model, input_tokens, output_tokens, cost):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO usage_stats (conversation_id, provider, model, input_tokens, output_tokens, estimated_cost, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (conversation_id, provider, model, input_tokens, output_tokens, cost, datetime.now()))
    conn.commit()
    conn.close()

def load_conversations_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, title, provider, model, created_at FROM conversations 
                 ORDER BY updated_at DESC LIMIT 20''')
    conversations = c.fetchall()
    conn.close()
    return conversations

def load_messages_from_db(conversation_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content FROM messages 
                 WHERE conversation_id = ? ORDER BY timestamp''', (conversation_id,))
    messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def get_usage_statistics():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT provider, model, SUM(input_tokens), SUM(output_tokens), SUM(estimated_cost)
                 FROM usage_stats GROUP BY provider, model''')
    stats = c.fetchall()
    c.execute('''SELECT SUM(input_tokens), SUM(output_tokens), SUM(estimated_cost) FROM usage_stats''')
    total_stats = c.fetchone()
    conn.close()
    return stats, total_stats

def delete_conversation_from_db(conversation_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM usage_stats WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
    conn.commit()
    conn.close()

def export_chat_as_json(messages):
    data = {"exported_at": datetime.now().isoformat(), "messages": messages}
    return json.dumps(data, indent=2)

def export_chat_as_markdown(messages):
    md = f"# Chat Export\n\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in messages:
        role = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        md += f"{role}\n\n{msg['content']}\n\n---\n\n"
    return md

# API call functions
def call_ollama(model, messages):
    try:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = requests.post(
            f"{st.session_state.ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": st.session_state.temperature,
                    "num_predict": st.session_state.max_tokens
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get('response', ''), 0, 0
        return f"Error: {response.status_code}", 0, 0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def call_lmstudio(model, messages):
    try:
        response = requests.post(
            f"{st.session_state.lmstudio_host}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens
            },
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            return (data['choices'][0]['message']['content'],
                   usage.get('prompt_tokens', 0),
                   usage.get('completion_tokens', 0))
        return f"Error: {response.status_code}", 0, 0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def call_groq(model, messages):
    api_key = st.session_state.api_keys.get("groq", "")
    if not api_key:
        return "Error: Groq API key not set", 0, 0
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            return (data['choices'][0]['message']['content'],
                   usage.get('prompt_tokens', 0),
                   usage.get('completion_tokens', 0))
        return f"Error: {response.status_code}", 0, 0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def call_openai(model, messages):
    api_key = st.session_state.api_keys.get("openai", "")
    if not api_key:
        return "Error: OpenAI API key not set", 0, 0
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            return (data['choices'][0]['message']['content'],
                   usage.get('prompt_tokens', 0),
                   usage.get('completion_tokens', 0))
        return f"Error: {response.status_code}", 0, 0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def call_anthropic(model, messages):
    api_key = st.session_state.api_keys.get("anthropic", "")
    if not api_key:
        return "Error: Anthropic API key not set", 0, 0
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": st.session_state.max_tokens,
                "temperature": st.session_state.temperature
            },
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            return (data['content'][0]['text'],
                   usage.get('input_tokens', 0),
                   usage.get('output_tokens', 0))
        return f"Error: {response.status_code}", 0, 0
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def get_ai_response(provider, model, messages):
    if provider == "ollama":
        return call_ollama(model, messages)
    elif provider == "lmstudio":
        return call_lmstudio(model, messages)
    elif provider == "groq":
        return call_groq(model, messages)
    elif provider == "openai":
        return call_openai(model, messages)
    elif provider == "anthropic":
        return call_anthropic(model, messages)
    return "Error: Unknown provider", 0, 0

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if st.button("🔧 Advanced" if not st.session_state.show_settings else "🔧 Hide", 
                 use_container_width=True):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()
    
    if st.session_state.show_settings:
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
        st.session_state.max_tokens = st.slider("Max Tokens", 256, 4096, st.session_state.max_tokens, 256)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Provider")
    
    for provider_id, provider_info in PROVIDERS.items():
        if st.button(f"{provider_info['icon']} {provider_info['name']}", 
                    key=f"provider_{provider_id}", use_container_width=True):
            st.session_state.current_provider = provider_id
            models = fetch_models_for_provider(provider_id)
            st.session_state.provider_models[provider_id] = models
            if models:
                st.session_state.current_model = models[0]
            st.rerun()
    
    st.markdown("---")
    
    # Model selection
    current_provider = st.session_state.current_provider
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### 🎯 Model")
    with col2:
        if st.button("🔄", key="refresh", help="Refresh models"):
            models = fetch_models_for_provider(current_provider)
            st.session_state.provider_models[current_provider] = models
            st.rerun()
    
    if current_provider not in st.session_state.provider_models:
        st.session_state.provider_models[current_provider] = fetch_models_for_provider(current_provider)
    
    models = st.session_state.provider_models.get(current_provider, [])
    
    if models:
        st.markdown(f'<div class="success-message">✓ {len(models)} models available</div>', 
                   unsafe_allow_html=True)
        st.session_state.current_model = st.selectbox(
            "Select model",
            models,
            index=0 if st.session_state.current_model not in models else models.index(st.session_state.current_model),
            label_visibility="collapsed"
        )
    else:
        st.error("⚠️ No models found")
    
    st.markdown("---")
    
    # API Configuration
    st.markdown("### 🔑 Configuration")
    
    if current_provider == "ollama":
        new_host = st.text_input("Ollama Host", value=st.session_state.ollama_host)
        if new_host != st.session_state.ollama_host:
            st.session_state.ollama_host = new_host
            st.rerun()
    elif current_provider == "lmstudio":
        new_host = st.text_input("LM Studio Host", value=st.session_state.lmstudio_host)
        if new_host != st.session_state.lmstudio_host:
            st.session_state.lmstudio_host = new_host
            st.rerun()
    
    if PROVIDERS[current_provider]['requires_api_key']:
        api_key = st.text_input(
            f"{PROVIDERS[current_provider]['name']} API Key",
            value=st.session_state.api_keys.get(current_provider, ""),
            type="password"
        )
        if api_key:
            st.session_state.api_keys[current_provider] = api_key
    
    st.markdown("---")
    
    # File Upload
    st.markdown("### 📁 Files")
    uploaded_file = st.file_uploader("Upload", type=['txt', 'pdf', 'md', 'json'], label_visibility="collapsed")
    if uploaded_file:
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        st.session_state.uploaded_files.append({"name": uploaded_file.name, "content": content[:5000]})
        st.success(f"✓ {uploaded_file.name}")
    
    st.markdown("---")
    
    if st.button("🗨️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.uploaded_files = []
        st.rerun()
    
    if st.session_state.messages:
        st.markdown("### 💾 Export")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 JSON", export_chat_as_json(st.session_state.messages),
                             f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                             "application/json", use_container_width=True)
        with col2:
            st.download_button("📝 MD", export_chat_as_markdown(st.session_state.messages),
                             f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                             "text/markdown", use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Usage")
    stats, total = get_usage_statistics()
    if total and total[0]:
        st.metric("Tokens", f"{total[0] + total[1]:,}")
        st.metric("Cost", f"${total[2]:.4f}")
    else:
        st.info("No data yet")
    
    st.markdown("---")
    
    # History
    st.markdown("### 💬 History")
    conversations = load_conversations_from_db()
    if conversations:
        for conv in conversations[:10]:
            conv_id, title, provider, model, created = conv
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"💬 {title[:25]}...", key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.messages = load_messages_from_db(conv_id)
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    delete_conversation_from_db(conv_id)
                    st.rerun()
    else:
        st.info("No history")

# Main content
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-message">
        <h1>How can I help you today?</h1>
        <p>Choose a provider and model to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="capability-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 2rem;">', unsafe_allow_html=True)
    
    capabilities = [
        {"icon": "💬", "title": "Chat", "desc": "Natural conversations"},
        {"icon": "💻", "title": "Code", "desc": "Programming help"},
        {"icon": "📝", "title": "Write", "desc": "Content creation"},
        {"icon": "🧠", "title": "Analyze", "desc": "Problem solving"}
    ]
    
    cols = st.columns(4)
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

if st.session_state.uploaded_files:
    st.info(f"📎 {len(st.session_state.uploaded_files)} file(s) attached")

# Display messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="message-wrapper">
            <div class="user-message">
                <div class="message-label"><span>👤</span><span>You</span></div>
                <div class="message-content">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        icon = PROVIDERS[st.session_state.current_provider]['icon']
        st.markdown(f"""
        <div class="message-wrapper">
            <div class="assistant-message">
                <div class="message-label"><span>{icon}</span><span>Assistant</span></div>
                <div class="message-content">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    if not st.session_state.current_model:
        st.error("Please select a model first!")
    else:
        if not st.session_state.current_conversation_id:
            title = prompt[:50]
            st.session_state.current_conversation_id = save_conversation_to_db(
                title, st.session_state.current_provider, st.session_state.current_model)
        
        context = ""
        if st.session_state.uploaded_files:
            context = "\n\n[Context from files]:\n"
            for file in st.session_state.uploaded_files:
                context += f"\n--- {file['name']} ---\n{file['content']}\n"
        
        full_prompt = context + prompt if context else prompt
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message_to_db(st.session_state.current_conversation_id, "user", prompt)
        
        with st.spinner("Thinking..."):
            response, input_tokens, output_tokens = get_ai_response(
                st.session_state.current_provider,
                st.session_state.current_model,
                st.session_state.messages if not context else 
                st.session_state.messages[:-1] + [{"role": "user", "content": full_prompt}]
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message_to_db(st.session_state.current_conversation_id, "assistant", response)
        
        cost = (input_tokens / 1000) * 0.001 + (output_tokens / 1000) * 0.002
        save_usage_stats(st.session_state.current_conversation_id,
                        st.session_state.current_provider,
                        st.session_state.current_model,
                        input_tokens, output_tokens, cost)
        
        st.rerun()
        
        st.rerun()

