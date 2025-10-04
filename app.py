import streamlit as st
import requests
import json
from datetime import datetime
import os
import sqlite3
from pathlib import Path
import base64
import io

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
    
    # Conversations table
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  provider TEXT,
                  model TEXT,
                  created_at TIMESTAMP,
                  updated_at TIMESTAMP)''')
    
    # Messages table
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id INTEGER,
                  role TEXT,
                  content TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')
    
    # Usage statistics table
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

# Initialize database on startup
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
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "ollama_connected" not in st.session_state:
    st.session_state.ollama_connected = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# Theme CSS
def get_theme_css():
    if st.session_state.theme == "dark":
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            .stApp {
                background-color: #1a1a1a;
            }
            
            section[data-testid="stSidebar"] {
                background-color: #2d2d2d;
                border-right: 1px solid #404040;
                padding-top: 1rem;
            }
            
            .user-message {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 1.25rem;
                margin-left: 3rem;
            }
            
            .assistant-message {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 1.25rem;
                margin-right: 3rem;
            }
            
            .message-label {
                font-weight: 600;
                font-size: 0.875rem;
                color: #a0a0a0;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .message-content {
                color: #e5e5e5;
                line-height: 1.6;
                font-size: 0.9375rem;
            }
            
            .capability-card {
                background: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
            }
            
            .capability-card h3 {
                color: #e5e5e5;
            }
            
            .capability-card p {
                color: #a0a0a0;
            }
            
            .welcome-message h1 {
                color: #e5e5e5;
            }
            
            .stTextInput > div > div > input {
                background-color: #2d2d2d;
                color: #e5e5e5;
                border: 1px solid #404040;
            }
            
            .stSelectbox > div > div {
                background-color: #2d2d2d;
                color: #e5e5e5;
            }
        </style>
        """
    else:
        return """
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
            
            .capability-card {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
            }
            
            .capability-card h3 {
                color: #1f2937;
            }
            
            .capability-card p {
                color: #6b7280;
            }
            
            .welcome-message h1 {
                color: #1f2937;
            }
        </style>
        """

# Common CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)
st.markdown("""
<style>
    .message-wrapper {
        margin-bottom: 1.5rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-container {
        max-width: 48rem;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .stats-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
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
    
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        background-color: #ffffff;
        color: #374151;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
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
        color: #e5e5e5;
    }
</style>
""", unsafe_allow_html=True)

# Provider configurations
PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "icon": "🦙",
        "models": [],
        "requires_api_key": False,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0
    },
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "requires_api_key": True,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🤖",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "requires_api_key": True,
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🔮",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "requires_api_key": True,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015
    }
}

# Database functions
def save_conversation_to_db(title, provider, model):
    """Save conversation to database"""
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
    """Save message to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT INTO messages (conversation_id, role, content, timestamp)
                 VALUES (?, ?, ?, ?)''', (conversation_id, role, content, datetime.now()))
    
    conn.commit()
    conn.close()

def save_usage_stats(conversation_id, provider, model, input_tokens, output_tokens, cost):
    """Save usage statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''INSERT INTO usage_stats (conversation_id, provider, model, input_tokens, output_tokens, estimated_cost, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (conversation_id, provider, model, input_tokens, output_tokens, cost, datetime.now()))
    
    conn.commit()
    conn.close()

def load_conversations_from_db():
    """Load all conversations from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, title, provider, model, created_at FROM conversations 
                 ORDER BY updated_at DESC LIMIT 20''')
    
    conversations = c.fetchall()
    conn.close()
    return conversations

def load_messages_from_db(conversation_id):
    """Load messages for a conversation"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT role, content FROM messages 
                 WHERE conversation_id = ? ORDER BY timestamp''', (conversation_id,))
    
    messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def get_usage_statistics():
    """Get usage statistics"""
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
    """Delete conversation and its messages"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM usage_stats WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
    
    conn.commit()
    conn.close()

def export_chat_as_json(messages):
    """Export chat as JSON"""
    data = {
        "exported_at": datetime.now().isoformat(),
        "messages": messages
    }
    return json.dumps(data, indent=2)

def export_chat_as_markdown(messages):
    """Export chat as Markdown"""
    md = f"# Chat Export\n\n"
    md += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    
    for msg in messages:
        role = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        md += f"{role}\n\n{msg['content']}\n\n---\n\n"
    
    return md

# API Functions
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
            
            if not st.session_state.current_model and models:
                st.session_state.current_model = models[0]
            
            return models
        else:
            st.session_state.ollama_connected = False
            return []
    except Exception as e:
        st.session_state.ollama_connected = False
        return []

def estimate_tokens(text):
    """Rough estimation of tokens (1 token ≈ 4 characters)"""
    return len(text) // 4

def calculate_cost(provider, input_tokens, output_tokens):
    """Calculate estimated cost"""
    provider_info = PROVIDERS.get(provider, {})
    input_cost = (input_tokens / 1000) * provider_info.get("cost_per_1k_input", 0)
    output_cost = (output_tokens / 1000) * provider_info.get("cost_per_1k_output", 0)
    return input_cost + output_cost

def call_ollama(model, messages):
    """Call Ollama API"""
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
            data = response.json()
            return data.get('response', ''), 0, 0
        else:
            return f"Error: Ollama returned status code {response.status_code}", 0, 0
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}", 0, 0

def call_groq(model, messages):
    """Call Groq API"""
    api_key = st.session_state.api_keys.get("groq", "")
    if not api_key:
        return "Error: Groq API key not set. Please add it in the sidebar.", 0, 0
    
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
        else:
            return f"Error: Groq API returned status code {response.status_code}", 0, 0
    except Exception as e:
        return f"Error calling Groq API: {str(e)}", 0, 0

def call_openai(model, messages):
    """Call OpenAI API"""
    api_key = st.session_state.api_keys.get("openai", "")
    if not api_key:
        return "Error: OpenAI API key not set.", 0, 0
    
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
        else:
            return f"Error: OpenAI API returned status code {response.status_code}", 0, 0
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}", 0, 0

def call_anthropic(model, messages):
    """Call Anthropic API"""
    api_key = st.session_state.api_keys.get("anthropic", "")
    if not api_key:
        return "Error: Anthropic API key not set.", 0, 0
    
    try:
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
        else:
            return f"Error: Anthropic API returned status code {response.status_code}", 0, 0
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}", 0, 0

def get_ai_response(provider, model, messages):
    """Route to appropriate API"""
    if provider == "ollama":
        return call_ollama(model, messages)
    elif provider == "groq":
        return call_groq(model, messages)
    elif provider == "openai":
        return call_openai(model, messages)
    elif provider == "anthropic":
        return call_anthropic(model, messages)
    else:
        return "Error: Unknown provider", 0, 0

# Sidebar
with st.sidebar:
    # Theme toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### Settings")
    with col2:
        if st.button("🌓", help="Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    # Advanced settings toggle
    if st.button("⚙️ Advanced Settings" if not st.session_state.show_settings else "⚙️ Hide Settings", 
                 use_container_width=True):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()
    
    if st.session_state.show_settings:
        st.markdown("#### Model Parameters")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random"
        )
        
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=256,
            help="Maximum length of response"
        )
    
    st.markdown("---")
    
    # Provider selection
    st.markdown("### AI Provider")
    
    for provider_id, provider_info in PROVIDERS.items():
        if st.button(
            f"{provider_info['icon']} {provider_info['name']}",
            key=f"provider_{provider_id}",
            use_container_width=True
        ):
            st.session_state.current_provider = provider_id
            if provider_id == "ollama":
                models = fetch_ollama_models()
                if models:
                    st.session_state.current_model = models[0]
            else:
                st.session_state.current_model = provider_info['models'][0]
            st.rerun()
    
    st.markdown("---")
    
    # Model selection
    current_provider_info = PROVIDERS[st.session_state.current_provider]
    
    if st.session_state.current_provider == "ollama":
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Model")
        with col2:
            if st.button("🔄", key="refresh_models", help="Refresh"):
                fetch_ollama_models()
                st.rerun()
        
        available_models = st.session_state.ollama_models
        
        if not available_models:
            available_models = fetch_ollama_models()
        
        if available_models:
            if st.session_state.ollama_connected:
                st.markdown(f'<div class="success-message">✓ {len(available_models)} models</div>', 
                          unsafe_allow_html=True)
            
            st.session_state.current_model = st.selectbox(
                "Select model",
                available_models,
                index=0 if st.session_state.current_model not in available_models 
                      else available_models.index(st.session_state.current_model),
                label_visibility="collapsed"
            )
        else:
            st.error("⚠️ No models found")
            st.info("Install: `ollama pull llama2`")
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
    
    # API Configuration
    st.markdown("### API Configuration")
    
    if st.session_state.current_provider == "ollama":
        new_host = st.text_input(
            "Ollama Host",
            value=st.session_state.ollama_host,
            placeholder="http://localhost:11434"
        )
        if new_host != st.session_state.ollama_host:
            st.session_state.ollama_host = new_host
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
    
    # File Upload
    st.markdown("### 📁 Upload Files")
    uploaded_file = st.file_uploader(
        "Upload document",
        type=['txt', 'pdf', 'md', 'json'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        st.session_state.uploaded_files.append({
            "name": uploaded_file.name,
            "content": file_content[:5000]  # Limit content
        })
        st.success(f"✓ Loaded {uploaded_file.name}")
    
    st.markdown("---")
    
    # Chat management
    if st.button("🗨️ New Chat", use_container_width=True):
        if len(st.session_state.messages) > 0 and st.session_state.current_conversation_id:
            pass  # Already saved
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.uploaded_files = []
        st.rerun()
    
    # Export options
    if st.session_state.messages:
        st.markdown("### 💾 Export Chat")
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = export_chat_as_json(st.session_state.messages)
            st.download_button(
                "📄 JSON",
                data=json_data,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            md_data = export_chat_as_markdown(st.session_state.messages)
            st.download_button(
                "📝 MD",
                data=md_data,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Usage Stats")
    stats, total = get_usage_statistics()
    
    if total and total[0]:
        st.metric("Total Tokens", f"{total[0] + total[1]:,}")
        st.metric("Estimated Cost", f"${total[2]:.4f}")
        
        with st.expander("📈 Details"):
            for stat in stats:
                provider, model, inp, out, cost = stat
                st.text(f"{provider}/{model}")
                st.text(f"  In: {inp:,} | Out: {out:,}")
                st.text(f"  Cost: ${cost:.4f}")
    else:
        st.info("No usage data yet")
    
    st.markdown("---")
    
    # Conversation history
    st.markdown("### 💬 Chat History")
    conversations = load_conversations_from_db()
    
    if conversations:
        for conv in conversations[:10]:
            conv_id, title, provider, model, created = conv
            col1, col2 = st.columns([5, 1])
            
            with col1:
                if st.button(
                    f"💬 {title[:25]}...",
                    key=f"conv_{conv_id}",
                    use_container_width=True
                ):
                    st.session_state.messages = load_messages_from_db(conv_id)
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    delete_conversation_from_db(conv_id)
                    st.rerun()
    else:
        st.info("No saved chats yet")

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

# Display uploaded files context
if st.session_state.uploaded_files:
    st.info(f"📎 {len(st.session_state.uploaded_files)} file(s) attached to context")

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
    if not st.session_state.current_model:
        st.error("Please select a model from the sidebar first!")
    else:
        # Create new conversation if needed
        if not st.session_state.current_conversation_id:
            title = prompt[:50]
            st.session_state.current_conversation_id = save_conversation_to_db(
                title,
                st.session_state.current_provider,
                st.session_state.current_model
            )
        
        # Add file context if available
        context = ""
        if st.session_state.uploaded_files:
            context = "\n\n[Context from uploaded files]:\n"
            for file in st.session_state.uploaded_files:
                context += f"\n--- {file['name']} ---\n{file['content']}\n"
        
        full_prompt = context + prompt if context else prompt
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message_to_db(st.session_state.current_conversation_id, "user", prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response, input_tokens, output_tokens = get_ai_response(
                st.session_state.current_provider,
                st.session_state.current_model,
                st.session_state.messages if not context else 
                st.session_state.messages[:-1] + [{"role": "user", "content": full_prompt}]
            )
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message_to_db(st.session_state.current_conversation_id, "assistant", response)
        
        # Save usage statistics
        cost = calculate_cost(st.session_state.current_provider, input_tokens, output_tokens)
        save_usage_stats(
            st.session_state.current_conversation_id,
            st.session_state.current_provider,
            st.session_state.current_model,
            input_tokens,
            output_tokens,
            cost
        )
        
        st.rerun()
