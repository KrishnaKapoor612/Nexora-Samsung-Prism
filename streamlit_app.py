import streamlit as st
import requests

st.set_page_config(page_title="Samsung Troubleshooting", page_icon="📱", layout="centered")

st.markdown("""
<style>
    .deeplink-btn {
        display: inline-block;
        background-color: #0381fe;
        color: white !important;
        padding: 8px 16px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 10px;
        margin-bottom: 20px;
        transition: background-color 0.2s;
    }
    .deeplink-btn:hover {
        background-color: #026bda;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Smart Troubleshooting")
st.caption("Galaxy AI-Powered Issue Resolution")

query = st.text_input("What seems to be the problem with your Galaxy device?", placeholder="e.g., My phone screen flickers when I open email")

if st.button("Diagnose Issue", type="primary", use_container_width=True):
    if query:
        with st.spinner("Analyzing complaint and searching Samsung knowledge base..."):
            try:
                res = requests.post("http://127.0.0.1:8000/v1/troubleshoot", json={"query": query}, timeout=15)
                
                if res.status_code == 200:
                    data = res.json()
                    meta = data.get("meta", {})
                    
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Response Latency", f"{meta.get('latency_ms', 0)} ms")
                    col2.metric("P95 Cache Hit", "✅ Yes" if meta.get("cache_hit") else "❌ Miss")
                    col3.metric("Generation Model", meta.get("model", "N/A").upper())
                    st.divider()
                    
                    contexts = data.get("response", {}).get("contexts", [])
                    if not contexts:
                        st.warning("⚠️ No trusted Samsung solution found for this issue. (Pipeline safely blocked hallucination)")
                    else:
                        for ctx in contexts:
                            st.header(ctx['title'])
                            st.subheader(f"*{ctx['goal']}*")
                            
                            for i, action in enumerate(ctx.get("actions", [])):
                                with st.container(border=True):
                                    st.markdown(f"### 🔹 Action {i+1}: {action['actionName'].title()}")
                                    st.caption(f"**{action['category'].upper()}** — *{action['description']}*")
                                    
                                    for group in action.get("stepGroups", []):
                                        for step in group.get("steps", []):
                                            st.markdown(f"- {step}")
                                        
                                        dl = group.get("actionableDeeplink")
                                        if dl and dl.get("deeplink"):
                                            st.markdown(f'<a href="{dl["deeplink"]}" class="deeplink-btn" target="_blank">⚙️ Open: {dl.get("description", "Settings")}</a>', unsafe_allow_html=True)
                else:
                    st.error(f"API Error {res.status_code}: {res.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is the FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please enter a query first.")
