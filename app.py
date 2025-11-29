import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import date
from rag_pipeline import build_rag_chain
from planner import generate_onboarding_plan
from logger import log_interaction, log_tasks, INTERACTIONS_PATH, TASKS_PATH
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="Employee Onboarding Copilot",
    page_icon="🧭",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 3rem; padding-bottom: 1.5rem; }
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(148,163,184,0.35);
}
h1 { font-size: 2.6rem; margin-bottom: 1.2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    return build_rag_chain()

rag_chain = load_components()

tab_plan, tab_chat, tab_analytics = st.tabs(
    ["🗓️ Plan Onboarding", "💬 Copilot Chat", "📊 Analytics"]
)

# PLAN TAB
with tab_plan:
    st.title("Onboarding Planner")

    col_info, col_plan = st.columns(2)

    with col_info:
        st.markdown("### New hire details")
        user_name = st.text_input("New hire name")
        role = st.text_input("Role (e.g., Backend Developer, HR Executive)")
        start_date = st.date_input("Start date")
        generate_btn = st.button("Generate 2‑week onboarding plan")

    with col_plan:
        st.markdown("### Generated plan")
        if generate_btn and user_name and role and start_date:
            start_date_str = start_date.isoformat()
            tasks = generate_onboarding_plan(user_name, role, start_date_str)
            if tasks:
                log_tasks(user_name, role, start_date_str, tasks)
                st.success("Onboarding plan generated and saved.")
                df_tasks = pd.DataFrame(tasks)
                st.dataframe(df_tasks, use_container_width=True)
            else:
                st.warning("Could not parse tasks from LLM response. Try again.")
        elif generate_btn:
            st.warning("Please fill name, role, and start date.")

# CHAT TAB
with tab_chat:
    st.title("Employee Onboarding Copilot")

    with st.sidebar:
        st.markdown("### 🧭 Onboarding Copilot")
        st.write("Ask anything about your first weeks, tools, policies, and trainings.")
        if st.button("🔁 Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_name_chat = st.text_input("Your name (for logging)", key="chat_name")
    role_chat = st.text_input("Your role", key="chat_role")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask an onboarding or policy question...")

    if prompt:
        user_msg = {"role": "user", "content": prompt, "avatar": "🧑‍💻"}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Looking into your onboarding docs..."):
                answer = rag_chain(prompt)
                st.markdown(answer)

        assistant_msg = {"role": "assistant", "content": answer, "avatar": "🤖"}
        st.session_state.messages.append(assistant_msg)

        category = "onboarding"
        if user_name_chat or role_chat:
            log_interaction(
                user_name=user_name_chat or "anonymous",
                role=role_chat or "unknown",
                question=prompt,
                answer=answer,
                category=category,
            )

# ANALYTICS TAB
with tab_analytics:
    st.title("Onboarding Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tasks overview")
        if os.path.exists(TASKS_PATH):
            df_t = pd.read_csv(TASKS_PATH)
            total_tasks = len(df_t)
            pending = (df_t["status"] == "pending").sum()
            done = (df_t["status"] == "done").sum()
            st.metric("Total tasks", total_tasks)
            st.metric("Pending", pending)
            st.metric("Completed", done)
            st.bar_chart(df_t["type"].value_counts())
        else:
            st.info("No tasks generated yet.")

    with col2:
        st.subheader("Interactions overview")
        if os.path.exists(INTERACTIONS_PATH):
            df_i = pd.read_csv(INTERACTIONS_PATH)
            st.metric("Total questions", len(df_i))
            st.dataframe(
                df_i.sort_values("timestamp", ascending=False).head(15),
                use_container_width=True,
            )
        else:
            st.info("No questions logged yet.")
