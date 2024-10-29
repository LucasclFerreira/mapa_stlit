import streamlit as st

st.logo("menu_icon.png")

painel_page = st.Page(
    "./painel.py",
    title="Painel",
    icon="📊",
    default=True
)

chatbot_page = st.Page(
    "./chat.py",
    title="Climate Risk Assistant",
    icon="✨"
)

pages = {
    "Gráficos": [painel_page],
    "Inteligência Artificial": [chatbot_page]
}

router = st.navigation(pages)

router.run()