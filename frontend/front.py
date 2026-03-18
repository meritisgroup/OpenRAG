import streamlit as st
from ecologits import EcoLogits
from dotenv import load_dotenv

from streamlit_.core import init_session_state

load_dotenv()
EcoLogits.init()


@st.fragment
def status_display():
    backend_status = "🟢" if st.session_state.get('backend_connected', True) else "🔴"
    es_status = "🟢" if st.session_state.get('elasticsearch_connected', True) else "🔴"
    st.markdown(f"**Status:** Backend {backend_status} | ES {es_status}")
    
    if st.button("🔄 Refresh Status"):
        st.session_state['force_backend_check'] = True
        st.rerun()


chat = st.Page('streamlit_/pages/1_💬_chat.py', title='Chat')
config = st.Page('streamlit_/pages/2_🧠_configuration.py', title='Configuration')
benchmark = st.Page('streamlit_/pages/3_📚_benchmark.py', title='Benchmark')
rag_maker = st.Page('streamlit_/pages/4_🔧_rag_maker.py', title='Rag Maker')
databases = st.Page('streamlit_/pages/5_🌐_databases.py', title='Databases')
documentation = st.Page('streamlit_/pages/6_📖_documentation.py', title='Documentation')
advanced_configuration = st.Page('streamlit_/pages/7_🛠️_advanced_configuration.py', title='Advanced Configuration')
metadatas = st.Page('streamlit_/pages/8_⚡_metadatas.py', title='Metadatas')

pg = st.navigation([chat, config, benchmark, rag_maker, databases, documentation, advanced_configuration, metadatas])

st.set_page_config(page_title='OpenRAG by Meritis', page_icon='streamlit_/images/symbole_meritis.png', layout='wide')
st.set_option('client.showSidebarNavigation', True)
st.logo('streamlit_/images/logomeritis_horizontal.png', size='large', link='https://meritis.fr/', icon_image='streamlit_/images/logomeritis_horizontal_rvb.png')

init_session_state(st)
with st.sidebar:
    status_display()

pg.run()
