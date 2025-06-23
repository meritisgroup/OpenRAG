import streamlit as st
import requests
from streamlit.components.v1 import html
import requests


main_menu = st.sidebar.selectbox("Main topic", ["Rag explanation",
                                                "First launch", 
                                                "Configuration server explanation",
                                                "Benchmark report explanation"])
language = "EN"
if main_menu=="Rag explanation": 
    url = f"https://meritisgroup.github.io/OpenRAG/rags_{language}.html"
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text
    st.html(html)

elif main_menu=="First launch":
    url = f"https://meritisgroup.github.io/OpenRAG/first_launch_{language}.html" 
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text
    st.html(html)
elif main_menu=="Configuration server explanation":
    url = f"https://meritisgroup.github.io/OpenRAG/config_{language}.html"
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text
    st.html(html)
elif main_menu=="Benchmark report explanation":
    url = f"https://meritisgroup.github.io/OpenRAG/report_{language}.html"
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text
    st.html(html)