import streamlit as st
import time

# Menampilkan animasi balon
st.balloons()

# Menampilkan progress bar dengan nilai awal 10%
st.progress(10)

# Menampilkan spinner selama 10 detik
with st.spinner('Wait for it...'):
    time.sleep(10)
