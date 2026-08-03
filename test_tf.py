import streamlit as st
import sys

st.write("Python executable")
st.code(sys.executable)

st.write("Python version")
st.code(sys.version)

st.write("Importing TensorFlow...")

import tensorflow as tf

st.success("TensorFlow imported successfully!")

st.write(tf.__version__)