import streamlit as st
from utils.helper import load_css

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Login",
    page_icon="🔐",
    layout="centered"
)
load_css()
# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

.big-title{
    font-size:42px;
    font-weight:bold;
    color:#0F62FE;
    text-align:center;
}

.login-box{

    padding:30px;

    border-radius:15px;

    background-color:#F8F9FA;

    border:1px solid #DDDDDD;

}

</style>
""",unsafe_allow_html=True)

# ==========================================
# Session
# ==========================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in=False

# ==========================================
# Login Credentials
# ==========================================

USERNAME="admin"

PASSWORD="admin123"

# ==========================================
# Already Logged In
# ==========================================

if st.session_state.logged_in:

    st.success("✅ Login Successful")

    st.balloons()

    st.write("Welcome to the Kharada → English Translator")

    st.divider()

    st.subheader("Project Modules")

    st.write("🏠 Home")

    st.write("🌐 Translator")

    st.write("🕘 History")

    st.write("📊 Model Evaluation")

    st.write("📉 Training Graphs")

    st.write("📈 Dataset Statistics")

    st.write("📖 About Project")

    st.divider()

    if st.button("🚪 Logout"):

        st.session_state.logged_in=False

        st.rerun()

# ==========================================
# Login Screen
# ==========================================

else:

    st.markdown(
    '<div class="big-title">🔐 Login</div>',
    unsafe_allow_html=True
    )

    st.write("")

    username=st.text_input(

        "Username"

    )

    password=st.text_input(

        "Password",

        type="password"

    )

    remember=st.checkbox("Remember Me")

    st.write("")

    if st.button("Login",use_container_width=True):

        if username==USERNAME and password==PASSWORD:

            st.session_state.logged_in=True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Username or Password")

    st.divider()

    st.info("""

Demo Credentials

Username : admin

Password : admin123

""")

st.divider()

st.caption("© 2026 Kharada → English Neural Machine Translation")