import streamlit as st
from utils.helper import load_css

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Login",
    page_icon="🔐",
    layout="centered"
)

# Load external CSS
load_css()


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
        .big-title {
            font-size: 42px;
            font-weight: bold;
            color: #0F62FE;
            text-align: center;
            margin-bottom: 20px;
        }

        .login-box {
            padding: 30px;
            border-radius: 15px;
            background-color: #F8F9FA;
            border: 1px solid #DDDDDD;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Session State
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ============================================================
# Login Credentials
# ============================================================

USERNAME = "admin"
PASSWORD = "admin123"


# ============================================================
# Logged-In Screen
# ============================================================

if st.session_state.logged_in:

    st.markdown(
        '<div class="big-title">🔐 Login Successful</div>',
        unsafe_allow_html=True
    )

    st.success("Welcome! You have successfully logged in.")
    st.balloons()

    st.write(
        "Welcome to the **Kharada → English Neural Machine Translation System**."
    )

    st.divider()

    # --------------------------------------------------------
    # Project Modules
    # --------------------------------------------------------

    st.subheader("📚 Project Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.write("🏠 **Home**")
        st.write("🌐 **Translator**")
        st.write("🕘 **History**")
        st.write("📊 **Model Evaluation**")

    with col2:
        st.write("📉 **Training Graphs**")
        st.write("📈 **Dataset Statistics**")
        st.write("📖 **About Project**")

    st.divider()

    # --------------------------------------------------------
    # Logout
    # --------------------------------------------------------

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


# ============================================================
# Login Screen
# ============================================================

else:

    st.markdown(
        '<div class="big-title">🔐 Welcome Back!</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "Login to access the Kharada → English Translator"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # Login Form
    # --------------------------------------------------------

    username = st.text_input(
        "👤 Username",
        placeholder="Enter your username"
    )

    password = st.text_input(
        "🔑 Password",
        type="password",
        placeholder="Enter your password"
    )

    remember = st.checkbox("Remember Me")

    st.write("")

    # --------------------------------------------------------
    # Login Button
    # --------------------------------------------------------

    if st.button(
        "🔓 Login",
        use_container_width=True,
        type="primary"
    ):

        if username == USERNAME and password == PASSWORD:

            st.session_state.logged_in = True

            st.success("✅ Login Successful!")
            st.rerun()

        else:

            st.error("❌ Invalid Username or Password")


    # ========================================================
    # Demo Credentials
    # ========================================================

    st.divider()

    st.info(
        """
        **🔑 Demo Credentials**

        **Username:** `admin`

        **Password:** `admin123`
        """
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "© 2026 Kharada → English Neural Machine Translation"
)