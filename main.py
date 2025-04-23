import streamlit as st

# --- SHARED ON ALL PAGES ---
#st.logo(image=":material/medical_information:")
st.logo("images/medical_information_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png")
st.sidebar.text("Project by JA-RAD")


# --- PAGE SETUP ---
home_page = st.Page(
    page="pages/home.py",
    title="Home",
    icon=":material/home:",
    default=True,)

type_text_page = st.Page(
    page="pages/type_text.py",
    title="type text",
    icon=":material/keyboard:",
    default=False,)

upload_file_page = st.Page(
    page="pages/upload_file.py",
    title="upload file",
    icon=":material/file_upload:",
    default=False,)

about_page = st.Page(
    page="pages/about.py",
    title="About the app",
    icon=":material/info:",
    default=False)


# --- NAVIGATION SETUP ---
#pg = st.navigation(pages=[home_page, type_text_page, upload_file_page, about_page]) # WITHOUT SECTIONS
pg = st.navigation({"Home": [home_page], "Demo": [type_text_page, upload_file_page], "About": [about_page]}) # WITH SECTIONS

pg.run()


