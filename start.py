import streamlit as st

# --- SHARED ON ALL PAGES ---
#st.logo(image="images/menu_book_60dp_75FBFD.png")
st.logo("images/medical_information_24dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png")
st.sidebar.title("SBS V2.0 mapper")
st.sidebar.subheader("(work in progress)")
st.sidebar.text("Demo by JA-RAD")


# --- PAGE SETUP ---
demo_page = st.Page(
    page="pages/demo.py",
    title="Demo",
    icon=":material/home:",
    default=True,)

reasoning_page = st.Page(
    page="pages/analyze.py",
    title="type text (work in progress)",
    icon=":material/keyboard:",
    default=False,)

upload_file_page = st.Page(
    page="pages/upload_file.py",
    title="upload file (page not yet active)",
    icon=":material/file_upload:",
    default=False,)

about_page = st.Page(
    page="pages/about.py",
    title="About the app",
    icon=":material/info:",
    default=False)


# --- NAVIGATION SETUP ---
pg = st.navigation(pages=[demo_page,reasoning_page,]) # WITHOUT SECTIONS
#pg = st.navigation({"Demo": [demo_page], "Work in progess": [reasoning_page, upload_file_page], "About": [about_page]}) # WITH SECTIONS

pg.run()
