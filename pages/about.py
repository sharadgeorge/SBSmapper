import streamlit as st

st.title("Info") 

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This app maps internal codes to the corresponding SBS codes. 
-   myDemo model was developed from xxxxxxxx in HuggingFace, and fine-tuned on xxxxx (xxxx). 
-   The model uses the default pretrained tokenizer.
       """
    )
