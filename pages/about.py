import streamlit as st

st.title("Info") 

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This app maps internal descriptions to the corresponding SBS codes. 
-   This model was developed from xxxxxxxx in HuggingFace, and fine-tuned on xxxxx (xxxx). 
-   The model uses the default pretrained tokenizer.
       """
    )
