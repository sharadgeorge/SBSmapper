import streamlit as st
import pandas as pd
from io import StringIO
import json
from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForTokenClassification

def on_click():
    st.session_state.user_input = ""

#@st.cache
def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')

#@st.cache
def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    #st.json(json_string, expanded=True)
    return json_string

#st.title("📘SBS mapper")

text_input = st.text_input("Type internal description and hit Enter", key="user_input")
st.button("Clear text", on_click=on_click)

#my_model_results = pipeline("ner", model= "checkpoint-92")
HuggingFace_model_results = pipeline("ner", model = "blaze999/Medical-NER")

createNER_button = st.button("Create SBS codes")

col1, col2, col3 = st.columns([1,1.5,1])
col1.subheader("SBS code V2.0")
col2.subheader("SBS description)
col2.subheader("Similarity score")

dictA = {"word": [], "entity": []}
dictB = {"word": [], "entity": []}

if text_input is not None and createNER_button == True: 

    with col1:
        #st.write(my_model_results(text_input))
        #col1.subheader("myDemo Model")
        #for result in my_model_results(text_input): 
        #    st.write(result['word'], result['entity'])
        #    dictA["word"].append(result['word']), dictA["entity"].append(result['entity'])
        #dfA = pd.DataFrame.from_dict(dictA)
        #st.write(dfA)            
    with col2:
        st.write(HuggingFace_model_results(text_input))
        col2.subheader("Hugging Face Model")
        for result in HuggingFace_model_results(text_input):
            st.write(result['word'], result['entity'])
            dictB["word"].append(result['word']), dictB["entity"].append(result['entity'])         
        dfB = pd.DataFrame.from_dict(dictB)
        #st.write(dfB)
     
    bs, b1, b2, b3, bLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])
    with b1:
        #csvbutton = download_button(results, "results.csv", "📥 Download .csv")
        csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(dfA), file_name= "results.csv", mime='text/csv', key='csv_b')
    with b2:
        #textbutton = download_button(results, "results.txt", "📥 Download .txt")
        textbutton = st.download_button(label="📥 Download .txt", data=convert_df(dfA), file_name= "results.text", mime='text/plain',  key='text_b')
    with b3:
        #jsonbutton = download_button(results, "results.json", "📥 Download .json")
        jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(dfA), file_name= "results.json", mime='application/json',  key='json_b')


