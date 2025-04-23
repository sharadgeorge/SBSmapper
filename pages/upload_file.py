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

uploaded_file = st.file_uploader(label = "Upload single csv file")
if uploaded_file is not None: 
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    st.success('Your file input is: '+ string_data, icon="✅")


#my_model_results = pipeline("ner", model= "checkpoint-92")
HuggingFace_model_results = pipeline("ner", model = "blaze999/Medical-NER")


createNER_button = st.button("Create SBS codes")

col1, col2, col3 = st.columns([1,1.5,1])
col1.subheader("SBS code V2.0")
col2.subheader("SBS description")
col2.subheader("Similarity score")

if uploaded_file is not None and createNER_button == True: 
    dict1 = {"word": [], "entity": []}
    dict2 = {"word": [], "entity": []}
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #string_data = stringio.read()
    #st.write("Your input is: ", string_data)
    #with col1: 
    #    #st.write(my_model_results(string_data))  
    #    #col1.subheader("myDemo Model")
    #    #for result in my_model_results(string_data): 
    #    #    st.write(result['word'], result['entity'])
    #    #    dict1["word"].append(result['word']), dict1["entity"].append(result['entity'])         
    #    #df1 = pd.DataFrame.from_dict(dict1)
    #    #st.write(df1)
    with col2:
        #st.write(HuggingFace_model_results(string_data))
        #col2.subheader("Hugging Face Model")
        for result in HuggingFace_model_results(string_data):
            st.write(result['word'], result['entity'])     
            dict2["word"].append(result['word']), dict2["entity"].append(result['entity'])
        df2 = pd.DataFrame.from_dict(dict2)
        #st.write(df2)


    cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])
    with c1:
        #csvbutton = download_button(results, "results.csv", "📥 Download .csv")
        csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(df1), file_name= "results.csv", mime='text/csv', key='csv')
    with c2:
        #textbutton = download_button(results, "results.txt", "📥 Download .txt")
        textbutton = st.download_button(label="📥 Download .txt", data=convert_df(df1), file_name= "results.text", mime='text/plain',  key='text')
    with c3:
        #jsonbutton = download_button(results, "results.json", "📥 Download .json")
        jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(df1), file_name= "results.json", mime='application/json',  key='json')

