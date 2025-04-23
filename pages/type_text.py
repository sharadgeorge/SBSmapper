import streamlit as st
import pandas as pd
from io import StringIO
import json
from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util

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

INTdesc_input = st.text_input("Type internal description and hit Enter", key="user_input")
st.button("Clear text", on_click=on_click)

numMAPPINGS_input = 5
#numMAPPINGS_input = st.text_input("Type number of mappings and hit Enter", key="user_input_numMAPPINGS")
#st.button("Clear text", on_click=on_click)


model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
#model = SentenceTransformer("all-mpnet-base-v2") # best performance
#model = SentenceTransformers("all-distilroberta-v1")

INTdesc_embedding = model.encode(INTdesc_input)

# Compute cosine similarity between all pairs of SBS descriptions


# Semantic search
SBSdesc_1 = 'Computerised tomography of chest and abdomen'
SBSdesc_2 = 'Computerised tomography of chest and abdomen with intravenous contrast medium'
SBSdesc_3 = 'Computerised tomography of brain and chest'
SBSdesc_4 = 'Computerised tomography of brain, chest and abdomen'
SBSdesc_5 = 'Radiography of cervical spine'
SBSdesc_6 = 'Radiography of thoracic spine'
SBSdesc_7 = 'Radiography of lumbosacral spine'
SBSdesc_8 = 'Radiography of sacrococcygeal spine'
SBSdesc_9 = 'Radiography of spine, 2 regions'
SBSdesc_10 = 'Radiography of spine, 3 regions'

SBScorpus = [SBSdesc_1, SBSdesc_2, SBSdesc_3, SBSdesc_4, SBSdesc_5, SBSdesc_6,SBSdesc_7, SBSdesc_8, SBSdesc_9, SBSdesc_10,]
SBScorpus_embeddings = model.encode(SBScorpus)
#SBScorpus_embeddings = model.encode([SBSdesc_1, SBSdesc_2, SBSdesc_3, SBSdesc_4, SBSdesc_5, SBSdesc_6,SBSdesc_7, SBSdesc_8, SBSdesc_9, SBSdesc_10,])

#my_model_results = pipeline("ner", model= "checkpoint-92")
HF_model_results = util.semantic_search(INTdesc_embedding, SBScorpus_embeddings)
HF_model_results_sorted = sorted(HF_model_results, key=lambda x: x[1], reverse=True)
HF_model_results_displayed = HF_model_results_sorted[0:numMAPPINGS_input]

createSBScodes_button = st.button("Create SBS codes")

col1, col2, col3 = st.columns([1,1,2.5])
col1.subheader("Score")
col2.subheader("SBS code")
col3.subheader("SBS description V2.0")



dictA = {"word": [], "entity": []}
dictB = {"word": [], "entity": []}


#df_SBS = pd.read_csv("SBS_V2_Table.csv", index_col=1, na_values=['NA'], usecols=[3])
df_SBS = pd.read_csv("SBS_V2_Table.csv")
st.write(df_SBS.head(5))

if INTdesc_input is not None and createSBScodes_button == True: 
    #for i, result in enumerate(HF_model_results_displayed):
    for result in HF_model_results_displayed:
        with st.container():
            col1.write("%.4f" % result[0]["score"])
            col2.write("CODE PENDING")
            col3.write(SBScorpus[result[0]["corpus_id"]])
            col1.write("%.4f" % result[1]["score"])
            col2.write("CODE PENDING")
            col3.write(SBScorpus[result[1]["corpus_id"]])
            col1.write("%.4f" % result[2]["score"])
            col2.write("CODE PENDING")
            col3.write(SBScorpus[result[2]["corpus_id"]])
            col1.write("%.4f" % result[3]["score"])
            col2.write("CODE PENDING")
            col3.write(SBScorpus[result[3]["corpus_id"]])
            col1.write("%.4f" % result[4]["score"])
            col2.write("CODE PENDING")
            col3.write(SBScorpus[result[4]["corpus_id"]])


"""    
    #with col1:
    #    #st.write(my_model_results(INTdesc_input))
    #    #col1.subheader("SBS code V2.0")
    #    #for result in HF_model_results_displayed: 
    #    #    st.write(result['word'], result['entity'])
    #    #    dictA["word"].append(result['word']), dictA["entity"].append(result['entity'])
    #    #dfA = pd.DataFrame.from_dict(dictA)
    #    #st.write(dfA)            
    with col2:
        #st.write(HF_model_results)
        #col2.subheader("SBS description V2.0")
        for result in HF_model_results_displayed:
            st.write(SBScorpus[result[0]["corpus_id"]])
            st.write(SBScorpus[result[1]["corpus_id"]])
            st.write(SBScorpus[result[2]["corpus_id"]])
            st.write(SBScorpus[result[3]["corpus_id"]])
            st.write(SBScorpus[result[4]["corpus_id"]])
            #st.write(result['word'], result['entity'])
            #dictB["word"].append(result['word']), dictB["entity"].append(result['entity'])         
        #dfB = pd.DataFrame.from_dict(dictB)
        #st.write(dfB)
    with col3:
        #st.write(HF_model_results)
        #col3.subheader("Similarity score")
        for result in HF_model_results_displayed:
            st.write(result[0]["score"])
            st.write(result[1]["score"])
            st.write(result[2]["score"])
            st.write(result[3]["score"])
            st.write(result[4]["score"])
            #st.write(result['word'], result['entity'])
            #dictB["word"].append(result['word']), dictB["entity"].append(result['entity'])         
        #dfB = pd.DataFrame.from_dict(dictB)
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

"""
