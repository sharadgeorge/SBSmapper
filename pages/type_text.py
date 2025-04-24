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


#model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
model = SentenceTransformer("all-mpnet-base-v2") # best performance
#model = SentenceTransformers("all-distilroberta-v1")
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

INTdesc_embedding = model.encode(INTdesc_input)

# Compute cosine similarity between all pairs of SBS descriptions


# Semantic search
#df_SBS = pd.read_csv("SBS_V2_Table.csv", index_col="SBS_Code", usecols=["Long_Description"]) # na_values=['NA']
#df_SBS = pd.read_csv("SBS_V2_Table.csv", usecols=["SBS_Code_Hyphenated","Long_Description"]) 
from_line = 7727 # Imaging services chapter start, adjust as needed
to_line = 8239 # Imaging services chapter end, adjust as needed
nrows = to_line - from_line + 1
skiprows = list(range(1,from_line - 1))
df_SBS = pd.read_csv("SBS_V2_Table.csv", header=0, skip_blank_lines=False, skiprows=skiprows, nrows=nrows)
#st.write(df_SBS.head(5))

SBScorpus = df_SBS['Long_Description'].values.tolist()
SBScorpus_embeddings = model.encode(SBScorpus)

#my_model_results = pipeline("ner", model= "checkpoint-92")
HF_model_results = util.semantic_search(INTdesc_embedding, SBScorpus_embeddings)
HF_model_results_sorted = sorted(HF_model_results, key=lambda x: x[1], reverse=True)
HF_model_results_displayed = HF_model_results_sorted[0:numMAPPINGS_input]

createSBScodes_button = st.button("Create SBS codes")

col1, col2, col3 = st.columns([1,1,2.5])
col1.subheader("Score")
col2.subheader("SBS code")
col3.subheader("SBS description V2.0")

dictA = {"Score": [], "SBS Code": [], "SBS Description V2.0": []}

if INTdesc_input is not None and createSBScodes_button == True: 
    #for i, result in enumerate(HF_model_results_displayed):
    for result in HF_model_results_displayed:
        with st.container():
            col1.write("%.4f" % result[0]["score"])
            col2.write(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[0]["corpus_id"]],"SBS_Code_Hyphenated"].values[0])
            col3.write(SBScorpus[result[0]["corpus_id"]])
            dictA["Score"].append("%.4f" % result[0]["score"]), dictA["SBS Code"].append(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[0]["corpus_id"]],"SBS_Code_Hyphenated"].values[0]), dictA["SBS Description V2.0"].append(SBScorpus[result[0]["corpus_id"]])
            
            col1.write("%.4f" % result[1]["score"])
            col2.write(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[1]["corpus_id"]],"SBS_Code_Hyphenated"].values[0])
            col3.write(SBScorpus[result[1]["corpus_id"]])
            dictA["Score"].append("%.4f" % result[1]["score"]), dictA["SBS Code"].append(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[1]["corpus_id"]],"SBS_Code_Hyphenated"].values[0]), dictA["SBS Description V2.0"].append(SBScorpus[result[1]["corpus_id"]])
            
            col1.write("%.4f" % result[2]["score"])
            col2.write(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[2]["corpus_id"]],"SBS_Code_Hyphenated"].values[0])
            col3.write(SBScorpus[result[2]["corpus_id"]])
            dictA["Score"].append("%.4f" % result[2]["score"]), dictA["SBS Code"].append(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[2]["corpus_id"]],"SBS_Code_Hyphenated"].values[0]), dictA["SBS Description V2.0"].append(SBScorpus[result[2]["corpus_id"]])
            
            col1.write("%.4f" % result[3]["score"])
            col2.write(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[3]["corpus_id"]],"SBS_Code_Hyphenated"].values[0])
            col3.write(SBScorpus[result[3]["corpus_id"]])
            dictA["Score"].append("%.4f" % result[3]["score"]), dictA["SBS Code"].append(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[3]["corpus_id"]],"SBS_Code_Hyphenated"].values[0]), dictA["SBS Description V2.0"].append(SBScorpus[result[3]["corpus_id"]])
            
            col1.write("%.4f" % result[4]["score"])
            col2.write(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[4]["corpus_id"]],"SBS_Code_Hyphenated"].values[0])
            col3.write(SBScorpus[result[4]["corpus_id"]])
            dictA["Score"].append("%.4f" % result[4]["score"]), dictA["SBS Code"].append(df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[4]["corpus_id"]],"SBS_Code_Hyphenated"].values[0]), dictA["SBS Description V2.0"].append(SBScorpus[result[4]["corpus_id"]])
                        
            dfA = pd.DataFrame.from_dict(dictA) 
            
     
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

