import streamlit as st
import pandas as pd
from io import StringIO
import json
import torch
from transformers import pipeline # AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util 

import os
os.getenv("HF_TOKEN")

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device_map()  # 'cpu'

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

INTdesc_input = st.text_input("Type internal description and hit Enter", key="user_input") 

createSBScodes, right_column = st.columns(2)
createSBScodes_clicked = createSBScodes.button("Map to SBS codes", key="user_createSBScodes")
right_column.button("Reset", on_click=on_click)

numMAPPINGS_input = 5
#numMAPPINGS_input = st.text_input("Type number of mappings and hit Enter", key="user_input_numMAPPINGS")
#st.button("Clear text", on_click=on_click)

@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
    return model
#load_model() 

model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
#model = SentenceTransformer('all-mpnet-base-v2') # best performance
#model = SentenceTransformers('all-distilroberta-v1')
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5') 
#model = SentenceTransformer('clips/mfaq')

INTdesc_embedding = model.encode(INTdesc_input)

# Semantic search, Compute cosine similarity between all pairs of SBS descriptions

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

#@st.cache_resource
#def load_model_pipe():
#    pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map=device,) # device_map="auto", torch_dtype=torch.bfloat16 
#    return pipe 
#load_model_pipe()

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device_map=device,) # device_map="auto", torch_dtype=torch.bfloat16 

dictA = {"Score": [], "SBS Code": [], "SBS Description V2.0": []}
dfALL = pd.DataFrame.from_dict(dictA)

if INTdesc_input is not None and createSBScodes_clicked == True: 
    #for i, result in enumerate(HF_model_results_displayed):
    for result in HF_model_results_displayed:
        dictA.update({"Score": "%.4f" % result[i]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[i]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[i]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        #dictA.update({"Score": "%.4f" % result[1]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[1]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[1]["corpus_id"]]})
        #dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        #dictA.update({"Score": "%.4f" % result[2]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[2]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[2]["corpus_id"]]})
        #dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        #dictA.update({"Score": "%.4f" % result[3]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[3]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[3]["corpus_id"]]})
        #dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        #dictA.update({"Score": "%.4f" % result[4]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[4]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[4]["corpus_id"]]})
        #dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        
        st.dataframe(data=dfALL, hide_index=True)

    display_format = "ask REASONING MODEL: Which, if any, of the above SBS descriptions corresponds best to " + INTdesc_input +"? " 
    st.write(display_format)
    question = "Which, if any, of the below Saudi Billing System descriptions corresponds best to " + INTdesc_input +"? " 
    shortlist = [SBScorpus[result[0]["corpus_id"]], SBScorpus[result[1]["corpus_id"]], SBScorpus[result[2]["corpus_id"]], SBScorpus[result[3]["corpus_id"]], SBScorpus[result[4]["corpus_id"]]] 
    prompt = [question + " " + shortlist[0] + " " + shortlist[1] + " " + shortlist[2] + " " + shortlist[3] + " " + shortlist[4]]
    #st.write(prompt)
    
    messages = [
    {"role": "system", "content": "You are a knowledgable AI assistant who always answers truthfully and precisely!"},
    {"role": "user", "content": prompt},
        ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    st.write(outputs[0]["generated_text"][-1]["content"])
    
    bs, b1, b2, b3, bLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])
    with b1:
        #csvbutton = download_button(results, "results.csv", "📥 Download .csv")
        csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(dfALL), file_name= "results.csv", mime='text/csv', key='csv_b')
    with b2:
        #textbutton = download_button(results, "results.txt", "📥 Download .txt")
        textbutton = st.download_button(label="📥 Download .txt", data=convert_df(dfALL), file_name= "results.text", mime='text/plain',  key='text_b')
    with b3:
        #jsonbutton = download_button(results, "results.json", "📥 Download .json")
        jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(dfALL), file_name= "results.json", mime='application/json',  key='json_b') 
