import streamlit as st 
import pandas as pd
from io import StringIO
import json
import torch
from transformers import pipeline # AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util 
import time
import os
os.getenv("HF_TOKEN") 

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device_map()  # 'cpu'

def on_click():
    st.session_state.user_input = "" 

def make_spinner(text = "In progress..."):
    with st.spinner(text):
        yield
    
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


## Define the Reasoning models
rs_models = {
    '(medium speed) original model for general domain: meta-llama/Llama-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct', 
    '(slower speed) original model for general domain: Qwen/Qwen2-1.5B-Instruct': 'Qwen/Qwen2-1.5B-Instruct', 
    '(medium speed) original model for general domain: EpistemeAI/ReasoningCore-1B-r1-0': 'EpistemeAI/ReasoningCore-1B-r1-0',
    '(expected in future) fine-tuned model for medical domain: meta-llama/Llama-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct', 
    '(expected in future) fine-tuned model for medical domain: Qwen/Qwen2-1.5B-Instruct': 'Qwen/Qwen2-1.5B-Instruct',
}
 
## Create the select Reasoning box
selected_rs_model = st.selectbox('Current selected Reasoning model:', list(rs_models.keys())) # or 'Choose a Reasoning Model'
#st.write("Current selection:", selected_rs_model)

## Get the selected Reasoning model
Reasoning_model = rs_models[selected_rs_model]


### Load the Sentence Transformer model ...
#@st.cache_resource
#def load_model():
#    model = SentenceTransformer(SentTrans_model)
#    return model
#model = load_model() 

## Load the Reasoning model as pipeline ...
@st.cache_resource
def load_pipe():
    pipe = pipeline("text-generation", model=Reasoning_model, device_map=device,) # device_map="auto", torch_dtype=torch.bfloat16 
    return pipe 
pipe = load_pipe()


# Semantic search, Compute cosine similarity between INTdesc_embedding and SBS descriptions
INTdesc_embedding = model.encode(INTdesc_input)
SBScorpus_embeddings = model.encode(SBScorpus)

if INTdesc_input is not None and st.button("Analyze the SBS codes", key="run_rs_model"): 
    HF_model_results = util.semantic_search(INTdesc_embedding, SBScorpus_embeddings)
    HF_model_results_sorted = sorted(HF_model_results, key=lambda x: x[1], reverse=True)
    HF_model_results_displayed = HF_model_results_sorted[0:numMAPPINGS_input]

    for i, result in enumerate(HF_model_results_displayed):
        dictA.update({"Score": "%.4f" % result[0]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[0]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[0]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        dictA.update({"Score": "%.4f" % result[1]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[1]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[1]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        dictA.update({"Score": "%.4f" % result[2]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[2]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[2]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        dictA.update({"Score": "%.4f" % result[3]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[3]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[3]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        dictA.update({"Score": "%.4f" % result[4]["score"], "SBS Code": df_SBS.loc[df_SBS["Long_Description"] == SBScorpus[result[4]["corpus_id"]],"SBS_Code_Hyphenated"].values[0], "SBS Description V2.0": SBScorpus[result[4]["corpus_id"]]})
        dfALL = pd.concat([dfALL, pd.DataFrame([dictA])], ignore_index=True)
        
    st.dataframe(data=dfALL, hide_index=True)

    display_format = "ask REASONING MODEL: Which, if any, of the following SBS descriptions corresponds best to " + INTdesc_input +"? " 
    #st.write(display_format)
    question = "Which one, if any, of the following Saudi Billing System descriptions A, B, C, D, or E corresponds best to " + INTdesc_input +"? " 
    shortlist = [SBScorpus[result[0]["corpus_id"]], SBScorpus[result[1]["corpus_id"]], SBScorpus[result[2]["corpus_id"]], SBScorpus[result[3]["corpus_id"]], SBScorpus[result[4]["corpus_id"]]] 
    prompt = question + " " +"A: "+ shortlist[0] + " " +"B: " + shortlist[1] + " " + "C: " + shortlist[2] + " " + "D: " + shortlist[3] + " " + "E: " + shortlist[4]
    #st.write(prompt)

    messages = [
    {"role": "system", "content": "You are a knowledgable AI assistant who always answers truthfully and precisely!"},
    {"role": "user", "content": prompt},
    ]

    status_text = st.empty()
    status_text.warning("It may take several minutes for Reasoning Model to analyze above 5 options and output results below")
    #runningToggle(True)
 
        
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
