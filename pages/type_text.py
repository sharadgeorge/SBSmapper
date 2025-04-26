import streamlit as st
import pandas as pd
from io import StringIO
import json
import torch
from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util 
#import lmdeploy
#import turbomind as tm 

from huggingface_hub import login
#import os
#access_token = os.environ.get('HF_TOKEN')
login(token = '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')

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

createSBScodes, right_column = st.columns(2)
createSBScodes_clicked = createSBScodes.button("Create SBS codes", key="user_createSBScodes")
right_column.button("Reset", on_click=on_click)

numMAPPINGS_input = 5
#numMAPPINGS_input = st.text_input("Type number of mappings and hit Enter", key="user_input_numMAPPINGS")
#st.button("Clear text", on_click=on_click)


model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
#model = SentenceTransformer('all-mpnet-base-v2') # best performance
#model = SentenceTransformers('all-distilroberta-v1')
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5') 
#model = SentenceTransformer('clips/mfaq')

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

#qa_model = pipeline("question-answering", model= "distilbert_uncased_qa")
#rs_model = pipeline("text-generation", model="EpistemeAI/OpenReasoner-Llama-3.2-3B-rs1.01", torch_dtype=torch.bfloat16, device_map="auto")
#reasoning_model = "internlm/internlm3-8b-instruct"
#tokenizer = AutoTokenizer.from_pretrained("nirajandhakal/LLaMA3-Reasoning")
#model = AutoModelForCausalLM.from_pretrained("nirajandhakal/LLaMA3-Reasoning") 
#pipe = pipeline("text-generation", model="nirajandhakal/LLaMA3-Reasoning", truncation=True)
#model_id = "EpistemeAI/Reasoning-Llama-3.1-CoT-RE1-NMT-V2"
#pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto",)
#pipe = pipeline("text-generation", model="EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2") 
#model_id = "meta-llama/Llama-3.2-1B" 
#pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")
#model_id = "meta-llama/Llama-3.2-1B-Instruct"
#pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto",)

model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
pipeline = transformers.pipeline(
   "text-generation",
   model=model_id,
   tokenizer=tokenizer,
   max_new_tokens=32768,
   temperature=0.6,
   top_p=0.95,
   **model_kwargs
)
# Thinking can be "on" or "off"
thinking = "on"

col1, col2, col3 = st.columns([1,1,2.5])
col1.subheader("Score")
col2.subheader("SBS code")
col3.subheader("SBS description V2.0")

dictA = {"Score": [], "SBS Code": [], "SBS Description V2.0": []}

if INTdesc_input is not None and createSBScodes_clicked == True: 
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

    question = "Which, if any, of the following Saudi Billing System descriptions corresponds best to " + INTdesc_input +"? " 
    shortlist = [SBScorpus[result[0]["corpus_id"]], SBScorpus[result[1]["corpus_id"]], SBScorpus[result[2]["corpus_id"]], SBScorpus[result[3]["corpus_id"]], SBScorpus[result[4]["corpus_id"]]] 
    prompt = [question + " " + shortlist[0] + " " + shortlist[1] + " " + shortlist[2] + " " + shortlist[3] + " " + shortlist[4]]
    st.write(prompt)
    #st.write(qa_model(question = question, context = shortlist[0] + " " + shortlist[1] + " " + shortlist[2] + " " + shortlist[3] + " " + shortlist[4]])
    #st.write(rs_model(prompt))
    #lmdeploy.pipeline(reasoning_model)(prompt)
    #generated_text = pipe(prompt, max_length=200)
    #st.write(generated_text[0]) #['generated_text'])
    #messages = [
    #{"role": "system", "content": "You are a powerful AI math assistant"},
    #{"role": "user", "content": "Given the quadratic function $f(x)=ax^{2}+bx+c$ with its derivative $f′(x)$, where $f′(0) > 0$, and $f(x)\geqslant 0$ for any real number $x$, find the minimum value of $\frac{f(1)}{f′(0)}$."},
    #]
    #outputs = pipe(messages, max_new_tokens=2048,)
    #st.write(outputs[0]["generated_text"][-1])
    #messages = [
    #{"role": "user", "content": "Who are you?"},
    #]
    #st.write(pipe(messages)) 
    messages = [
    #{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #{"role": "user", "content": "Who are you?"},
    #]
    #outputs = pipe(messages, max_new_tokens=256,) 
    #st.write(outputs[0]["generated_text"][-1])
    st.write(pipeline([{"role": "system", "content": f"detailed thinking {thinking}"}, {"role": "user", "content": "Solve x*(sin(x)+2)=0"}]))
    
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

