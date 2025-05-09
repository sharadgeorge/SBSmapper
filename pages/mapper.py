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

export HF_ENDPOINT=https://hf-mirror.com
torch.set_grad_enabled(False) 

st.title("📘SBS V2.0 mapper")
#st.header("Internal descriptions can be mapped to SBS codes in the below chapters")
#st.image("images/SBS_Chapter_Index.png", use_container_width=True)
st.header("Map internal descriptions to SBS codes with        Sentence Transformer + Reasoning Models")

st.subheader("Select specific Chapter for quicker results")
#df_chapters = pd.read_csv("SBS_V2_0/Chapter_Index_Rows.csv") 
df_chapters = pd.read_csv("SBS_V2_0/Chapter_Index_Rows_with_total.csv") 

startrowindex_list = df_chapters["from_row_index"].tolist()
endrowindex_list = df_chapters["to_row_index"].tolist()
allchapters_rows_list = []
for s, e in zip(startrowindex_list, endrowindex_list):
    eachchapter_rows_list = list(range(s,e))
    allchapters_rows_list.append(eachchapter_rows_list)
df_chapters['range_of_rows'] = allchapters_rows_list 

def dataframe_with_selections(df_chapters: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    df_with_selections = df_chapters.copy()
    df_with_selections.insert(0, "Select", init_value)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df_chapters.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

if "selected_chapters" not in st.session_state:
    st.session_state['selected_chapters'] = []
    st.session_state['selected_rows'] = []
selected_chapters_list = st.session_state.selected_chapters
selected_rows_list = st.session_state.selected_rows

selected_chapters = dataframe_with_selections(df_chapters)
#st.write("Your selection:")
#st.write(selected_chapters) 
#chapter_start_row_index = selected_chapters['from_row_index']
#chapter_end_row_index = selected_chapters['to_row_index']
chapter_rows_indexes_list = selected_chapters['range_of_rows'].tolist()
#st.write("CHAPTER START ROW INDEX: ", chapter_start_row_index)
#st.write("CHAPTER END ROW INDEX: ", chapter_end_row_index)
#st.write("CHAPTER ROWS INDEXES LIST: ", chapter_rows_indexes_list)
combined_chapters_rows_indexes_list = [0] 
for item in chapter_rows_indexes_list:
    combined_chapters_rows_indexes_list.extend(item)

if len(combined_chapters_rows_indexes_list) == 1: 
    st.warning("Please select at least one chapter")
#st.write("COMBINED CHAPTERS ROWS INDEXES LIST: ", combined_chapters_rows_indexes_list)
df_SBS = pd.read_csv("SBS_V2_0/Code_Sheet.csv", header=0, skip_blank_lines=False, skiprows = lambda x: x not in combined_chapters_rows_indexes_list)
#st.write(df_SBS.head(5))
SBScorpus = df_SBS['Long_Description'].values.tolist()

dictA = {"Score": [], "SBS Code": [], "SBS Description V2.0": []}
dfALL = pd.DataFrame.from_dict(dictA)


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

#INTdesc_input = st.text_input("Type internal description", key="user_input") 
INTdesc_input = st.text_input(r"$\textsf{\Large Type internal description}$", key="user_input")

placeholder, right_column = st.columns(2)
#placeholder_clicked = placeholder.button("Perform some action", key="user_placeholder")
right_column.button("Reset description", on_click=on_click)

numMAPPINGS_input = 5


## Define the Sentence Transformer models
st_models = {
    '(higher speed) original model for general domain: all-MiniLM-L6-v2': 'all-MiniLM-L6-v2', 
    '(high performance) original model for general domain: all-mpnet-base-v2': 'all-mpnet-base-v2', 
    '(expected in future) fine-tuned model for medical domain: all-MiniLM-L6-v2': 'all-MiniLM-L6-v2', 
    '(expected in future) fine-tuned model for medical domain: all-mpnet-base-v2': 'all-mpnet-base-v2',
}

#model = SentenceTransformer('all-MiniLM-L6-v2') # fastest
#model = SentenceTransformer('all-mpnet-base-v2') # best performance
#model = SentenceTransformers('all-distilroberta-v1')
#model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5') 
#model = SentenceTransformer('clips/mfaq')

## Create the select Sentence Transformer box
selected_st_model = st.selectbox('Current selected Sentence Transformer model:', list(st_models.keys())) # or 'Choose a Sentence Transformer Model'
#st.write("Current selection:", selected_st_model)

## Get the selected SentTrans model
SentTrans_model = st_models[selected_st_model]

## Load the Sentence Transformer model ...
@st.cache_resource
def load_model():
    model = SentenceTransformer(SentTrans_model)
    return model
model = load_model() 

### Define the Reasoning models
#rs_models = {
#    '(medium speed) original model for general domain: meta-llama/Llama-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct', 
#    '(slower speed) original model for general domain: Qwen/Qwen2-1.5B-Instruct': 'Qwen/Qwen2-1.5B-Instruct', 
#    '(medium speed) original model for general domain: EpistemeAI/ReasoningCore-1B-r1-0': 'EpistemeAI/ReasoningCore-1B-r1-0',
#    '(expected in future) fine-tuned model for medical domain: meta-llama/Llama-3.2-1B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct', 
#    '(expected in future) fine-tuned model for medical domain: Qwen/Qwen2-1.5B-Instruct': 'Qwen/Qwen2-1.5B-Instruct',
#}
### Create the select Reasoning box
#selected_rs_model = st.selectbox('Current selected Reasoning model:', list(rs_models.keys())) # or 'Choose a Reasoning Model'
##st.write("Current selection:", selected_rs_model)
### Get the selected Reasoning model
#Reasoning_model = rs_models[selected_rs_model]
### Load the Reasoning model as pipeline ...
#@st.cache_resource
#def load_pipe():
#    pipe = pipeline("text-generation", model=Reasoning_model, device_map=device,) # device_map="auto", torch_dtype=torch.bfloat16 
#    return pipe 
#pipe = load_pipe()

# Semantic search, Compute cosine similarity between INTdesc_embedding and SBS descriptions
INTdesc_embedding = model.encode(INTdesc_input)
SBScorpus_embeddings = model.encode(SBScorpus)

if INTdesc_input is not None and st.button("Map to SBS codes", key="run_st_model"): 
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
