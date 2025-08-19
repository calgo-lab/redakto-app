from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from flair.data import Sentence
from pandas import DataFrame
from pydantic import BaseModel
from somajo import SoMaJo
from src.infrastructure.frameworks import FlairTaggerLoader, MT5Loader
from src.infrastructure.model_loading.model_service import ModelService
from typing import Any, Dict, List

import logging
import pandas as pd
import re
import subprocess
import threading
import uvicorn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()
    yield

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

labels = ['CITY', 'DATE', 'EMAIL', 'FAMILY', 'FEMALE', 'MALE', 'ORG', 
          'PHONE', 'STREET', 'STREETNO', 'UFID', 'URL', 'USER', 'ZIP']

supported_entity_set_model_dict = {
    "codealltag": ["bilstm-crf-plus", "deepset-gelectra-large", "google-mt5-base"]
}

# Define input schema
class ApiRequest(BaseModel):
    entity_set_id: str
    model_id: str
    input_texts: list[str]
    repeat: int

# Define output unit schema
class DataItem(BaseModel):
    output_dict: Dict[str, Any]
    output_text: str

# Define output schema
class ApiResponse(BaseModel):
    output: List[List[DataItem]]

def get_annotation_df_with_input_text_and_predicted_text(input_text: str, 
                                                         predicted_text: str,
                                                         labels: List[str]) -> DataFrame:
    tuples = list()

    input_text_length = len(input_text)
    input_text_copy = input_text[0: input_text_length]

    item_delim = ";"
    token_delim = ":"
    token_id = 0
    next_cursor = 0

    predicted_items = predicted_text.split(item_delim)
    for item in predicted_items:
        item = item.strip()
        label, token, pseudonym = "", "", ""

        for l in labels:
            if re.search((r'\b' + l + r'\b' +':'), item):
                label = l
                break

        if label != "":
            item = item[item.find(label + ":"):]
            value_splits = item.split(label+token_delim)
            token_pseudonym = value_splits[1].strip()
            pattern = r'^(.*?)\s*\*\*(.*?)\*\*'
            matches = re.search(pattern, token_pseudonym)
            if matches:
                token = matches.group(1)
                pseudonym = matches.group(2)
            else:
                token = token_pseudonym

            if len(token.strip()) > 0:

                start = input_text_copy.find(token)
                if start == -1 and ' ' in token:
                    token = token.replace(' ', '')
                    start = input_text_copy.find(token)

                if start != -1:
                    end = start + len(token)

                    token_id += 1
                    prev_cursor = next_cursor
                    next_cursor += end
                    input_text_copy = input_text[next_cursor: input_text_length]

                    start = prev_cursor + start
                    end = prev_cursor + end

                    tuples.append((
                        'T' + str(token_id),
                        label,
                        start,
                        end,
                        input_text[start:end],
                        pseudonym
                    ))

    return pd.DataFrame(
        tuples,
        columns=["Token_ID", "Label", "Start", "End", "Token", "Pseudonym"]
    )

def get_pseudonymized_text(input_text: str, predicted_annotation_df: DataFrame) -> str:
    output_text = input_text
    offset = 0
    for index, row in predicted_annotation_df.iterrows():
        output_text = output_text[:(row.Start+offset)] + row.Pseudonym + output_text[(row.End+offset):]
        offset += len(row.Pseudonym) - len(row.Token)
    return output_text


def _get_somajo_tokenized_sentences(text: str) -> List[Sentence]:
    tokenizer = SoMaJo("de_CMC", split_camel_case=False)
    sentences: List[Sentence] = list()
    for sentence in tokenizer.tokenize_text([text]):
        sentences.append(Sentence([token.text for token in sentence]))
    return sentences

def get_annotation_df_with_flair_tagger(input_text: str, tagger_id: str) -> DataFrame:
    tuples = list()
    
    email_content = input_text
    sentences = _get_somajo_tokenized_sentences(input_text)
    tagger = app.state.model_service.get_model('codealltag', tagger_id)

    sentences = FlairTaggerLoader.predict(tagger, sentences)

    email_content_length = len(email_content)
    email_content_copy = email_content[0:email_content_length]

    token_id = 0
    next_cursor = 0
    for sentence in sentences:
        labels = sentence.get_labels()
        for label in labels:
            text = label.data_point.text
            start = email_content_copy.find(text)
            if start == -1 and ' ' in text:
                start = email_content_copy.find(text.split(' ')[0])
                text = text.replace(' ', '')

            if start != -1:
                end = start + len(text)

                token_id += 1
                prev_cursor = next_cursor
                next_cursor += end
                email_content_copy = email_content[next_cursor:email_content_length]

                start = prev_cursor + start
                end = prev_cursor + end

                tuples.append((
                    'T' + str(token_id),
                    label.value,
                    start,
                    end,
                    email_content[start:end]
                ))
            else:
                token_id += 1
                tuples.append((
                    'T' + str(token_id),
                    label.value,
                    -1,
                    -1,
                    text
                ))

    return pd.DataFrame(
        tuples,
        columns=["Token_ID", "Label", "Start", "End", "Token"]
    )

def _process_for_codealltag_mT5(input_data, output):
    model, tokenizer = app.state.model_service.get_model(input_data.entity_set_id, input_data.model_id)
    for input_text in input_data.input_texts:
        per_text_output: List[DataItem] = list()
        tokenized_inputs = MT5Loader.tokenize_text(tokenizer, 
                                                   input_text, 
                                                   max_length=512, 
                                                   padding="max_length", 
                                                   truncation=True)
        
        for repeat_count in range(0, input_data.repeat):
            predicted_text = MT5Loader.generate(
                model=model, 
                tokenizer=tokenizer, 
                tokenized_inputs=tokenized_inputs, 
                max_length=512, 
                temperature=0.8, 
                do_sample=True, 
                top_k=100
            )

            print(predicted_text)
            
            output_df = get_annotation_df_with_input_text_and_predicted_text(input_text, predicted_text, labels)
            output_text = get_pseudonymized_text(input_text, output_df)
            data_item = DataItem(output_dict=output_df.to_dict(), output_text=output_text)
            per_text_output.append(data_item)

        output.append(per_text_output)
    
    return output

def _process_for_codealltag_tagger(input_data, output):
    tagger_id = input_data.model_id
    for input_text in input_data.input_texts:
        per_text_output: List[DataItem] = list()
        output_df = get_annotation_df_with_flair_tagger(input_text, tagger_id)
        data_item = DataItem(output_dict=output_df.to_dict(), output_text='not_available')
        per_text_output.append(data_item)
        output.append(per_text_output)
    
    return output

def _process_for_entity_set_and_model(input_data, output):
    if input_data.entity_set_id == 'codealltag':
        if input_data.model_id == 'google-mt5-base':
            output = _process_for_codealltag_mT5(input_data, output)
        else:
            output = _process_for_codealltag_tagger(input_data, output)

    return output

@app.post("/predict", response_model=ApiResponse)
def predict(input_data: ApiRequest):
    output: List[List[DataItem]] = list()
    try:
        if not input_data.entity_set_id or input_data.entity_set_id not in supported_entity_set_model_dict.keys():
            msg: str = f'Invalid entity_set_id={input_data.entity_set_id}, supported values: {list(supported_entity_set_model_dict.keys())}'
            raise Exception(msg)
        
        if not input_data.model_id or input_data.model_id not in supported_entity_set_model_dict[input_data.entity_set_id]:
            msg: str = f'Invalid model_id, supported values: {supported_entity_set_model_dict[input_data.entity_set_id]}'
            raise Exception(msg)
        
        output = _process_for_entity_set_and_model(input_data, output)
        return ApiResponse(output=output)
    
    except Exception as e:
        logger.error(f"predict failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"request failed, {str(e)}")

# Function to run streamlit with a subprocess
def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"])

# Start streamlit subprocess in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)