import asyncio
from dotenv import load_dotenv
import os
from llama_index.vector_stores.opensearch import OpensearchVectorClient
import openai
from App.src.VectorStore.RetrieveVectorStoreIndex import Retrive_Vector_Store
from App.src.Chatbot.prompt import Custom_prmpt
from App.src.Chatbot.QueryEngine import Query_Engine

async def create_client():
    # Load environment variables from .env file
    load_dotenv()

    # Access the environment variables for OpenAI and Hugging Face API keys
    OPENAPIKEY = os.getenv('OPENAPIKEY')
    HUGGINGFACETOKEN = os.getenv('HUGGINGFACETOKEN')
    endpoint = os.getenv('endpoint')
    OPENSEARCHUSERNAME = os.getenv('OPENSEARCHUSERNAME')
    OPENSEARCHUSERPASSWORD = os.getenv('OPENSEARCHUSERPASSWORD')

    openai.api_key = OPENAPIKEY

    # Define paths for data folder and Hugging Face model
    Folder_Path = "Documents"
    huggingface_model_path = "osmanh/Harry_Potter_and_the_Sorcerers_Stone_en"
    model_id = "mboth/distil-eng-quora-sentence"
    opensearch_idx = "document-index"
    # OpensearchVectorClient stores text in this field by default
    text_field = "content"
    # OpensearchVectorClient stores embeddings in this field by default
    embedding_field = "embedding"

    auth = (OPENSEARCHUSERNAME, OPENSEARCHUSERPASSWORD)

    client = OpensearchVectorClient(
        endpoint=endpoint,
        index=opensearch_idx,
        dim=768,
        embedding_field=embedding_field,
        text_field=text_field,
        http_auth=auth,
    )
    # If we want to finetune our model
    # FineTune.finetune(Folder_Path, OPENAPIKEY, HUGGINGFACETOKEN, model_id)
    # VectorStore.vectorsearch(client, HUGGINGFACETOKEN, model_id, Folder_Path)
    Index = Retrive_Vector_Store(client, HUGGINGFACETOKEN, model_id)

    custom_qa_prompt = Custom_prmpt()
    default_query_engine = Query_Engine(Index, custom_qa_prompt)

    return default_query_engine
