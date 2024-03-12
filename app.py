# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import os
import requests
import chainlit as cl
from dotenv import load_dotenv

import llama_index
from llama_index.core import set_global_handler

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# ChatOpenAI Templates
system_template = """You are a helpful assistant who always speaks in a pleasant tone!
"""

user_template = """{input}
Think through your response step by step.
"""
# query_engine = index.as_query_engine()
# response = query_engine.query("Who is the E-VP, Operations - and how old are they?")
# print(response.response)
#
# response = query_engine.query("What is the gross carrying amount of Total Amortizable Intangible Assets for Jan 29, 2023?")
# print(response.response)
# if storage folder exists and is not empty, load the index from it else from documents


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    load_dotenv()

    set_global_handler("wandb", run_args={"project": "aie1-llama-index-middleterm"})
    wandb_callback = llama_index.core.global_handler

    Settings.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    index = None
    if os.path.exists("./storage") and os.listdir("./storage"):
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)
    else:
        with requests.get('https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf',
                          stream=True) as r:
            r.raise_for_status()  # Raises a HTTPError if the response status code is 4XX/5XX
            os.makedirs(os.path.dirname('nvidia_data/paper.pdf'), exist_ok=True)
            with open('nvidia_data/paper.pdf', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        documents = SimpleDirectoryReader('nvidia_data/').load_data()
        faiss_index = faiss.IndexFlatL2(1536)
        storage_context = StorageContext.from_defaults(vector_store=FaissVectorStore(faiss_index=faiss_index))
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            persist_dir="./storage"
        )

    cl.user_session.set("wandb_callback", wandb_callback)
    cl.user_session.set("query_engine", index.as_query_engine())


@cl.on_message
async def main(message: cl.Message):
    Settings.callback_manager = cl.user_session.get("wandb_callback")
    query_engine = cl.user_session.get("query_engine")
    template = (f"You are a helpful assistant who always speaks in a pleasant tone! responds to user input with a step by step guide using this context: {message.content} input: {input}")
    response = query_engine.query(template)

    response_message = cl.Message(content="")
    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
    cl.user_session.get("wandb_callback").finish()


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
    cl.user_session.get("wandb_callback").finish()