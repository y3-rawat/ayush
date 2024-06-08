import os
import boto3
import botocore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

# ------------------------------------------------------------------------
# Amazon Bedrock Settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_kwargs =  { 
    "max_tokens_to_sample": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model_id = "anthropic.claude-instant-v1"

# ------------------------------------------------------------------------
# MongoDB

# Connect to MDB
import bson
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# load environment variable
load_dotenv()

# get mongodb connection url
mongodb_uri = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(mongodb_uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# ------------------------------------------------------------------------
# LCEL: chain(prompt | model | output_parser) + RunnableWithMessageHistory

template = [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
]

prompt = ChatPromptTemplate.from_messages(template)

model = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain = prompt | model | StrOutputParser()

# MongoDB Chat Message History
history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=mongodb_uri,
    database_name="chat_db",
    collection_name="chat_histories",
)

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: MongoDBChatMessageHistory(
        session_id="test_session",
        connection_string=mongodb_uri,
        database_name="chat_db",
        collection_name="chat_histories",
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# ------------------------------------------------------------------------
# Streamlit

import streamlit as st

# Page title
st.set_page_config(page_title='Streamlit Chat')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear Chat History
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With MongoDB Memory :brain:')
    if st.button('Delete Chat History'): # Delete MongoDB Collection Sidebar Button
        db = client["chat_db"]
        col = db["chat_histories"]
        try:
            col.drop()
            st.write("Collection 'chat_histories' deleted")
        except pymongo.errors.PyMongoError:
            st.write("Error deleting collection")
    streaming_on = st.toggle('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

# Streamlit Chat Input - User Prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # This is where we configure the session id
    config = {"configurable": {"session_id": "<SESSION_ID>"}}

    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            for chunk in chain_with_history.stream({"question": prompt}, config = config):
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = chain_with_history.invoke({"question": prompt}, config = config)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})