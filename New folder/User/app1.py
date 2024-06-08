from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langdetect import detect
from deep_translator import GoogleTranslator
import os
import json

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')  # Use environment variable for secret key

# Constants
KEY = os.environ.get('API_KEY', "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")
DB_FAISS_PATH = 'collector/store/db_faiss'
CSV_FILE_PATH = "data_car.csv"

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_llm(api_key):
    return ChatGroq(model="llama3-8b-8192", api_key=api_key)

# Initialize components
def initialize_components():
    loader = CSVLoader(file_path=CSV_FILE_PATH, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm(KEY)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    return chain

chain = initialize_components()

# Generate prompt template
def prompt_template(query):
    template = (f"You are Eurotech Xchange Bot, a specialized assistant with detailed information about car collectors and their activities. "
                f"We also accept digital currencies like Bitcoin. If the user asks anything complex, provide the email and say that our team will help: info@eurotechxchange.com\n"                
                f"Answer the user in a professional manner with clear and concise information.\n"
                f"-------\n"
                f"Question: {query}\n")
    
    return template

# Conversational chat function with language detection and translation
def conversational_chat(query):
    try:
        
        

        # Detect language
        detected_language = detect(query)
        
        

        prompt = prompt_template(query)
        result = chain({"question": prompt, "chat_history": session.get('history', [])})
        response = result["answer"]

        # Translate response if necessary
        if detected_language != 'en':
            response = GoogleTranslator(source='auto', target=detected_language).translate(response)

        session['history'].append((query, response))
        return response
    except Exception as e:
        error_message = "An error occurred while processing your request. Please try again later."
       

@app.route('/')
def index():
    session.setdefault('history', [])
    session.setdefault('generated', ["Hello! Ask me anything about 🤗"])
    session.setdefault('past', ["Hey! 👋"])
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['query']
    unique_id = int(request.form['unique_id'])
    if user_input:
        output = conversational_chat(user_input)
        session['past'].append(user_input)
        session['generated'].append(output)
    return jsonify({'past': session['past'], 'generated': session['generated']})

@app.route("/get", methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    
    if user_text:
        output = conversational_chat(user_text)
        return output
    return "Please provide a query parameter ?msg=Your_MSG and unique_id parameter ?unique_id=Your_ID"

if __name__ == "__main__":
    app.run(debug=True)
