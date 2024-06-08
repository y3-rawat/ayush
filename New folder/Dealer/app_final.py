from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for session management

key = "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"
GENERAL_DB_FAISS_PATH = 'collector/store/general_db_faiss'
SPECIFIC_DB_FAISS_PATH = 'collector/store/specific_db_faiss'

# Load the model
def load_llm():
    llm = ChatGroq(model="llama3-8b-8192", api_key=key)
    return llm

# Load general database
general_loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
general_data = general_loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})
general_db = FAISS.from_documents(general_data, embeddings)
general_db.save_local(GENERAL_DB_FAISS_PATH)

# Load specific database
specific_loader = CSVLoader(file_path="Dealer/dealers.csv", encoding="utf-8", csv_args={'delimiter': ','})
specific_data = specific_loader.load()
specific_db = FAISS.from_documents(specific_data, embeddings)
specific_db.save_local(SPECIFIC_DB_FAISS_PATH)

llm = load_llm()

general_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=general_db.as_retriever())
specific_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=specific_db.as_retriever())

def prompt_template(query, unique_id):
    cars = pd.read_csv("data_car.csv")
    df = pd.read_json("collector/collector_data.json")
    d = df.where(df.user_id == unique_id).dropna().iloc[0]
    
    
    template = f"""You are Eurotech Xchange Bot, a specialized assistant with detailed information about dealer activities and data. You have access to the following data about the dealer:
    As the Eurotech Xchange Bot, you can answer any questions related to this collector's car activities, preferences, bid amounts, or any other related information.
    , information about the data


    You have access to the following data about a specific collector:

    Booked Car: {d["car_added"]}
    how many peoples are visited :{d["peoples_visited"]}
    Logs:
    The visiters have visited the following cars with count: {d["visited_logs"]}
    
    this is a data about bids on cars and with which amount and car user has placed bid 
    Bid in car and Amount:
    {d["bid_placed_on_cars"]}

    total bids on car is : {d["Total bid placed on cars"]}
    As the Eurotech Xchange Bot, you can answer any questions related to the dealer's sales  
     
    You will provide the relevant information based on the data available.
    
    if user is just doing casual chat then just do casual chatting about the cats
    ---------------
        
    User_Chat: {query}
    """
    return template

def conversational_chat(query, unique_id):
    prompt = prompt_template(query, unique_id)
    general_result = general_chain({"question": prompt, "chat_history": session.get('history', [])})
    specific_result = specific_chain({"question": prompt, "chat_history": session.get('history', [])})
    # Combine results from both chains
    combined_result = f"General Info: {general_result['answer']}\nSpecific Info: {specific_result['answer']}"
    session['history'].append((query, combined_result))
    return combined_result

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    if 'generated' not in session:
        session['generated'] = ["Hello! Ask me anything about 🤗"]
    if 'past' not in session:
        session['past'] = ["Hey! 👋"]
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['query']
    unique_id = int(request.form['unique_id'])
    if user_input:
        output = conversational_chat(user_input, unique_id)
        session['past'].append(user_input)
        session['generated'].append(output)
    return jsonify({'past': session['past'], 'generated': session['generated']})

@app.route("/get", methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    unique_id = int(request.args.get('unique_id'))
    if user_text:
        output = conversational_chat(user_text, unique_id)
        return output
    return "Please provide a query parameter ?msg=Your_MSG and unique_id parameter ?unique_id=Your_ID"

if __name__ == "__main__":
    app.run(debug=True)
