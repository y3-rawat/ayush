from langchain_groq import ChatGroq
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import pandas as pd
from langchain.chains import ConversationChain
import os
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
with open ("website_about.text") as f:
        key = f.read()
with open ("data_car.csv") as f:
        Car_details = f.read()
prompt_for_conversational = f"""You are an chatbot on eurotech. it is a car website here is a detail of this webiste {key}
        Your job is to conversate with user give answer in less then 40 words conversation will be start after this  DO Not Answer Out of DataBase """

api_key_for_conversation = "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"
key = "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"
firsttime = True



# import requests

# url = "https://eurotechxchange.com/api/trpc/car.getCars"
# response = requests.get(url)

# # Check if the request was successful
# if response.status_code == 200:
#     print(response.json())  # or response.text if it's not in JSON format
# else:
#     print(f"Error {response.status_code}: {response.reason}")




# Initialize LLM model and data processing objects
model = ChatGroq(groq_api_key=api_key_for_conversation, model_name="llama3-70b-8192", temperature=0.2)
llm_groq = ChatGroq(groq_api_key=key, model_name="llama3-70b-8192")
conversation = ConversationChain(llm=llm_groq)

user_defined_path = os.getcwd()
cfg = {
    "save_charts": True,
    "save_charts_path": user_defined_path,
    "llm": model
}

# Load data and create a SmartDataFrame object


db = SQLDatabase.from_uri("sqlite:///Eurotech_Chat_with_csv.db")

chain = create_sql_query_chain(model, db)


# Function to determine whether to use the database and return the appropriate response
def userQuestion(user_query):
    global firsttime
    prompt = f"""You are an accomplished query understanding
        Your job is to analyse a question and determine if car_database will be use or not.  Carefully pay attention to the details and context of the question and of the text excerpt!
        here is information of Database-> in data base You have the names of car, price, how much miles is it ran etc.
        ===END===        
        This is the user chat: "{user_query}" 
        if user is asking any thing which should be check from the car's specifications then it means that you have give 1
        Does the question requires the information from database? Respond only with 1 or 0"""

    answer = llm_groq.invoke(prompt)
    print(answer,"0,1")
    if answer.content == "1":
        print("taking from database")
        response = chain.invoke({"question": user_query})
        print("ouutput_namseer",response)
        responnse=f'SELECT {response.split("SELECT")[1]}'
        print("res ->",responnse)
        query_answer_data = db.run(responnse)
        print("answerfromQuery -> ",query_answer_data)
        return query_answer_data

    elif(answer.content == "0"):
        if firsttime:
            print("First time hit, initializing conversation prompt.")
            conversation(prompt_for_conversational)
            firsttime = False
        # Respond using the conversation chain for subsequent queries
        print("Conversation without dataframe.")
        print('second time')
        response = conversation(user_query)
        return response
    else:
        a = f"this was the answer -> {answer.content}"
        return a

    
def main_module_for_genrating_answer(question,answer_from_database):
    prompt_for_answer = f"""You are an chatbot on eurotech. it is a car website here is a detail of this webiste {key}
    Your job is to conversate with user give answer in less then 40 words conversation will be start after this  DO Not Answer Out of DataBase 
    answer this question of user -> {question} 
    this is the answer of user query {answer_from_database} you have to answer it in a conversational way in less then 40 words"""

    ans = llm_groq.invoke(prompt_for_answer)
    print(ans,"calling from ans")
    return ans