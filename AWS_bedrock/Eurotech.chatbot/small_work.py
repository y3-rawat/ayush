
sm = {
    "questions": [
      {
        "question": "How can you help me?",
        "answer": "I'm here to assist you with your queries related to cars."
      },
      {
        "question": "who are you",
        "answer": "I am Car Chatbot"
      },
       {
        "question": "ok",
        "answer": "👍"
      },{
        "question": "oky",
        "answer": "👍"
      },{
        "question": "okey",
        "answer": "👍"
      },{
        "question": "okai",
        "answer": "👍"
      },
      {
        "question": "What is your name?",
        "answer": "I don't have any name yet but you can call me just chatbot."
      },
      {
        "question": "What do they call you?",
        "answer": "You can call me Chatbot."
      },
      {
        "question": "Who are you?",
        "answer": "I am Chatbot, Who is made on gemini."
      },
      {
        "question": "How old are you?",
        "answer": "I am just a computer program, so I don't have an age."
      },
      {
        "question": "When were you created?",
        "answer": "I was created in 2024."
      },
      {
        "question": "how are you?",
        "answer": "just chilling with you"
      },
      {
        "question": "how do you do",
        "answer": "I am doing fine, you can ask questions from me I will try my best"
      },
      {
        "question": "hy",
        "answer": "Hellow!"
      },  
      {
        "question": "hey",
        "answer": "hey there!"
      },
      {
        "question": "How are you",
        "answer": "I'm fine what about you!"
      },
        {
        "question": "fantastic",
        "answer": "that's greate"
      },
      {
        "question": "Hellow",
        "answer": "Hy!"
      },
      {
        "question": "hy there",
        "answer": "Hellow, How can i Assist you in This Great day"
      },
      {
        "question": "exit",
        "answer": "Okay"
      },  {
        "question": "thanks",
        "answer": "😊"
      },
        {
        "question": "thank you",
        "answer": "😊"
      },  {
        "question": "Okay",
        "answer": "😊"
      },{
        "question": "by",
        "answer": "😊"
      }
    ]
  }



import json
import string

def load_data():
    """Loads questions and answers from a JSON file, removing punctuation."""
    return sm['questions']

def create_chatbot():
    """Creates a chatbot using the loaded data, removing punctuation from questions."""
    data = load_data()
    chatbot = {}
    for item in data:
        # Remove punctuation from question (using translate() + str.maketrans() for efficiency)
        question = item['question'].lower().translate(str.maketrans('', '', string.punctuation))
        answer = item['answer']
        # Optionally handle synonyms or variations of the question
        chatbot[question] = answer
    return chatbot


