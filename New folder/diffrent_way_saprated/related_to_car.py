import pymongo
import requests
client = pymongo.MongoClient("mongodb+srv://wwwyashrawat542:L4O2cJ7g3yZeN6aS@euro.grkmi2o.mongodb.net/?retryWrites=true&w=majority&appName=euro")
db = client.eurotech
collection = db.dummyeuro
items = collection.find().limit(5)
hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

for doc in collection.find({'description': {"$exists": True}}, {'_id': 1, 'description': 1}).limit(50):
        embedding = generate_embedding(doc['description'])
        collection.update_one({'_id': doc['_id']}, {'$set': {'description_embedding_hf': embedding}})
