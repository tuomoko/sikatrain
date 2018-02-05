#MongoDB
from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017")
db = client.sikatables
collection = db.train_data

for record in collection.find():
	pprint.pprint(record)
