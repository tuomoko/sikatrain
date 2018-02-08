#MongoDB
from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017")
db = client.sikatables
collection = db.train_data

print "Printing all training data"
for record in collection.find():
	pprint.pprint(record)

collection = db.score_data

print "Printing all score data"
for record in collection.find():
	pprint.pprint(record)
