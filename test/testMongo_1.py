from pymongo import MongoClient

mongo_client=MongoClient("mongodb://192.168.171.128:27017")

db=mongo_client["test"]

# db.create_collection("classes")

db["classes"].insert_one({"name":"0929","age":18})

find_result=db["classes"].find()
# print(find_result)

for doc in find_result:
    print(doc)
    print(doc["name"])