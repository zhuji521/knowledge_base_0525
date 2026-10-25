from utils.mongo_test_utils import get_history_mongo_tool

mongo_client=get_history_mongo_tool()

mongo_client.chat_message.insert_one({"session_id":1,"message":"hello","ts":1234567890})

