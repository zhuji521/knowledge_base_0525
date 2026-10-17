import dataclasses
import os

import load_dotenv

load_dotenv()

@dataclasses
class MineruConfig:
    base_url:str
    api_token:str

mineru_config=MineruConfig(
    base_url=os.getenv("MINURU_URL",""),
    api_token=os.getenv("MINERU_API_TOKEN","")
)
