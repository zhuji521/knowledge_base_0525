import dataclasses
import os

from dotenv import load_dotenv
load_dotenv()

@dataclasses.dataclass
class MineruConfig:
    base_url:str
    api_token:str

mineru_config=MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),
    api_token=os.getenv("MINERU_API_TOKEN","")
)
