from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch


web_search = TavilySearch(
    max_results=5
)