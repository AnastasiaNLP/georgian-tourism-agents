from langsmith import Client
from config.settings import get_settings

def setup_langsmith():
    settings = get_settings()
    api_key = settings.langchain_api_key or settings.langsmith_api_key
    if not api_key:
        return None

    return Client(
        api_key=api_key,
        project=settings.langchain_project or settings.langsmith_project,
    )
