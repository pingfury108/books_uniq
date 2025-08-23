import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # 火山引擎配置
    volcengine_api_key: Optional[str] = None
    volcengine_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    embedding_model: str = "volcengine/doubao-embedding"
    
    # ChromaDB配置
    chroma_persist_directory: str = "./chroma_db"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局设置实例
settings = Settings()