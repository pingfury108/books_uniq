#!/usr/bin/env python3
"""
Books Uniq API 启动脚本
"""
import uvicorn
from src.books_uniq.main import app
from src.books_uniq.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.books_uniq.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info"
    )