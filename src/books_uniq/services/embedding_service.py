import asyncio
from typing import List, Optional
from openai import AsyncOpenAI
import numpy as np
from ..config import settings


class EmbeddingService:
    def __init__(self):
        self.api_key = settings.volcengine_api_key
        self.base_url = settings.volcengine_base_url
        self.model = settings.embedding_model
        
        # 只在实际调用时检查API密钥，而不是在初始化时
        if self.api_key:
            # 初始化 OpenAI 客户端
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的向量表示
        """
        if not self.api_key or not self.client:
            raise ValueError("VOLCENGINE_API_KEY 环境变量未设置，请配置后再使用向量化功能")
            
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            raise Exception(f"获取向量失败: {str(e)}")
    
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的向量表示
        """
        if not self.api_key or not self.client:
            raise ValueError("VOLCENGINE_API_KEY 环境变量未设置，请配置后再使用向量化功能")
            
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            raise Exception(f"批量获取向量失败: {str(e)}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        return dot_product / (norm1 * norm2)