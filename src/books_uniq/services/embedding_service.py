import asyncio
import logging
from typing import List, Optional
from openai import AsyncOpenAI
import numpy as np
from ..config import settings

logger = logging.getLogger(__name__)


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
        批量获取文本的向量表示（遵守API限制：1200 RPM, 1200000 TPM）
        """
        if not self.api_key or not self.client:
            raise ValueError("VOLCENGINE_API_KEY 环境变量未设置，请配置后再使用向量化功能")
        
        # 批次设置：每批100个文档，遵守API限制
        batch_size = 100
        all_embeddings = []
        
        # API限制计算
        max_requests_per_minute = 1200  # 1200 RPM
        max_tokens_per_minute = 1200000  # 1200000 TPM
        
        # 计算每个文本的平均token数（估算，中文字符约等于token数）
        avg_tokens_per_text = sum(len(text) for text in texts[:10]) // min(10, len(texts)) if texts else 50
        estimated_tokens_per_batch = avg_tokens_per_text * batch_size
        
        # 计算需要的延迟时间
        # 基于RPM限制：每60秒最多1200个请求
        min_delay_for_rpm = 60.0 / max_requests_per_minute  # ~0.05秒
        
        # 基于TPM限制：确保不超过token限制
        estimated_batches_per_minute = max_tokens_per_minute // estimated_tokens_per_batch
        min_delay_for_tpm = 60.0 / max(estimated_batches_per_minute, 1)
        
        # 使用较大的延迟值，确保不超过任何限制
        delay_seconds = max(min_delay_for_rpm, min_delay_for_tpm, 3.0)  # 最少3秒间隔
        
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"开始批量处理 {len(texts)} 条文本，分为 {total_batches} 个批次")
            logger.info(f"每批次 {batch_size} 个文档，批次间隔 {delay_seconds:.1f} 秒")
            logger.info(f"预估每文本 {avg_tokens_per_text} tokens，每批次约 {estimated_tokens_per_batch} tokens")
            logger.info(f"预计总耗时约 {(total_batches * delay_seconds) / 60:.1f} 分钟")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"处理第 {batch_num}/{total_batches} 批次，文本数量: {len(batch_texts)}")
                
                # 发送API请求
                start_time = asyncio.get_event_loop().time()
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                request_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"第 {batch_num}/{total_batches} 批次完成，耗时 {request_time:.2f} 秒")
                
                # 添加延迟以遵守API限制
                if batch_num < total_batches:
                    logger.info(f"等待 {delay_seconds:.1f} 秒后处理下一批次...")
                    await asyncio.sleep(delay_seconds)
            
            logger.info(f"所有批次处理完成，总向量数: {len(all_embeddings)}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"批量获取向量失败: {str(e)}")
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