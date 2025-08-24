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
        批量获取文本的向量表示（优化后的API限制策略：1200 RPM, 1200000 TPM）
        """
        if not self.api_key or not self.client:
            raise ValueError("VOLCENGINE_API_KEY 环境变量未设置，请配置后再使用向量化功能")
        
        all_embeddings = []
        
        # API限制参数
        max_requests_per_minute = 1200  # 1200 RPM
        max_tokens_per_minute = 1200000  # 1200000 TPM
        
        # 计算每个文本的平均token数（中文字符约等于token数）
        sample_texts = texts[:min(20, len(texts))]  # 扩大样本提高准确性
        avg_tokens_per_text = sum(len(text) for text in sample_texts) // len(sample_texts) if sample_texts else 50
        
        # 动态批次大小：基于TPM限制优化
        # 目标：使用90%的RPM配额，80%的TPM配额
        target_requests_per_minute = max_requests_per_minute * 0.9  # 1080 requests/min
        target_tokens_per_minute = max_tokens_per_minute * 0.8     # 960000 tokens/min
        
        # 计算最优批次大小
        max_batch_size_by_tokens = int(target_tokens_per_minute / target_requests_per_minute / avg_tokens_per_text)
        max_batch_size_by_api = 2000  # API单次最大限制（保守估算）
        optimal_batch_size = min(max_batch_size_by_tokens, max_batch_size_by_api, 300)  # 不超过300条
        
        # 确保批次大小合理
        batch_size = max(optimal_batch_size, 20)  # 最少20条
        
        # 计算请求间隔
        estimated_tokens_per_batch = avg_tokens_per_text * batch_size
        
        # 基于RPM的最小间隔
        min_delay_rpm = 60.0 / target_requests_per_minute  # ~0.056秒
        
        # 基于TPM的最小间隔
        requests_per_minute_by_tokens = target_tokens_per_minute / estimated_tokens_per_batch
        min_delay_tpm = 60.0 / requests_per_minute_by_tokens if requests_per_minute_by_tokens > 0 else min_delay_rpm
        
        # 选择更严格的限制，并加10%安全边际
        delay_seconds = max(min_delay_rpm, min_delay_tpm) * 1.1
        
        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"开始优化批量处理 {len(texts)} 条文本，分为 {total_batches} 个批次")
            logger.info(f"动态批次大小: {batch_size} 个文档，优化间隔: {delay_seconds:.3f} 秒")
            logger.info(f"预估每文本 {avg_tokens_per_text} tokens，每批次约 {estimated_tokens_per_batch} tokens")
            logger.info(f"预计总耗时约 {(total_batches * delay_seconds) / 60:.2f} 分钟 (优化前需 {(total_batches * 3.0) / 60:.1f} 分钟)")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"处理第 {batch_num}/{total_batches} 批次，文本数量: {len(batch_texts)}")
                
                # 发送API请求，添加重试机制
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        start_time = asyncio.get_event_loop().time()
                        response = await self.client.embeddings.create(
                            model=self.model,
                            input=batch_texts
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        
                        request_time = asyncio.get_event_loop().time() - start_time
                        logger.info(f"第 {batch_num}/{total_batches} 批次完成，耗时 {request_time:.2f} 秒")
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            # API速率限制错误，使用指数退避
                            backoff_time = delay_seconds * (2 ** retry_count)
                            logger.warning(f"遇到API速率限制，第 {retry_count} 次重试，等待 {backoff_time:.1f} 秒...")
                            await asyncio.sleep(backoff_time)
                        else:
                            # 其他错误，直接抛出
                            raise e
                
                if retry_count >= max_retries:
                    raise Exception(f"API请求失败，已重试 {max_retries} 次")
                
                # 添加优化后的延迟
                if batch_num < total_batches:
                    logger.debug(f"等待 {delay_seconds:.3f} 秒后处理下一批次...")
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