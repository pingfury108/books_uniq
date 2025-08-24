import uuid
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from ..config import settings

# 检查是否安装了numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class VectorStore:
    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 存储已创建的集合
        self._collections = {}
    
    def get_or_create_collection(self, collection_name: str = "books"):
        """
        获取或创建集合
        """
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
        return self._collections[collection_name]
    
    async def check_md5_exists(self, md5_hash: str, collection_name: str = "books") -> bool:
        """
        检查MD5是否已存在于集合中
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 尝试查询是否存在具有该MD5的记录
            results = collection.get(
                where={"md5": md5_hash},
                limit=1
            )
            
            return len(results['ids']) > 0
            
        except Exception as e:
            # 如果查询失败，假设不存在
            return False
    
    async def check_batch_md5_exists(self, md5_hashes: List[str], collection_name: str = "books") -> Dict[str, bool]:
        """
        批量检查MD5是否已存在于集合中，提高效率
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 获取所有已存在的MD5哈希
            results = collection.get(
                include=["metadatas"]
            )
            
            existing_hashes = set()
            metadatas = results.get('metadatas')
            if metadatas is not None and len(metadatas) > 0:
                for metadata in metadatas:
                    if metadata and 'md5' in metadata:
                        existing_hashes.add(metadata['md5'])
            
            # 为每个查询的MD5返回是否存在的结果
            return {md5: md5 in existing_hashes for md5 in md5_hashes}
            
        except Exception as e:
            # 如果查询失败，假设都不存在
            return {md5: False for md5 in md5_hashes}
    
    async def add_document(
        self, 
        text: str, 
        embedding: List[float],
        metadata: Dict[str, Any] = None,
        collection_name: str = "books"
    ) -> str:
        """
        添加文档到向量数据库（带MD5检查）
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 准备元数据
            doc_metadata = metadata or {}
            
            # 如果没有MD5，则生成一个
            if 'md5' not in doc_metadata:
                doc_metadata['md5'] = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            # 检查MD5是否已存在
            if await self.check_md5_exists(doc_metadata['md5'], collection_name):
                raise Exception(f"MD5 {doc_metadata['md5']} 已存在，跳过重复嵌入")
            
            # 生成文档ID，使用MD5作为基础
            document_id = f"doc_{doc_metadata['md5'][:12]}"
            
            doc_metadata.update({
                "text_length": len(text),
                "document_id": document_id
            })
            
            # 添加到集合
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
                ids=[document_id]
            )
            
            return document_id
            
        except Exception as e:
            raise Exception(f"添加文档失败: {str(e)}")
    
    async def batch_add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "books"
    ) -> Dict[str, Any]:
        """
        批量添加文档（带MD5检查，避免重复）
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            # 准备批量数据，过滤重复的MD5
            batch_texts = []
            batch_embeddings = []
            batch_metadatas = []
            batch_ids = []
            skipped_count = 0
            processed_hashes = set()
            
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                # 确保每个metadata都有MD5
                if 'md5' not in metadata:
                    metadata['md5'] = hashlib.md5(text.encode('utf-8')).hexdigest()
                
                md5_hash = metadata['md5']
                
                # 检查是否已处理过这个MD5（在当前批次中）
                if md5_hash in processed_hashes:
                    skipped_count += 1
                    continue
                
                # 检查数据库中是否已存在
                if await self.check_md5_exists(md5_hash, collection_name):
                    skipped_count += 1
                    continue
                
                # 准备数据
                document_id = f"doc_{md5_hash[:12]}"
                metadata.update({
                    "text_length": len(text),
                    "document_id": document_id,
                    "batch_index": i
                })
                
                batch_texts.append(text)
                batch_embeddings.append(embedding)
                batch_metadatas.append(metadata)
                batch_ids.append(document_id)
                processed_hashes.add(md5_hash)
            
            # 如果有数据需要添加
            if batch_texts:
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            return {
                "added_count": len(batch_ids),
                "skipped_count": skipped_count,
                "total_processed": len(texts),
                "document_ids": batch_ids
            }
            
        except Exception as e:
            raise Exception(f"批量添加文档失败: {str(e)}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        collection_name: str = "books",
        n_results: int = 5,
        min_similarity: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档（增加最小相似度阈值）
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 50),  # 增加查询数量以便过滤
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果并过滤低相似度结果
            formatted_results = []
            documents = results.get("documents")
            if documents is not None and len(documents) > 0 and len(documents[0]) > 0:
                distances = results.get("distances")
                metadatas = results.get("metadatas")
                
                for i in range(len(documents[0])):
                    distance = distances[0][i] if distances is not None and len(distances) > 0 and len(distances[0]) > i else 1.0
                    similarity = 1 - distance
                    
                    # 只保留相似度高于阈值的结果
                    if similarity >= min_similarity:
                        result = {
                            "document": documents[0][i],
                            "metadata": metadatas[0][i] if metadatas is not None and len(metadatas) > 0 and len(metadatas[0]) > i else {},
                            "distance": distance,
                            "similarity": similarity
                        }
                        formatted_results.append(result)
            
            # 按相似度降序排列
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return formatted_results[:n_results]
            
        except Exception as e:
            raise Exception(f"搜索失败: {str(e)}")
    
    async def get_collection_stats(self, collection_name: str = "books") -> Dict[str, Any]:
        """
        获取集合统计信息，包括MD5分布
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 获取所有记录
            all_results = collection.get(
                include=["metadatas"]
            )
            
            total_count = len(all_results.get('ids', []))
            
            # 统计MD5信息
            md5_hashes = set()
            metadatas = all_results.get('metadatas')
            if metadatas is not None and len(metadatas) > 0:
                for metadata in metadatas:
                    if metadata and 'md5' in metadata:
                        md5_hashes.add(metadata['md5'])
            
            return {
                "collection_name": collection_name,
                "total_documents": total_count,
                "unique_md5_count": len(md5_hashes),
                "collection_metadata": collection.metadata if hasattr(collection, 'metadata') else {}
            }
            
        except Exception as e:
            raise Exception(f"获取统计信息失败: {str(e)}")
    
    async def list_collections(self) -> List[str]:
        """
        列出所有集合
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            raise Exception(f"获取集合列表失败: {str(e)}")
    
    async def delete_document(self, document_id: str, collection_name: str = "books") -> bool:
        """
        删除文档
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=[document_id])
            return True
        except Exception as e:
            raise Exception(f"删除文档失败: {str(e)}")
    
    async def get_collection_info(self, collection_name: str = "books") -> Dict[str, Any]:
        """
        获取集合信息
        """
        try:
            stats = await self.get_collection_stats(collection_name)
            return stats
        except Exception as e:
            raise Exception(f"获取集合信息失败: {str(e)}")
    
    async def clear_collection(self, collection_name: str = "books") -> bool:
        """
        清空集合
        """
        try:
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            # 删除集合
            try:
                self.client.delete_collection(collection_name)
            except:
                pass  # 集合可能不存在
            
            # 重新创建空集合
            self.get_or_create_collection(collection_name)
            
            return True
        except Exception as e:
            raise Exception(f"清空集合失败: {str(e)}")
    
    async def get_collection_documents(
        self, 
        collection_name: str = "books", 
        limit: int = 50, 
        offset: int = 0,
        search_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取集合中的文档，支持分页和搜索
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # ChromaDB没有直接的offset功能，所以我们需要获取所有数据然后手动分页
            # 对于大数据集，这不是最优解，但ChromaDB的限制
            all_results = collection.get(
                include=["documents", "metadatas"]
            )
            
            documents = []
            all_documents = all_results.get('documents')
            all_metadatas = all_results.get('metadatas')
            if all_documents is not None and all_metadatas is not None and len(all_documents) > 0 and len(all_metadatas) > 0:
                all_ids = all_results.get('ids')
                for i, (doc, metadata) in enumerate(zip(all_documents, all_metadatas)):
                    # 如果有搜索文本，过滤结果
                    if search_text:
                        search_lower = search_text.lower()
                        if not (search_lower in doc.lower() or
                                (metadata and any(search_lower in str(v).lower() for v in metadata.values()))):
                            continue
                    
                    document_data = {
                        "id": all_ids[i] if all_ids is not None and len(all_ids) > i else f"doc_{i}",
                        "document": doc,
                        "metadata": metadata or {}
                    }
                    documents.append(document_data)
            
            # 手动实现分页
            total_filtered = len(documents)
            start_idx = offset
            end_idx = min(offset + limit, total_filtered)
            
            return documents[start_idx:end_idx]
            
        except Exception as e:
            raise Exception(f"获取集合文档失败: {str(e)}")
    
    async def get_collection_count(self, collection_name: str = "books", search_text: Optional[str] = None) -> int:
        """
        获取集合中的文档总数，支持搜索过滤
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            all_results = collection.get(
                include=["documents", "metadatas"]
            )
            
            if not search_text:
                all_ids = all_results.get('ids')
                return len(all_ids) if all_ids is not None else 0
            
            # 如果有搜索文本，计算匹配的文档数
            count = 0
            all_documents = all_results.get('documents')
            all_metadatas = all_results.get('metadatas')
            if all_documents is not None and all_metadatas is not None and len(all_documents) > 0 and len(all_metadatas) > 0:
                search_lower = search_text.lower()
                for doc, metadata in zip(all_documents, all_metadatas):
                    if (search_lower in doc.lower() or
                        (metadata and any(search_lower in str(v).lower() for v in metadata.values()))):
                        count += 1
            
            return count
            
        except Exception as e:
            raise Exception(f"获取集合文档数量失败: {str(e)}")

    async def get_document_metadata(
        self, 
        document_id: str, 
        collection_name: str = "books"
    ) -> Dict[str, Any]:
        """
        获取单个文档的元数据
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 获取指定文档
            results = collection.get(
                ids=[document_id],
                include=["metadatas", "documents"]
            )
            
            ids = results.get('ids')
            if ids is None or len(ids) == 0:
                raise Exception(f"文档 {document_id} 未找到")
            
            metadatas = results.get('metadatas')
            documents = results.get('documents')
            metadata = metadatas[0] if metadatas is not None and len(metadatas) > 0 and metadatas[0] else {}
            document = documents[0] if documents is not None and len(documents) > 0 and documents[0] else ""
            
            return {
                "document_id": document_id,
                "metadata": metadata,
                "document_preview": document[:200] + "..." if len(document) > 200 else document,
                "document_length": len(document)
            }
            
        except Exception as e:
            raise Exception(f"获取文档元数据失败: {str(e)}")

    async def get_embedding_by_md5(
        self, 
        md5_hash: str, 
        collection_name: str = "books"
    ) -> Optional[List[float]]:
        """
        通过MD5哈希获取单个记录的嵌入向量
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 获取指定MD5的记录
            results = collection.get(
                where={"md5": md5_hash},
                include=["embeddings"],
                limit=1
            )
            
            # 检查是否有结果，避免直接使用numpy数组在if条件中
            ids = results.get('ids')
            if ids is None or len(ids) == 0:
                return None
                
            embeddings = results.get('embeddings')
            if embeddings is None or len(embeddings) == 0:
                return None
                
            # 获取第一个匹配记录的嵌入向量
            embedding_array = embeddings[0]
            
            # 安全地转换为Python列表
            if HAS_NUMPY and isinstance(embedding_array, np.ndarray):
                return embedding_array.tolist()
            elif hasattr(embedding_array, 'tolist'):
                return embedding_array.tolist()
            elif isinstance(embedding_array, list):
                return embedding_array
            else:
                return None
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"获取MD5 {md5_hash} 的嵌入向量失败: {str(e)}")
            return None

