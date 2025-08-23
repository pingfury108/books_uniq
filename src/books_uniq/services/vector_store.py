import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from ..config import settings


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
    
    async def add_document(
        self, 
        text: str, 
        embedding: List[float],
        metadata: Dict[str, Any] = None,
        collection_name: str = "books"
    ) -> str:
        """
        添加文档到向量数据库
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 生成文档ID
            document_id = str(uuid.uuid4())
            
            # 准备元数据
            doc_metadata = metadata or {}
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
    
    async def search_similar(
        self,
        query_embedding: List[float],
        collection_name: str = "books",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                        "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else None,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] and results["distances"][0] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"搜索失败: {str(e)}")
    
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
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata
            }
        except Exception as e:
            raise Exception(f"获取集合信息失败: {str(e)}")
    
    async def batch_add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "books"
    ) -> List[str]:
        """
        批量添加文档
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 生成文档ID
            document_ids = [str(uuid.uuid4()) for _ in texts]
            
            # 准备元数据
            if metadatas is None:
                metadatas = [{"text_length": len(text)} for text in texts]
            else:
                for i, metadata in enumerate(metadatas):
                    metadata.update({
                        "text_length": len(texts[i]),
                        "document_id": document_ids[i]
                    })
            
            # 批量添加到集合
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=document_ids
            )
            
            return document_ids
            
        except Exception as e:
            raise Exception(f"批量添加文档失败: {str(e)}")