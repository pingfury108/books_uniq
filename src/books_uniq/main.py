from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from .services.embedding_service import EmbeddingService
from .services.vector_store import VectorStore
from .config import settings

load_dotenv()

app = FastAPI(title="Books Uniq API", description="文本向量化和相似性搜索API", version="1.0.0")

# 初始化服务
embedding_service = EmbeddingService()
vector_store = VectorStore()


class EmbedRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None
    collection_name: str = "books"


class BatchEmbedRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None
    collection_name: str = "books"


class SearchRequest(BaseModel):
    text: str
    collection_name: str = "books"
    n_results: int = 5


class EmbedResponse(BaseModel):
    message: str
    document_id: str


class BatchEmbedResponse(BaseModel):
    message: str
    document_ids: List[str]
    count: int


class SearchResponse(BaseModel):
    results: List[dict]


@app.get("/")
async def root():
    return {"message": "Books Uniq API is running!"}


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    将文本嵌入到向量数据库中
    """
    try:
        # 生成向量
        embedding = await embedding_service.get_embedding(request.text)
        
        # 存储到ChromaDB
        document_id = await vector_store.add_document(
            text=request.text,
            embedding=embedding,
            metadata=request.metadata or {},
            collection_name=request.collection_name
        )
        
        return EmbedResponse(
            message="文本已成功嵌入向量数据库",
            document_id=document_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入失败: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    搜索相似文本
    """
    try:
        # 生成查询向量
        query_embedding = await embedding_service.get_embedding(request.text)
        
        # 搜索相似文档
        results = await vector_store.search_similar(
            query_embedding=query_embedding,
            collection_name=request.collection_name,
            n_results=request.n_results
        )
        
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/collections")
async def list_collections():
    """
    列出所有集合
    """
    try:
        collections = await vector_store.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合列表失败: {str(e)}")


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_texts_batch(request: BatchEmbedRequest):
    """
    批量将文本嵌入到向量数据库中
    """
    try:
        # 批量生成向量
        embeddings = await embedding_service.get_batch_embeddings(request.texts)
        
        # 批量存储到ChromaDB
        document_ids = await vector_store.batch_add_documents(
            texts=request.texts,
            embeddings=embeddings,
            metadatas=request.metadatas,
            collection_name=request.collection_name
        )
        
        return BatchEmbedResponse(
            message=f"成功嵌入 {len(request.texts)} 条文本到向量数据库",
            document_ids=document_ids,
            count=len(document_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量嵌入失败: {str(e)}")


@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """
    获取集合信息
    """
    try:
        info = await vector_store.get_collection_info(collection_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合信息失败: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, collection_name: str = "books"):
    """
    删除文档
    """
    try:
        success = await vector_store.delete_document(document_id, collection_name)
        if success:
            return {"message": f"文档 {document_id} 已成功删除"}
        else:
            raise HTTPException(status_code=404, detail="文档未找到")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port, 
        debug=settings.api_debug
    )