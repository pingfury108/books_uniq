from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import logging
import traceback
import sys

from .services.embedding_service import EmbeddingService
from .services.vector_store import VectorStore
from .services.excel_processor import ExcelProcessor
from .config import settings

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('books_uniq.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Books Uniq API", description="文本向量化和相似性搜索API", version="1.0.0")

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="src/books_uniq/static"), name="static")
templates = Jinja2Templates(directory="src/books_uniq/templates")

# 初始化服务
try:
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    excel_processor = ExcelProcessor()
    logger.info("所有服务初始化成功")
except Exception as e:
    logger.error(f"服务初始化失败: {str(e)}")
    logger.error(traceback.format_exc())
    raise


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器，记录所有未处理的异常
    """
    error_id = id(exc)  # 生成错误ID以便跟踪
    logger.error(f"[ERROR-{error_id}] 全局异常捕获:")
    logger.error(f"[ERROR-{error_id}] 请求路径: {request.url}")
    logger.error(f"[ERROR-{error_id}] 请求方法: {request.method}")
    logger.error(f"[ERROR-{error_id}] 异常类型: {type(exc).__name__}")
    logger.error(f"[ERROR-{error_id}] 异常信息: {str(exc)}")
    logger.error(f"[ERROR-{error_id}] 异常堆栈:\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"内部服务器错误 (ID: {error_id})",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_id": error_id
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP异常处理器
    """
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail} - 路径: {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


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


# Web界面路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/database", response_class=HTMLResponse)
async def database_manager(request: Request):
    return templates.TemplateResponse("database.html", {"request": request})


@app.get("/api")
async def api_root():
    return {"message": "Books Uniq API is running!"}


# Excel文件上传和处理
@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    """
    上传并解析Excel文件
    """
    logger.info(f"开始处理Excel文件: {file.filename}")
    try:
        # 验证文件类型
        if not file.filename.endswith(('.xlsx', '.xls')):
            logger.warning(f"无效文件类型: {file.filename}")
            raise HTTPException(status_code=400, detail="请上传Excel文件 (.xlsx 或 .xls)")
        
        # 读取文件内容
        content = await file.read()
        logger.info(f"文件大小: {len(content)} 字节")
        
        # 处理Excel文件
        result = excel_processor.process_excel_file(content)
        logger.info(f"Excel处理完成: 总记录数 {result['total_records']}")
        
        # 验证数据质量
        validation = excel_processor.validate_data(result['records'])
        logger.info(f"数据验证结果: {validation['message']}")
        
        return {
            "filename": file.filename,
            "total_records": result['total_records'],
            "columns": result['columns'],
            "records": result['records'],
            "sample_data": result['sample_data'],
            "validation": validation
        }
        
    except HTTPException:
        raise  # 重新抛出HTTP异常
    except Exception as e:
        logger.error(f"Excel文件处理失败: {file.filename}")
        logger.error(f"错误详情: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    将文本嵌入到向量数据库中
    """
    logger.info(f"开始嵌入文本到集合: {request.collection_name}")
    try:
        # 生成向量
        embedding = await embedding_service.get_embedding(request.text)
        logger.debug(f"向量生成完成，维度: {len(embedding)}")
        
        # 存储到ChromaDB
        document_id = await vector_store.add_document(
            text=request.text,
            embedding=embedding,
            metadata=request.metadata or {},
            collection_name=request.collection_name
        )
        logger.info(f"文档嵌入成功，ID: {document_id}")
        
        return EmbedResponse(
            message="文本已成功嵌入向量数据库",
            document_id=document_id
        )
    except Exception as e:
        logger.error(f"文本嵌入失败: {str(e)}")
        logger.error(f"请求文本: {request.text[:100]}...")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"嵌入失败: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """
    搜索相似文本
    """
    logger.info(f"开始搜索相似文本，集合: {request.collection_name}, 返回数量: {request.n_results}")
    try:
        # 生成查询向量
        query_embedding = await embedding_service.get_embedding(request.text)
        logger.debug(f"查询向量生成完成")
        
        # 搜索相似文档
        results = await vector_store.search_similar(
            query_embedding=query_embedding,
            collection_name=request.collection_name,
            n_results=request.n_results
        )
        logger.info(f"搜索完成，找到 {len(results)} 个相似结果")
        
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"相似性搜索失败: {str(e)}")
        logger.error(f"查询文本: {request.text[:100]}...")
        logger.error(traceback.format_exc())
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
    批量将文本嵌入到向量数据库中（支持MD5去重）
    """
    logger.info(f"开始批量嵌入 {len(request.texts)} 条文本到集合: {request.collection_name}")
    try:
        # 批量生成向量
        logger.info("开始生成向量...")
        embeddings = await embedding_service.get_batch_embeddings(request.texts)
        logger.info(f"向量生成完成，数量: {len(embeddings)}")
        
        # 批量存储到ChromaDB（带MD5检查）
        logger.info("开始存储到向量数据库...")
        result = await vector_store.batch_add_documents(
            texts=request.texts,
            embeddings=embeddings,
            metadatas=request.metadatas,
            collection_name=request.collection_name
        )
        logger.info(f"批量嵌入完成: 添加 {result['added_count']} 条，跳过 {result['skipped_count']} 条")
        
        return BatchEmbedResponse(
            message=f"处理完成: 成功嵌入 {result['added_count']} 条，跳过重复 {result['skipped_count']} 条",
            document_ids=result['document_ids'],
            count=result['added_count']
        )
    except Exception as e:
        logger.error(f"批量嵌入失败: {str(e)}")
        logger.error(f"文本数量: {len(request.texts)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"批量嵌入失败: {str(e)}")


@app.post("/process-books")
async def process_books_data(request: dict):
    """
    处理图书数据：格式化、向量化、去重
    """
    records = request.get('records', [])
    collection_name = request.get('collection_name', 'books')
    
    logger.info(f"开始处理 {len(records)} 条图书数据，集合: {collection_name}")
    
    try:
        if not records:
            logger.warning("没有提供数据记录")
            raise HTTPException(status_code=400, detail="没有提供数据记录")
        
        # 第一步：批量检查哪些记录已存在，避免重复向量化
        logger.info("开始检查已存在的记录...")
        all_md5s = []
        record_data = []
        
        # 准备所有记录的数据和MD5
        for i, record in enumerate(records):
            # 使用分号分隔的格式
            formatted_text = record.get('formatted_text', '')
            if not formatted_text:
                # 如果没有格式化文本，则创建一个
                formatted_text = f"{record.get('title', '')};{record.get('author', '')};{record.get('publisher', '')}"
            
            # 准备元数据
            metadata = {
                'title': record.get('title', ''),
                'author': record.get('author', ''),
                'publisher': record.get('publisher', ''),
                'md5': record.get('md5', ''),
                'original_index': record.get('original_index', i)
            }
            
            all_md5s.append(metadata['md5'])
            record_data.append({
                'formatted_text': formatted_text,
                'metadata': metadata
            })
        
        # 批量检查MD5是否存在
        logger.info(f"批量检查 {len(all_md5s)} 个MD5哈希...")
        md5_exists_map = await vector_store.check_batch_md5_exists(all_md5s, collection_name)
        
        # 过滤出需要向量化的记录
        texts_to_embed = []
        metadatas_to_embed = []
        existing_count = 0
        
        for record_item in record_data:
            metadata = record_item['metadata']
            if md5_exists_map.get(metadata['md5'], False):
                existing_count += 1
                logger.debug(f"跳过已存在记录: {metadata['md5'][:8]}...")
            else:
                texts_to_embed.append(record_item['formatted_text'])
                metadatas_to_embed.append(metadata)
        
        logger.info(f"数据检查完成: 总记录 {len(records)} 条，已存在 {existing_count} 条，需要向量化 {len(texts_to_embed)} 条")
        
        if len(texts_to_embed) == 0:
            logger.info("所有记录都已存在，无需向量化")
            return {
                "message": "所有记录都已存在，无需处理",
                "total_processed": len(records),
                "added_count": 0,
                "skipped_count": len(records),
                "document_ids": []
            }
        
        logger.info(f"开始生成 {len(texts_to_embed)} 个向量...")
        logger.info(f"预计分为 {(len(texts_to_embed) + 99) // 100} 个批次，每批100条")
        
        # 只对不存在的记录生成向量
        embeddings = await embedding_service.get_batch_embeddings(texts_to_embed)
        logger.info("向量生成完成，开始存储...")
        
        # 批量存储（这里应该不会有重复，因为已经预过滤了）
        result = await vector_store.batch_add_documents(
            texts=texts_to_embed,
            embeddings=embeddings,
            metadatas=metadatas_to_embed,
            collection_name=collection_name
        )
        
        # 更新结果统计，包括预检查跳过的记录
        result['total_processed'] = len(records)
        result['skipped_count'] = result['skipped_count'] + existing_count
        
        logger.info(f"图书数据处理完成: 总处理 {result['total_processed']} 条，添加 {result['added_count']} 条，跳过 {result['skipped_count']} 条")
        
        return {
            "message": "图书数据处理完成",
            "total_processed": result['total_processed'],
            "added_count": result['added_count'],
            "skipped_count": result['skipped_count'],
            "document_ids": result['document_ids']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理图书数据失败: {str(e)}")
        logger.error(f"记录数量: {len(records)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"处理图书数据失败: {str(e)}")


@app.post("/find-duplicates")
async def find_duplicates(request: dict):
    """
    查找重复的图书记录
    """
    try:
        records = request.get('records', [])
        collection_name = request.get('collection_name', 'books')
        similarity_threshold = request.get('similarity_threshold', 0.85)
        
        if not records:
            raise HTTPException(status_code=400, detail="没有提供数据记录")
        
        duplicate_groups = []
        processed_indices = set()
        
        for i, record in enumerate(records):
            if i in processed_indices:
                continue
                
            # 搜索相似记录
            formatted_text = record.get('formatted_text', '')
            if not formatted_text:
                formatted_text = f"{record.get('title', '')};{record.get('author', '')};{record.get('publisher', '')}"
            
            # 生成查询向量
            query_embedding = await embedding_service.get_embedding(formatted_text)
            
            # 搜索相似文档
            similar_results = await vector_store.search_similar(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=20,
                min_similarity=similarity_threshold
            )
            
            # 过滤出真正相似的记录
            duplicates = []
            for result in similar_results:
                if result['similarity'] >= similarity_threshold:
                    # 查找原始记录索引
                    original_index = result['metadata'].get('original_index')
                    if original_index is not None and original_index != i:
                        duplicates.append({
                            'record': records[original_index] if original_index < len(records) else {},
                            'similarity': result['similarity'],
                            'original_index': original_index
                        })
                        processed_indices.add(original_index)
            
            if duplicates:
                duplicate_groups.append({
                    'original': record,
                    'original_index': i,
                    'duplicates': duplicates,
                    'total_count': len(duplicates) + 1
                })
                processed_indices.add(i)
        
        return {
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "processed_records": len(processed_indices)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找重复记录失败: {str(e)}")


@app.get("/collection-stats/{collection_name}")
async def get_collection_stats(collection_name: str):
    """
    获取集合统计信息
    """
    try:
        stats = await vector_store.get_collection_stats(collection_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.delete("/clear-collection/{collection_name}")
async def clear_collection(collection_name: str):
    """
    清空集合
    """
    try:
        success = await vector_store.clear_collection(collection_name)
        if success:
            return {"message": f"集合 {collection_name} 已清空"}
        else:
            raise HTTPException(status_code=500, detail="清空集合失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空集合失败: {str(e)}")


@app.get("/collections/{collection_name}/documents")
async def get_collection_documents(
    collection_name: str, 
    page: int = 1, 
    page_size: int = 50,
    search: str = None
):
    """
    获取集合中的文档数据，支持分页和搜索
    """
    try:
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 500:
            page_size = 50
            
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 获取文档
        documents = await vector_store.get_collection_documents(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            search_text=search
        )
        
        # 获取总数
        total_count = await vector_store.get_collection_count(collection_name, search_text=search)
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "documents": documents,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "search": search
        }
        
    except Exception as e:
        logger.error(f"获取集合文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取集合文档失败: {str(e)}")


@app.delete("/collections/{collection_name}/documents/{document_id}")
async def delete_collection_document(collection_name: str, document_id: str):
    """
    删除集合中的指定文档
    """
    try:
        success = await vector_store.delete_document(document_id, collection_name)
        if success:
            return {"message": f"文档 {document_id} 已成功删除"}
        else:
            raise HTTPException(status_code=404, detail="文档未找到")
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


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