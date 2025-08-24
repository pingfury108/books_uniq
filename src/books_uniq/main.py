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


# 用于跟踪处理进度的全局状态
processing_status = {
    "is_processing": False,
    "current_batch": 0,
    "total_batches": 0,
    "processed_records": 0,
    "total_records": 0,
    "added_count": 0,
    "skipped_count": 0,
    "current_operation": "",
    "start_time": None,
    "estimated_completion": None,
    "task_id": None,
    "completed": False,
    "error": None,
    "final_result": None
}


@app.get("/processing-status")
async def get_processing_status():
    """
    获取当前处理状态
    """
    status = processing_status.copy()
    
    if status["is_processing"] and status["start_time"]:
        import time
        elapsed = time.time() - status["start_time"]
        if status["current_batch"] > 0:
            # 估算剩余时间
            avg_time_per_batch = elapsed / status["current_batch"]
            remaining_batches = status["total_batches"] - status["current_batch"]
            estimated_remaining = avg_time_per_batch * remaining_batches
            status["estimated_remaining_seconds"] = int(estimated_remaining)
        
        status["elapsed_seconds"] = int(elapsed)
    
    return status


@app.post("/start-processing")
async def start_processing(request: dict):
    """
    启动异步处理任务，立即返回任务ID
    """
    global processing_status
    
    # 检查是否已有任务在进行
    if processing_status["is_processing"]:
        raise HTTPException(status_code=409, detail="已有处理任务正在进行中，请稍后再试")
    
    records = request.get('records', [])
    collection_name = request.get('collection_name', 'books')
    batch_size = request.get('batch_size', 50)
    
    if not records:
        raise HTTPException(status_code=400, detail="没有提供数据记录")
    
    # 生成任务ID
    import uuid
    import time
    task_id = f"task_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    # 初始化任务状态
    processing_status.update({
        "is_processing": True,
        "current_batch": 0,
        "total_batches": (len(records) + batch_size - 1) // batch_size,
        "processed_records": 0,
        "total_records": len(records),
        "added_count": 0,
        "skipped_count": 0,
        "current_operation": "任务已启动",
        "start_time": time.time(),
        "task_id": task_id,
        "completed": False,
        "error": None,
        "final_result": None
    })
    
    # 启动后台任务
    import asyncio
    asyncio.create_task(process_books_async(records, collection_name, batch_size))
    
    logger.info(f"异步任务已启动: {task_id}, 总记录数: {len(records)}")
    
    return {
        "task_id": task_id,
        "message": "处理任务已启动",
        "total_records": len(records),
        "total_batches": processing_status["total_batches"],
        "batch_size": batch_size
    }


async def process_books_async(records: list, collection_name: str, batch_size: int):
    """
    后台异步处理函数
    """
    global processing_status
    
    try:
        logger.info(f"开始后台流式处理 {len(records)} 条图书数据，集合: {collection_name}, 批大小: {batch_size}")
        
        # 统计信息
        total_processed = 0
        total_added = 0
        total_skipped = 0
        all_document_ids = []
        
        # 分批处理记录
        total_batches = processing_status["total_batches"]
        
        for batch_num in range(0, len(records), batch_size):
            batch_records = records[batch_num:batch_num + batch_size]
            current_batch = (batch_num // batch_size) + 1
            
            # 更新进度状态
            processing_status.update({
                "current_batch": current_batch,
                "current_operation": f"处理批次 {current_batch}/{total_batches}"
            })
            
            logger.info(f"处理批次 {current_batch}/{total_batches}，记录 {batch_num+1}-{min(batch_num+batch_size, len(records))}")
            
            # 准备当前批次的数据
            batch_texts = []
            batch_metadatas = []
            batch_existing_count = 0
            
            processing_status["current_operation"] = f"检查批次 {current_batch} 中的重复记录"
            
            for i, record in enumerate(batch_records):
                # 使用分号分隔的格式
                formatted_text = record.get('formatted_text', '')
                if not formatted_text:
                    formatted_text = f"{record.get('title', '')};{record.get('author', '')};{record.get('publisher', '')}"
                
                # 准备元数据
                metadata = {
                    'title': record.get('title', ''),
                    'author': record.get('author', ''),
                    'publisher': record.get('publisher', ''),
                    'md5': record.get('md5', ''),
                    'original_index': record.get('original_index', batch_num + i),
                    'batch_number': current_batch
                }
                
                # 检查当前记录是否已存在
                if await vector_store.check_md5_exists(metadata['md5'], collection_name):
                    batch_existing_count += 1
                    logger.debug(f"跳过已存在记录: {metadata['md5'][:8]}...")
                else:
                    batch_texts.append(formatted_text)
                    batch_metadatas.append(metadata)
            
            # 处理当前批次的新记录
            if batch_texts:
                logger.info(f"批次 {current_batch}: 需要向量化 {len(batch_texts)} 条，跳过 {batch_existing_count} 条")
                
                # 生成向量（分批）
                processing_status["current_operation"] = f"为批次 {current_batch} 生成 {len(batch_texts)} 个向量"
                logger.debug(f"为批次 {current_batch} 生成 {len(batch_texts)} 个向量...")
                batch_embeddings = await embedding_service.get_batch_embeddings(batch_texts)
                
                # 立即写入数据库
                processing_status["current_operation"] = f"将批次 {current_batch} 写入数据库"
                logger.debug(f"将批次 {current_batch} 的数据写入数据库...")
                batch_result = await vector_store.batch_add_documents(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    collection_name=collection_name
                )
                
                # 更新统计信息
                total_added += batch_result['added_count']
                total_skipped += batch_result['skipped_count'] + batch_existing_count
                all_document_ids.extend(batch_result['document_ids'])
                
                # 更新全局进度
                processing_status.update({
                    "added_count": total_added,
                    "skipped_count": total_skipped
                })
                
                logger.info(f"批次 {current_batch} 完成: 添加 {batch_result['added_count']} 条")
            else:
                logger.info(f"批次 {current_batch}: 所有 {batch_existing_count} 条记录都已存在，跳过")
                total_skipped += batch_existing_count
                processing_status["skipped_count"] = total_skipped
            
            total_processed += len(batch_records)
            processing_status["processed_records"] = total_processed
            
            # 进度日志
            progress = (current_batch / total_batches) * 100
            logger.info(f"总体进度: {progress:.1f}% ({total_processed}/{len(records)}) - 已添加: {total_added}, 已跳过: {total_skipped}")
        
        # 处理完成
        processing_status.update({
            "current_operation": "处理完成",
            "completed": True,
            "final_result": {
                "message": "图书数据流式处理完成",
                "total_processed": total_processed,
                "added_count": total_added,
                "skipped_count": total_skipped,
                "document_ids": all_document_ids,
                "processing_info": {
                    "total_batches": total_batches,
                    "batch_size": batch_size,
                    "streaming_mode": True
                }
            }
        })
        
        logger.info(f"后台流式处理完成: 总处理 {total_processed} 条，添加 {total_added} 条，跳过 {total_skipped} 条")
        
    except Exception as e:
        # 处理失败
        processing_status.update({
            "is_processing": False,
            "current_operation": f"处理失败: {str(e)}",
            "error": str(e),
            "completed": True
        })
        
        logger.error(f"后台处理失败: {str(e)}")
        logger.error(f"记录数量: {len(records)}")
        logger.error(traceback.format_exc())
    
    finally:
        # 标记处理完成
        processing_status["is_processing"] = False


# 保留原来的同步接口作为备用
@app.post("/process-books")
async def process_books_data(request: dict):
    """
    处理图书数据：格式化、向量化、去重（流式处理，分批写入）
    """
    global processing_status
    
    # 检查是否已有任务在进行
    if processing_status["is_processing"]:
        raise HTTPException(status_code=409, detail="已有处理任务正在进行中，请稍后再试")
    
    records = request.get('records', [])
    collection_name = request.get('collection_name', 'books')
    batch_size = request.get('batch_size', 50)  # 每批处理50条记录
    
    # 初始化进度状态
    import time
    processing_status.update({
        "is_processing": True,
        "current_batch": 0,
        "total_batches": (len(records) + batch_size - 1) // batch_size,
        "processed_records": 0,
        "total_records": len(records),
        "added_count": 0,
        "skipped_count": 0,
        "current_operation": "开始处理",
        "start_time": time.time(),
        "estimated_completion": None
    })
    
    logger.info(f"开始流式处理 {len(records)} 条图书数据，集合: {collection_name}, 批大小: {batch_size}")
    
    try:
        if not records:
            logger.warning("没有提供数据记录")
            raise HTTPException(status_code=400, detail="没有提供数据记录")
        
        # 统计信息
        total_processed = 0
        total_added = 0
        total_skipped = 0
        all_document_ids = []
        
        # 分批处理记录
        total_batches = processing_status["total_batches"]
        
        for batch_num in range(0, len(records), batch_size):
            batch_records = records[batch_num:batch_num + batch_size]
            current_batch = (batch_num // batch_size) + 1
            
            # 更新进度状态
            processing_status.update({
                "current_batch": current_batch,
                "current_operation": f"处理批次 {current_batch}/{total_batches}"
            })
            
            logger.info(f"处理批次 {current_batch}/{total_batches}，记录 {batch_num+1}-{min(batch_num+batch_size, len(records))}")
            
            # 准备当前批次的数据
            batch_texts = []
            batch_metadatas = []
            batch_existing_count = 0
            
            processing_status["current_operation"] = f"检查批次 {current_batch} 中的重复记录"
            
            for i, record in enumerate(batch_records):
                # 使用分号分隔的格式
                formatted_text = record.get('formatted_text', '')
                if not formatted_text:
                    formatted_text = f"{record.get('title', '')};{record.get('author', '')};{record.get('publisher', '')}"
                
                # 准备元数据
                metadata = {
                    'title': record.get('title', ''),
                    'author': record.get('author', ''),
                    'publisher': record.get('publisher', ''),
                    'md5': record.get('md5', ''),
                    'original_index': record.get('original_index', batch_num + i),
                    'batch_number': current_batch
                }
                
                # 检查当前记录是否已存在
                if await vector_store.check_md5_exists(metadata['md5'], collection_name):
                    batch_existing_count += 1
                    logger.debug(f"跳过已存在记录: {metadata['md5'][:8]}...")
                else:
                    batch_texts.append(formatted_text)
                    batch_metadatas.append(metadata)
            
            # 处理当前批次的新记录
            if batch_texts:
                logger.info(f"批次 {current_batch}: 需要向量化 {len(batch_texts)} 条，跳过 {batch_existing_count} 条")
                
                # 生成向量（分批）
                processing_status["current_operation"] = f"为批次 {current_batch} 生成 {len(batch_texts)} 个向量"
                logger.debug(f"为批次 {current_batch} 生成 {len(batch_texts)} 个向量...")
                batch_embeddings = await embedding_service.get_batch_embeddings(batch_texts)
                
                # 立即写入数据库
                processing_status["current_operation"] = f"将批次 {current_batch} 写入数据库"
                logger.debug(f"将批次 {current_batch} 的数据写入数据库...")
                batch_result = await vector_store.batch_add_documents(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    collection_name=collection_name
                )
                
                # 更新统计信息
                total_added += batch_result['added_count']
                total_skipped += batch_result['skipped_count'] + batch_existing_count
                all_document_ids.extend(batch_result['document_ids'])
                
                # 更新全局进度
                processing_status.update({
                    "added_count": total_added,
                    "skipped_count": total_skipped
                })
                
                logger.info(f"批次 {current_batch} 完成: 添加 {batch_result['added_count']} 条")
            else:
                logger.info(f"批次 {current_batch}: 所有 {batch_existing_count} 条记录都已存在，跳过")
                total_skipped += batch_existing_count
                processing_status["skipped_count"] = total_skipped
            
            total_processed += len(batch_records)
            processing_status["processed_records"] = total_processed
            
            # 进度日志
            progress = (current_batch / total_batches) * 100
            logger.info(f"总体进度: {progress:.1f}% ({total_processed}/{len(records)}) - 已添加: {total_added}, 已跳过: {total_skipped}")
        
        processing_status["current_operation"] = "处理完成"
        logger.info(f"流式处理完成: 总处理 {total_processed} 条，添加 {total_added} 条，跳过 {total_skipped} 条")
        
        result = {
            "message": "图书数据流式处理完成",
            "total_processed": total_processed,
            "added_count": total_added,
            "skipped_count": total_skipped,
            "document_ids": all_document_ids,
            "processing_info": {
                "total_batches": total_batches,
                "batch_size": batch_size,
                "streaming_mode": True
            }
        }
        
        # 重置处理状态
        processing_status.update({
            "is_processing": False,
            "current_operation": "完成",
            "processed_records": total_processed,
            "added_count": total_added,
            "skipped_count": total_skipped
        })
        
        return result
        
    except HTTPException:
        # HTTP异常直接抛出，同时重置状态
        processing_status["is_processing"] = False
        processing_status["current_operation"] = "处理异常"
        raise
    except Exception as e:
        # 重置处理状态
        processing_status.update({
            "is_processing": False,
            "current_operation": f"处理失败: {str(e)}"
        })
        
        logger.error(f"处理图书数据失败: {str(e)}")
        logger.error(f"记录数量: {len(records)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"处理图书数据失败: {str(e)}")


@app.post("/find-duplicates")
async def find_duplicates(request: dict):
    """
    查找重复的图书记录（增强版：结合相似度、作者、出版社多重判定）
    """
    try:
        records = request.get('records', [])
        collection_name = request.get('collection_name', 'books')
        similarity_threshold = request.get('similarity_threshold', 0.85)
        
        if not records:
            raise HTTPException(status_code=400, detail="没有提供数据记录")
        
        duplicate_groups = []
        processed_indices = set()
        
        def is_author_similar(author1, author2):
            """判断作者是否相同或高度相似"""
            if not author1 or not author2:
                return False
            
            # 去除空格和特殊字符后比较
            author1_clean = ''.join(author1.lower().split())
            author2_clean = ''.join(author2.lower().split())
            
            # 完全相同
            if author1_clean == author2_clean:
                return True
            
            # 包含关系（一个作者包含另一个作者）
            if author1_clean in author2_clean or author2_clean in author1_clean:
                return True
            
            # 编辑距离相似性（对于短字符串）
            if len(author1_clean) > 3 and len(author2_clean) > 3:
                try:
                    from Levenshtein import ratio
                    similarity = ratio(author1_clean, author2_clean)
                    return similarity >= 0.8
                except ImportError:
                    # 如果没有安装Levenshtein，使用简单的字符串相似度
                    # 计算Jaccard相似度作为备选
                    set1 = set(author1_clean)
                    set2 = set(author2_clean)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    if union > 0:
                        jaccard_similarity = intersection / union
                        return jaccard_similarity >= 0.7
                    return False
            
            return False
        
        def get_match_type(original_record, duplicate_record):
            """获取匹配类型和置信度"""
            author_match = is_author_similar(
                original_record.get('author', ''), 
                duplicate_record.get('author', '')
            )
            
            publisher_match = (
                original_record.get('publisher', '').strip().lower() == 
                duplicate_record.get('publisher', '').strip().lower()
            )
            
            if author_match and publisher_match:
                return "author_publisher_match", 1.0
            elif author_match:
                return "author_match", 0.9
            elif publisher_match:
                return "publisher_match", 0.8
            else:
                return "similarity_only", 0.7
        
        # 导入hashlib用于MD5计算
        import hashlib
        
        logger.info("开始逐个处理记录进行去重分析...")
        
        # 统计信息
        reused_embeddings_count = 0
        new_embeddings_count = 0
        
        for i, record in enumerate(records):
            if i in processed_indices:
                continue
                
            # 搜索相似记录
            formatted_text = record.get('formatted_text', '')
            if not formatted_text:
                formatted_text = f"{record.get('title', '')};{record.get('author', '')};{record.get('publisher', '')}"
            
            # 生成MD5
            md5_hash = hashlib.md5(formatted_text.encode('utf-8')).hexdigest()
            
            # 尝试从数据库获取已存在的嵌入向量
            query_embedding = await vector_store.get_embedding_by_md5(md5_hash, collection_name)
            
            if query_embedding is None:
                # 如果数据库中没有，则计算新的嵌入向量
                logger.debug(f"为记录 {i} 生成新的嵌入向量 (MD5: {md5_hash[:8]}...)")
                query_embedding = await embedding_service.get_embedding(formatted_text)
                new_embeddings_count += 1
            else:
                logger.debug(f"复用记录 {i} 的现有嵌入向量 (MD5: {md5_hash[:8]}...)")
                reused_embeddings_count += 1
            
            # 搜索相似文档
            similar_results = await vector_store.search_similar(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=30,  # 增加查询数量以便更好过滤
                min_similarity=0.85  # 使用统一的相似度阈值
            )
            
            # 多重条件过滤
            qualified_duplicates = []
            for result in similar_results:
                # 基础相似度要求
                if result['similarity'] < similarity_threshold:
                    continue
                    
                # 查找原始记录索引
                original_index = result['metadata'].get('original_index')
                if original_index is None or original_index == i or original_index in processed_indices:
                    continue
                
                # 获取对应的原始记录
                duplicate_record = records[original_index] if original_index < len(records) else {}
                
                # 多重条件判定
                match_type, confidence = get_match_type(record, duplicate_record)
                
                # 只保留符合条件的重复记录
                if match_type != "similarity_only":  # 排除仅相似度匹配的情况
                    qualified_duplicates.append({
                        'record': duplicate_record,
                        'similarity': result['similarity'],
                        'original_index': original_index,
                        'match_type': match_type,
                        'confidence': confidence,
                        'match_reason': f"相似度: {result['similarity']:.3f}, 匹配类型: {match_type}"
                    })
            
            # 按置信度排序
            qualified_duplicates.sort(key=lambda x: x['confidence'], reverse=True)
            
            if qualified_duplicates:
                # 标记所有相关索引为已处理
                for dup in qualified_duplicates:
                    processed_indices.add(dup['original_index'])
                
                duplicate_groups.append({
                    'original': record,
                    'original_index': i,
                    'duplicates': qualified_duplicates,
                    'total_count': len(qualified_duplicates) + 1,
                    'primary_match_type': qualified_duplicates[0]['match_type'] if qualified_duplicates else "none",
                    'average_confidence': sum(d['confidence'] for d in qualified_duplicates) / len(qualified_duplicates) if qualified_duplicates else 0
                })
                processed_indices.add(i)
        
        logger.info(f"去重分析完成 - 复用向量: {reused_embeddings_count}, 新生成向量: {new_embeddings_count}")
        
        return {
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "processed_records": len(processed_indices),
            "matching_criteria": {
                "min_similarity": similarity_threshold,
                "require_author_or_publisher": True,
                "matching_strategy": "multi-criteria (similarity + author/publisher)"
            },
            "performance_stats": {
                "reused_embeddings": reused_embeddings_count,
                "new_embeddings": new_embeddings_count,
                "efficiency_ratio": f"{reused_embeddings_count}/{reused_embeddings_count + new_embeddings_count}"
            }
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


@app.get("/collections/{collection_name}/documents/{document_id}/metadata")
async def get_document_metadata(collection_name: str, document_id: str):
    """
    获取单个文档的元数据
    """
    try:
        metadata = await vector_store.get_document_metadata(document_id, collection_name)
        return metadata
    except Exception as e:
        logger.error(f"获取文档元数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档元数据失败: {str(e)}")


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
        reload=settings.api_debug
    )