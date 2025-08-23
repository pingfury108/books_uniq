# Books Uniq - 文本向量化与相似性搜索API

一个基于FastAPI的文本向量化和相似性搜索服务，使用OpenAI兼容接口调用火山引擎的向量化服务，并通过ChromaDB存储和搜索向量数据。

## 功能特性

- **文本向量化**: 使用火山引擎的嵌入模型将文本转换为向量
- **向量存储**: 基于ChromaDB的持久化向量数据库
- **相似性搜索**: 高效的向量相似性搜索
- **批量处理**: 支持批量文本嵌入和处理
- **RESTful API**: 完整的REST API接口
- **配置管理**: 基于环境变量的配置系统

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd books_uniq

# 安装依赖 (使用rye)
rye sync

# 或使用pip
pip install -e .
```

### 2. 环境变量配置

复制示例配置文件并编辑：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的环境变量：

```bash
# 火山引擎配置
VOLCENGINE_API_KEY=your_api_key_here
VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
EMBEDDING_MODEL=volcengine/doubao-embedding

# ChromaDB配置
CHROMA_PERSIST_DIRECTORY=./chroma_db

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
```

### 3. 启动服务

```bash
# 使用启动脚本
python run.py

# 或直接运行
python -m uvicorn src.books_uniq.main:app --host 0.0.0.0 --port 8000
```

### 4. API文档

启动服务后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API接口

### 嵌入文本

**POST** `/embed`

```json
{
  "text": "这是一个关于高考语文的文本内容",
  "metadata": {
    "subject": "语文",
    "year": "2025",
    "publisher": "教育科学出版社"
  },
  "collection_name": "books"
}
```

### 批量嵌入文本

**POST** `/embed/batch`

```json
{
  "texts": [
    "高考语文第一篇文章内容",
    "高考语文第二篇文章内容"
  ],
  "metadatas": [
    {"chapter": 1, "subject": "语文"},
    {"chapter": 2, "subject": "语文"}
  ],
  "collection_name": "books"
}
```

### 搜索相似文本

**POST** `/search`

```json
{
  "text": "查找关于语文的相关内容",
  "collection_name": "books",
  "n_results": 5
}
```

### 其他接口

- **GET** `/collections` - 列出所有集合
- **GET** `/collections/{collection_name}/info` - 获取集合信息
- **DELETE** `/documents/{document_id}` - 删除指定文档

## 项目结构

```
books_uniq/
├── src/
│   └── books_uniq/
│       ├── __init__.py
│       ├── main.py              # FastAPI应用主文件
│       ├── config.py            # 配置管理
│       └── services/
│           ├── __init__.py
│           ├── embedding_service.py   # 向量化服务
│           └── vector_store.py        # 向量存储服务
├── .env.example                # 环境变量示例
├── run.py                     # 启动脚本
├── pyproject.toml            # 项目配置
└── README.md                 # 项目说明
```

## 配置说明

### 火山引擎配置

- `VOLCENGINE_API_KEY`: 火山引擎API密钥 (必需)
- `VOLCENGINE_BASE_URL`: API基础URL
- `EMBEDDING_MODEL`: 使用的嵌入模型

### ChromaDB配置

- `CHROMA_PERSIST_DIRECTORY`: 向量数据库持久化目录

### API配置

- `API_HOST`: 服务监听地址
- `API_PORT`: 服务端口
- `API_DEBUG`: 是否启用调试模式

## 使用示例

### Python客户端示例

```python
import requests

# 嵌入文本
response = requests.post("http://localhost:8000/embed", json={
    "text": "2025版（5.3）高考A版新高考版 语文",
    "metadata": {
        "subject": "语文",
        "publisher": "教育科学出版社"
    }
})
print(response.json())

# 搜索相似文本
response = requests.post("http://localhost:8000/search", json={
    "text": "高考语文相关内容",
    "n_results": 3
})
print(response.json())
```

## 开发

### 安装开发依赖

```bash
rye sync --no-lock
```

### 运行测试

```bash
# 待实现
pytest
```

## 许可证

MIT License
