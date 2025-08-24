# Books Uniq - 文本向量化与相似性搜索API

一个基于FastAPI的文本向量化和相似性搜索服务，使用OpenAI兼容接口调用火山引擎的向量化服务，并通过ChromaDB存储和搜索向量数据。

## 功能特性

- **文本向量化**: 使用火山引擎的嵌入模型将文本转换为向量
- **向量存储**: 基于ChromaDB的持久化向量数据库
- **相似性搜索**: 高效的向量相似性搜索
- **批量处理**: 支持批量文本嵌入和处理
- **Excel处理**: 支持Excel文件上传和图书数据去重
- **异步处理**: 支持大批量数据的异步流式处理
- **RESTful API**: 完整的REST API接口和Web界面
- **Docker支持**: 完整的容器化部署方案
- **配置管理**: 基于环境变量的配置系统

## 快速开始

### 方式1: Docker部署（推荐）

```bash
# 使用docker-compose启动
docker-compose up -d

# 或者直接运行Docker容器
docker run -d \
  -p 8098:8098 \
  -v $(pwd)/data:/app/data \
  -e VOLCENGINE_API_KEY=your_api_key \
  your_dockerhub_username/books-uniq:latest
```

### 方式2: 本地开发

```bash
# 克隆项目
git clone <repository-url>
cd books_uniq

# 安装依赖 (使用rye)
rye sync

# 或使用pip
pip install -e .
```

### 环境变量配置

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
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# API配置
API_HOST=0.0.0.0
API_PORT=8098
API_DEBUG=false
```

### 启动服务

```bash
# 使用启动脚本
python run.py

# 或直接运行
python -m uvicorn src.books_uniq.main:app --host 0.0.0.0 --port 8098
```

## Docker构建和部署

### 构建镜像

```bash
# 构建Docker镜像
docker build -t books-uniq:latest .

# 多平台构建
docker buildx build --platform linux/amd64,linux/arm64 -t books-uniq:latest .
```

### 推送到仓库

```bash
# 登录Docker Hub
docker login

# 标记镜像
docker tag books-uniq:latest your_dockerhub_username/books-uniq:latest

# 推送镜像
docker push your_dockerhub_username/books-uniq:latest
```

### CI/CD自动构建

项目集成了GitHub Actions自动构建和推送：

1. 在GitHub仓库设置中添加以下Secrets：
   - `DOCKERHUB_USERNAME`: Docker Hub用户名
   - `DOCKERHUB_TOKEN`: Docker Hub访问令牌

2. 推送代码到main分支将自动触发构建和推送

## API文档

启动服务后，访问以下地址：

- **Web界面**: http://localhost:8098
- **Swagger UI**: http://localhost:8098/docs  
- **ReDoc**: http://localhost:8098/redoc

## 主要API接口

### 异步处理接口

**POST** `/start-processing` - 启动异步处理任务
**GET** `/processing-status` - 获取处理进度

### 文档处理接口

**POST** `/upload-excel` - 上传Excel文件
**POST** `/process-books` - 处理图书数据
**POST** `/find-duplicates` - 查找重复记录

### 向量化接口

**POST** `/embed` - 单个文本嵌入
**POST** `/embed/batch` - 批量文本嵌入
**POST** `/search` - 相似性搜索

### 数据管理接口

**GET** `/collections` - 列出所有集合
**GET** `/collections/{name}/info` - 获取集合信息
**DELETE** `/collections/{name}/documents/{id}` - 删除文档

## 项目结构

```
books_uniq/
├── src/
│   └── books_uniq/
│       ├── main.py                    # FastAPI应用主文件
│       ├── config.py                  # 配置管理
│       ├── services/                  # 服务层
│       │   ├── embedding_service.py   # 向量化服务
│       │   ├── vector_store.py        # 向量存储服务
│       │   └── excel_processor.py     # Excel处理服务
│       ├── templates/                 # Web模板
│       └── static/                    # 静态资源
├── .github/workflows/                 # GitHub Actions
├── Dockerfile                         # Docker构建文件
├── docker-compose.yml               # Docker Compose配置
├── .dockerignore                     # Docker忽略文件
├── .env.example                      # 环境变量示例
├── requirements.lock                 # 依赖锁定文件
└── README.md                         # 项目说明
```

## 使用示例

### Python客户端示例

```python
import requests

# 上传Excel文件
with open('books.xlsx', 'rb') as f:
    response = requests.post("http://localhost:8098/upload-excel", 
                           files={'file': f})
    print(response.json())

# 启动异步处理
response = requests.post("http://localhost:8098/start-processing", json={
    "records": [...],  # Excel解析出的记录
    "collection_name": "books"
})
task_id = response.json()['task_id']

# 监控处理进度
while True:
    status = requests.get("http://localhost:8098/processing-status").json()
    if status['completed']:
        break
    print(f"进度: {status['current_batch']}/{status['total_batches']}")
```

## 开发和贡献

### 本地开发

```bash
# 安装开发依赖
rye sync

# 启动开发服务器
python run.py
```

### Docker开发环境

```bash
# 启动开发环境
docker-compose up -d

# 查看日志
docker-compose logs -f books-uniq
```

## 许可证

MIT License
