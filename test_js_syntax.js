// 测试JavaScript语法
let currentFile = null;
let processResults = [];
let currentTab = 'all';

// 文件上传处理
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');

// 拖拽上传
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.name.match(/\.(xlsx|xls)$/)) {
        alert('请选择Excel文件 (.xlsx 或 .xls)');
        return;
    }
    
    currentFile = file;
    fileName.textContent = file.name;
    fileInfo.style.display = 'block';
}

// 处理文件
async function processFile() {
    if (!currentFile) return;

    // 显示进度区域
    document.getElementById('progress-section').style.display = 'block';
    document.getElementById('process-btn').disabled = true;
    
    // 重置进度
    updateProgress('embedding', 0, '准备上传文件...');
    updateProgress('dedup', 0, '等待向量化完成...');
    addLog('开始处理文件: ' + currentFile.name);

    try {
        // 1. 上传并解析Excel文件
        const formData = new FormData();
        formData.append('file', currentFile);
        
        addLog('正在上传和解析Excel文件...');
        const uploadResponse = await fetch('/upload-excel', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error('文件上传失败');
        }
        
        const uploadData = await uploadResponse.json();
        addLog(`文件解析完成，共发现 ${uploadData.total_records} 条记录`);
        
        // 2. 批量向量化
        await performEmbedding(uploadData.records);
        
        // 3. 相似性搜索和去重
        await performDeduplication(uploadData.records);
        
    } catch (error) {
        addLog(`处理失败: ${error.message}`, 'error');
        alert('处理失败: ' + error.message);
    } finally {
        document.getElementById('process-btn').disabled = false;
    }
}

// 向量化处理
async function performEmbedding(records) {
    updateProgress('embedding', 0, '正在检查已存在记录...');
    addLog(`开始检查 ${records.length} 条记录是否已存在...`);
    
    // 预估时间现在包括检查步骤（检查时间很短，主要是向量化时间）
    const batchSize = 100;
    // 注意：实际需要向量化的数量会在检查后确定
    const maxExpectedBatches = Math.ceil(records.length / batchSize);
    const delaySeconds = 3.0; // 与后端保持一致
    const maxEstimatedMinutes = (maxExpectedBatches * delaySeconds) / 60;
    
    if (maxExpectedBatches > 1) {
        addLog(`遵守API限制：每批 ${batchSize} 条，最多 ${maxExpectedBatches} 批次`, 'info');
        addLog(`预计最大耗时约 ${maxEstimatedMinutes.toFixed(1)} 分钟（如果全部需要向量化）`, 'warning');
        addLog(`如有已存在记录将大幅减少处理时间`, 'info');
    }
    
    try {
        // 使用新的process-books接口
        const startTime = Date.now();
        const response = await fetch('/process-books', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                records: records,
                collection_name: 'books'
            })
        });
        
        if (!response.ok) {
            const errorMessage = await handleApiError(response, '向量化处理');
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        const actualMinutes = (Date.now() - startTime) / 60000;
        
        updateProgress('embedding', 100, '向量化完成');
        addLog(`向量化完成: 添加 ${result.added_count} 条，跳过重复 ${result.skipped_count} 条`, 'success');
        addLog(`实际耗时 ${actualMinutes.toFixed(1)} 分钟`, 'info');
        
        if (result.skipped_count > 0) {
            addLog(`跳过 ${result.skipped_count} 条重复记录（基于MD5去重）`, 'warning');
        }
        
        // 显示API使用统计
        if (maxExpectedBatches > 1) {
            const avgTimePerBatch = actualMinutes * 60 / maxExpectedBatches;
            addLog(`平均每批次耗时 ${avgTimePerBatch.toFixed(1)} 秒`, 'info');
        }
        
        return result;
        
    } catch (error) {
        addLog(`向量化处理失败: ${error.message}`, 'error');
        throw error;
    }
}