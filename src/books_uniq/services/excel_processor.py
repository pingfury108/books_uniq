import pandas as pd
import hashlib
import logging
from typing import List, Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)


class ExcelProcessor:
    def __init__(self):
        # 只处理三个核心字段
        self.required_fields = ['title', 'author', 'publisher']
    
    def process_excel_file(self, file_content: bytes) -> Dict[str, Any]:
        """
        处理Excel文件内容，只读取书名、作者、出版社三个字段
        """
        try:
            logger.info("开始读取Excel文件...")
            # 读取Excel文件的第一个工作表
            df = pd.read_excel(BytesIO(file_content), sheet_name=0)
            logger.info(f"Excel文件读取成功，原始形状: {df.shape}")
            logger.info(f"原始列名: {list(df.columns)}")
            
            # 数据清洗和标准化
            logger.info("开始清洗DataFrame...")
            df = self._clean_dataframe(df)
            logger.info(f"清洗后DataFrame形状: {df.shape}")
            logger.info(f"清洗后列名: {list(df.columns)}")
            
            # 转换为记录列表，并生成格式化文本和MD5
            logger.info("开始格式化记录...")
            records = self._format_records(df)
            logger.info(f"记录格式化完成，总数: {len(records)}")
            
            return {
                "total_records": len(records),
                "columns": self.required_fields,
                "records": records,
                "sample_data": records[:5] if records else []  # 返回前5条作为示例
            }
            
        except Exception as e:
            logger.error(f"Excel文件处理过程中发生错误: {str(e)}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"完整堆栈: {traceback.format_exc()}")
            raise Exception(f"Excel文件处理失败: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗和标准化DataFrame，只保留三个核心字段
        """
        try:
            # 删除完全空白的行和列
            df = df.dropna(how='all').dropna(how='all', axis=1)
            
            # 如果DataFrame为空，创建一个包含必需字段的空DataFrame
            if df.empty:
                return pd.DataFrame(columns=self.required_fields)
            
            # 标准化列名
            df.columns = df.columns.astype(str)
            
            # 映射常见的列名到标准格式
            column_mapping = self._get_column_mapping(df.columns.tolist())
            df = df.rename(columns=column_mapping)
            
            # 确保三个核心字段存在
            for field in self.required_fields:
                if field not in df.columns:
                    # 尝试从现有列中查找相似的列名
                    similar_col = self._find_similar_column(field, df.columns.tolist())
                    if similar_col and similar_col in df.columns:
                        df[field] = df[similar_col]
                    else:
                        df[field] = ''  # 创建空列
            
            # 只保留三个核心字段
            df = df[self.required_fields].copy()
            
            # 数据类型转换和清理
            df = self._clean_data_types(df)
            
            return df
            
        except Exception as e:
            # 如果清理失败，返回一个包含必需字段的空DataFrame
            empty_df = pd.DataFrame(columns=self.required_fields)
            # 填充一些示例数据以便调试
            if not df.empty:
                try:
                    # 尝试从原始数据中获取前几行作为调试信息
                    sample_data = df.head(3).to_dict('records')
                    logger.warning(f"DataFrame清理失败，原始数据示例: {sample_data}")
                except:
                    pass
            raise Exception(f"数据清理失败: {str(e)}")
            
    
    def _format_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        将DataFrame格式化为记录列表，使用分号分隔格式并生成MD5
        """
        records = []
        
        for index, row in df.iterrows():
            title = self._clean_pandas_string(str(row.get('title', '')))
            author = self._clean_pandas_string(str(row.get('author', '')))
            publisher = self._clean_pandas_string(str(row.get('publisher', '')))
            
            # 跳过完全空白的记录
            if not title and not author and not publisher:
                continue
            
            # 创建分号分隔的文本格式
            formatted_text = f"{title};{author};{publisher}"
            
            # 生成MD5哈希
            md5_hash = hashlib.md5(formatted_text.encode('utf-8')).hexdigest()
            
            record = {
                'title': title,
                'author': author,
                'publisher': publisher,
                'formatted_text': formatted_text,
                'md5': md5_hash,
                'original_index': index
            }
            
            records.append(record)
        
        return records
    
    def _get_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """
        根据列名内容智能映射到标准字段名
        """
        mapping = {}
        
        # 定义字段识别规则 - 只关注三个核心字段
        field_patterns = {
            'title': ['书名', '标题', '图书名称', '名称', 'title', 'book', 'name', '书'],
            'author': ['作者', '编者', '主编', 'author', 'writer', 'editor', '著者'],
            'publisher': ['出版社', '出版商', 'publisher', 'press', '版社', '出版']
        }
        
        for col in columns:
            col_lower = col.lower().strip()
            for standard_field, patterns in field_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in col_lower or col_lower in pattern.lower():
                        mapping[col] = standard_field
                        break
                if col in mapping:
                    break
        
        return mapping
    
    def _find_similar_column(self, target: str, columns: List[str]) -> str:
        """
        查找与目标列名相似的列
        """
        target_lower = target.lower()
        
        # 定义相似度检查规则
        similarity_rules = {
            'title': ['书', '名', 'title', 'book', '标题'],
            'author': ['者', 'author', 'writer', '编', '著'],
            'publisher': ['版', 'publisher', 'press', '出版']
        }
        
        if target in similarity_rules:
            patterns = similarity_rules[target]
            for col in columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    return col
        
        return None

    def _clean_pandas_string(self, text: str) -> str:
        """
        清理pandas Series的字符串表示，移除类似'Name: 0, dtype: object'的内容
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ''
        
        # 移除pandas特定的字符串模式
        pandas_patterns = [
            r'Name:.*dtype:.*',
            r'dtype:.*',
            r'Name:.*',
            r'Series.*',
            r'DataFrame.*'
        ]
        
        import re
        cleaned_text = text.strip()
        
        for pattern in pandas_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 移除多余的分号和空格
        cleaned_text = re.sub(r';+', ';', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip().strip(';').strip()
        
        return cleaned_text if cleaned_text else ''

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据类型，避免pandas Series的字符串表示形式
        """
        try:
            # 文本字段处理
            for field in self.required_fields:
                if field in df.columns:
                    # 确保字段存在且是Series
                    series = df[field]
                    # 转换为字符串，处理NaN值
                    series = series.astype(str)
                    series = series.fillna('')
                    series = series.replace(['nan', 'None', 'null', 'NaT'], '')
                    # 去除首尾空格和pandas特定的字符串表示
                    series = series.str.strip()
                    # 处理pandas Series的字符串表示（如"Name: 0, dtype: object"）
                    series = series.apply(lambda x: self._clean_pandas_string(x))
                    df[field] = series
            
            # 确保所有必需字段都存在
            for field in self.required_fields:
                if field not in df.columns:
                    df[field] = ''
            
            # 过滤空记录
            mask = (df['title'].astype(str).str.strip() != '') | \
                   (df['author'].astype(str).str.strip() != '') | \
                   (df['publisher'].astype(str).str.strip() != '')
            
            df = df[mask]
            
        except Exception as e:
            # 如果数据清理失败，至少确保有必需的列
            for field in self.required_fields:
                if field not in df.columns:
                    df[field] = ''
            # 将所有数据转为字符串
            for field in self.required_fields:
                df[field] = df[field].astype(str).fillna('').replace(['nan', 'None', 'null', 'NaT'], '')
                df[field] = df[field].apply(lambda x: self._clean_pandas_string(x))
        
        return df
    
    def validate_data(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证数据质量
        """
        total_records = len(records)
        
        if total_records == 0:
            return {
                "valid": False,
                "message": "没有有效的数据记录",
                "statistics": {}
            }
        
        # 统计数据质量
        title_missing = sum(1 for r in records if not r.get('title', '').strip())
        author_missing = sum(1 for r in records if not r.get('author', '').strip())
        publisher_missing = sum(1 for r in records if not r.get('publisher', '').strip())
        
        # 统计MD5重复情况
        md5_hashes = [r['md5'] for r in records]
        unique_hashes = len(set(md5_hashes))
        duplicate_count = total_records - unique_hashes
        
        statistics = {
            "total_records": total_records,
            "unique_records": unique_hashes,
            "duplicate_records": duplicate_count,
            "title_missing": title_missing,
            "author_missing": author_missing,
            "publisher_missing": publisher_missing,
            "complete_records": sum(1 for r in records if all(r.get(field, '').strip() for field in self.required_fields))
        }
        
        # 判断数据是否可用
        valid = unique_hashes > 0 and title_missing < total_records * 0.8
        
        return {
            "valid": valid,
            "message": f"数据验证完成，发现 {duplicate_count} 条重复记录" if valid else "数据质量不佳，可能影响处理效果",
            "statistics": statistics
        }
    
    def get_unique_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据MD5去重，返回唯一记录
        """
        seen_hashes = set()
        unique_records = []
        
        for record in records:
            if record['md5'] not in seen_hashes:
                seen_hashes.add(record['md5'])
                unique_records.append(record)
        
        return unique_records