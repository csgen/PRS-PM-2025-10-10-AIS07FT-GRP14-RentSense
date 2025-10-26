import os
from pathlib import Path
from google.cloud import storage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
current_dir = os.path.dirname(__file__) 
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
sys.path.insert(0, project_root) 
from SystemCode.db.model import Base, ImageRecord
from SystemCode.db.envconfig import get_database_url
from typing import List, Optional, Dict, Any

class SQLAlchemyImageUploader:
    def __init__(self, gcs_key_path: str, bucket_name: str, database_url: str):
        """
        初始化上传器
        
        Args:
            gcs_key_path: GCS 服务账户密钥文件路径
            bucket_name: GCS 存储桶名称
            database_url: 数据库连接URL，例如：postgresql://user:pass@localhost/dbname
        """
        # 初始化GCS客户端
        self.storage_client = storage.Client.from_service_account_json(gcs_key_path)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_name = bucket_name
        
        # 初始化数据库
        self.engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # 创建表
        Base.metadata.create_all(bind=self.engine)
        print("数据库表和连接初始化完成")
    
    def get_db_session(self):
        """获取数据库会话"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def upload_image_to_gcs(self, local_file_path: str, gcs_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """上传图片到GCS并返回相关信息"""
        if gcs_filename is None:
            gcs_filename = os.path.basename(local_file_path)
        
        try:
            # 上传文件
            blob = self.bucket.blob(gcs_filename)
            
            # 检测文件类型
            content_type = self._get_content_type(local_file_path)
            
            blob.upload_from_filename(
                local_file_path, 
                content_type=content_type
            )
            
            # 设置文件为公开可读
            # blob.make_public()
            
            # 获取文件信息
            blob.reload()
            public_url = blob.public_url
            
            print(f"上传成功: {local_file_path} -> {public_url}")
            
            return {
                'filename': gcs_filename,
                'gcs_url': f"gs://{self.bucket.name}/{gcs_filename}",
                'public_url': public_url,
                'file_size': os.path.getsize(local_file_path),
                'content_type': content_type,
                'bucket_name': self.bucket_name,
                'gcs_path': gcs_filename,
                'md5_hash': blob.md5_hash,
                'etag': blob.etag
            }
            
        except Exception as e:
            print(f"上传失败 {local_file_path}: {e}")
            return None
    
    def _get_content_type(self, file_path: str) -> str:
        """根据文件扩展名获取content type"""
        ext = Path(file_path).suffix.lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            '.tiff': 'image/tiff',
            '.ico': 'image/x-icon',
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def save_to_database(self, image_info: Dict[str, Any]) -> Optional[int]:
        """将图片信息保存到数据库"""
        session = next(self.get_db_session())
        try:
            image_record = ImageRecord(
                filename=image_info['filename'],
                gcs_url=image_info['gcs_url'],
                public_url=image_info['public_url'],
                file_size=image_info['file_size'],
                content_type=image_info['content_type'],
                bucket_name=image_info['bucket_name'],
                gcs_path=image_info['gcs_path']
            )
            
            session.add(image_record)
            session.commit()
            session.refresh(image_record)
            
            print(f"数据库记录创建成功，ID: {image_record.id}")
            return image_record.id
            
        except Exception as e:
            session.rollback()
            print(f"数据库插入失败: {e}")
            return None
        finally:
            session.close()
    
    def upload_single_image(self, local_file_path: str, gcs_filename: Optional[str] = None) -> Optional[ImageRecord]:
        """上传单个图片并返回数据库记录"""
        image_info = self.upload_image_to_gcs(local_file_path, gcs_filename)
        if not image_info:
            return None
        
        image_id = self.save_to_database(image_info)
        if not image_id:
            return None
        
        # 获取完整的记录
        session = next(self.get_db_session())
        try:
            record = session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            return record
        finally:
            session.close()
    
    def upload_images_from_folder(self, folder_path: str) -> List[ImageRecord]:
        """上传文件夹中的所有图片"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.ico'}
        uploaded_records = []
        
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                print(f"处理文件: {file_path}")
                
                record = self.upload_single_image(str(file_path))
                if record:
                    uploaded_records.append(record)
        
        return uploaded_records
    
    def get_image_by_filename(self, filename: str) -> Optional[ImageRecord]:
        """根据文件名查找图片记录"""
        session = next(self.get_db_session())
        try:
            return session.query(ImageRecord).filter(ImageRecord.filename == filename).first()
        finally:
            session.close()
    
    def get_all_images(self, limit: int = 100) -> List[ImageRecord]:
        """获取所有图片记录"""
        session = next(self.get_db_session())
        try:
            return session.query(ImageRecord).order_by(ImageRecord.upload_time.desc()).limit(limit).all()
        finally:
            session.close()
    
    def delete_image_record(self, image_id: int) -> bool:
        """删除图片记录（不会删除GCS中的文件）"""
        session = next(self.get_db_session())
        try:
            record = session.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if record:
                session.delete(record)
                session.commit()
                print(f"已删除记录 ID: {image_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"删除记录失败: {e}")
            return False
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        print("数据库连接已关闭")

def main():
    # 配置信息
    GCS_KEY_PATH = "..\db\compact-harbor-475309-h1-7ac990971da7.json"
    BUCKET_NAME = "irrs-images"
    IMAGES_FOLDER = "C:\csg_Folder\MyProject\RentingRecommendation\99co_img"
    
    # 数据库URL (SQLAlchemy格式)
    DATABASE_URL = get_database_url()
    
    # 创建上传器实例
    uploader = SQLAlchemyImageUploader(GCS_KEY_PATH, BUCKET_NAME, DATABASE_URL)
    
    try:
        # 上传图片
        print("开始上传图片...")
        records = uploader.upload_images_from_folder(IMAGES_FOLDER)
        
        print(f"\n上传完成！共处理 {len(records)} 个文件")
        
        # 显示上传结果
        for record in records:
            print(f"ID: {record.id}, 文件名: {record.filename}")
            print(f"URL: {record.public_url}")
            
    finally:
        uploader.close()

if __name__ == "__main__":
    main()