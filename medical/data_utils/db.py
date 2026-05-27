"""app/database.py - 数据库连接配置"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Base 必须在最前面定义，audio_db_schema 依赖它
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from medical.data_utils.config import config
from medical.data_utils.log import logger

# 创建数据库引擎
engine = create_engine(
    config.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """获取数据库会话依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_audio_file_exists(audio_file: str) -> bool:
    """检查 audio_file 字段是否已存在记录"""
    from medical.data_utils.audio_db_schema import AudioLable
    db = SessionLocal()
    try:
        return db.query(AudioLable).filter(AudioLable.audio_file == audio_file).first() is not None
    finally:
        db.close()


def insert_audio_label(audio_file: str, aduio_json: dict = None) -> bool:
    """向 audio_lable 表插入一条记录，返回是否成功"""
    from medical.data_utils.audio_db_schema import AudioLable
    db = SessionLocal()
    try:
        record = AudioLable(audio_file=audio_file, aduio_json=aduio_json or {})
        db.add(record)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"[DB] 插入 audio_lable 失败: {audio_file} -> {e}")
        return False
    finally:
        db.close()


def check_database_connection():
    """检查数据库连接是否正常"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"数据库连接失败：{e}")
        return False
