"""app/models/audio_label.py - 音频标签模型"""
from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AudioLable(Base):
    """音频标签模型"""
    __tablename__ = "audio_lable"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    audio_file = Column(String(255))
    status = Column(Integer, default=0, comment="状态标识 0:未标注 1:已标注 2:疑难数据")
    aduio_json = Column(JSON)
    aduio_lable_json = Column(JSON)
    operator_name = Column(String(30), default="")
    is_valid = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
