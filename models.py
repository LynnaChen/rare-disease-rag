"""
兼容层：为了不破坏旧代码，这个文件只简单代理到 config 中的函数。
实际的模型与配置逻辑已经全部合并到 config.py 里。
"""

from config import get_embed_model, get_generation_model

__all__ = ["get_embed_model", "get_generation_model"]