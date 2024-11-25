import gradio as gr
import shutil
from PIL import Image
import random
from pathlib import Path
import os
from os import path as osp
import json
import cv2
import webvtt
from moviepy.editor import VideoFileClip
from PIL import Image
import base64
import whisperA
import pandas as pd
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from visual_bge.modeling import Visualized_BGE
import opencc
import lancedb
from typing import Any, List
import pyarrow as pa
import numpy as np
from langchain_community.retrievers import BM25Retriever
import jieba
from utils import download_video_bilibili, get_transcript_vtt
from moviepy.editor import *
from utils import extract_frames
from visual_bge.modeling import Visualized_BGE
import uuid
import lancedb
import pyarrow as pa
from typing import Optional, Any, List
import torch
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.embeddings import Embeddings
import lancedb
import numpy as np
import pyarrow as pa
from typing import Any, List
from moviepy.editor import *
from utils import (
    prediction_guard_llava_conv, 
    lvlm_inference_with_conversation
)
from utils import encode_image
import opencc
from collections import defaultdict

class ImageEmbeddingLanceDB:
    def __init__(
        self,
        embedding: Any,  # 传入编码模型
        db_path: str = "/tmp/lancedb",  # LanceDB数据库路径
        table_name: str = "image_embeddings2",  # 数据库表名
        vector_key: str = "embedding",  # 向量字段的键
        id_key: str = "id",  # ID字段的键
        image_name_key: str = "image_name",  # 图像名称字段的键
        mode: str = "append",  # 数据库操作模式，默认为“append”
    ):
        self.embedding = embedding  # 图像编码模型
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.vector_key = vector_key
        self.id_key = id_key
        self.image_name_key = image_name_key
        self.mode = mode

        # 确定嵌入向量的维度N
        DIM_VALUE = 1024  # 根据实际的embedding维度调整

        # 定义表结构
        self.schema = pa.schema([
            (self.id_key, pa.string()),                 # 图片的唯一ID
            (self.image_name_key, pa.string()),         # 图片名称
            (self.vector_key, pa.list_(pa.float32(), list_size=DIM_VALUE)),  # 向量编码
        ])

        # 如果表不存在，则创建表
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, schema=self.schema)

    def encode_image(self, image_path: str) -> List[float]:
        """
        使用模型对图像进行编码，返回嵌入向量的列表
        """
        # 对图像进行编码（假设输出为torch.Tensor）
        embedding_tensor = self.embedding.encode(image=image_path)
        
        # 将编码结果转换为列表，并确保是float32格式
        embedding = embedding_tensor.cpu().detach().numpy().flatten().astype('float32').tolist()
        return embedding

    def add_images_from_folder(self, folder_path: str):
        """
        遍历文件夹中的图片，将每张图片的嵌入存入LanceDB
        """
        table = self.db.open_table(self.table_name)
        docs = []
        
        # 遍历文件夹中的图片文件
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # 生成嵌入向量
            embedding = self.encode_image(image_path)
            # 生成唯一ID
            doc_id = str(uuid.uuid4())
            # 添加到文档列表
            docs.append({
                self.id_key: doc_id,
                self.image_name_key: image_name,
                self.vector_key: embedding,
            })
            print(f"Encoded and prepared for storage: {image_name}")

        # 批量插入文档到表
        table.add(docs, mode=self.mode)
        print(f"Successfully added {len(docs)} images to the LanceDB table '{self.table_name}'.")

class TextEmbeddingLanceDB:
    def __init__(
        self,
        embedding: Any,  # 传入编码模型
        db_path: str = "/root/autodl-fs/shared_data/sample-lancedb",  # LanceDB数据库路径
        table_name: str = "text_embeddings",  # 数据库表名
        vector_key: str = "embedding",  # 向量字段的键
        id_key: str = "id",  # ID字段的键
        text_name_key: str = "text_name",  # 新增的 text_name 字段的键
        mode: str = "append",  # 数据库操作模式，默认为“append”
        content_key:str="page_content",
    ):
        self.embedding = embedding  # 图像编码模型
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.vector_key = vector_key
        self.id_key = id_key
        self.text_name_key = text_name_key  # 保存 text_name 字段的键
        self.mode = mode
        self.content_key=content_key

        # 确定嵌入向量的维度N
        DIM_VALUE = 1024  # 根据实际的embedding维度调整

        # 定义表结构
        self.schema = pa.schema([
            (self.id_key, pa.string()),                 # 图片的唯一ID      
            (self.vector_key, pa.list_(pa.float32(), list_size=DIM_VALUE)),  # 向量编码
            (self.text_name_key, pa.string()),# 新增的 text_name 字段
            (self.content_key,pa.string()),
        ])

        # 如果表不存在，则创建表
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, schema=self.schema)
            
    def vectorize_and_store(self, documents: List[dict]):
        """
        向量化并将文档切片存储到 LanceDB
        :param documents: 切好的文档片段列表，每个片段是一个字典，包含文本内容和唯一ID
        """
        # 创建一个列表存储要插入的数据
        data_to_insert = []

        # 遍历文档切片并向量化
        for doc in documents:
            text = getattr(doc, 'page_content', '')  # 文档的文本内容
            doc_id = getattr(doc, self.id_key, None)  # 使用getattr()访问doc的id字段
            text_name = getattr(doc, self.text_name_key, 'default_name')  # 获取 text_name 字段，如果没有则设置默认值

            # 使用embedding模型对文档进行向量化
            embedding_vector = self.embedding.encode(text=text).detach().cpu().numpy()  # 假设 embedding.encode() 返回向量
            embedding_vector = np.squeeze(np.array(embedding_vector)).astype(np.float32)
            
            # 将结果添加到要插入的列表中
            data_to_insert.append({
                self.id_key: doc_id,
                self.vector_key: embedding_vector,
                self.text_name_key: text_name,  # 加入 text_name 字段
                self.content_key:text,
            })

        # 使用LanceDB插入数据
        table = self.db[self.table_name]
        table.add(data_to_insert, mode=self.mode)
        print(f"{len(data_to_insert)} 文档切片已成功插入到数据库。")
        
