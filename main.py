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

def gradio_interface(query, vid1_url, vid1_dir):
    # 视频和路径设置
    query=query
    vid1_url =vid1_url
    vid1_dir = vid1_dir
    LANCEDB_HOST_FILE = os.path.join(vid1_dir,"sample-lancedb")

    # 下载视频（伪代码，实际方法需自定义）
    vid1_filepath = download_video_bilibili(vid1_url, vid1_dir)
    video_filename = os.path.basename(vid1_filepath)
    video_absolute_path = os.path.abspath(os.path.join(vid1_dir, video_filename))

    # 音频提取
    audio_path = os.path.join(vid1_dir, "vid1_fil.mp3")

    def extract_audio_from_video(video_path, audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3')

    extract_audio_from_video(video_absolute_path, audio_path)

    # Whisper模型转录
    model = whisperA.load_model("small")
    options = dict(task="transcribe", best_of=1, language='zh')
    results = model.transcribe(audio_path, **options)
    text_content = results.get('text')

    # 保存转录文本
    output_file_path = os.path.join(vid1_dir, "vid1_fil.txt")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(text_content)

    # 抽取视频帧
    path_to_video_photo = os.path.join(vid1_dir, "photo")
    extract_frames(video_absolute_path, path_to_video_photo, interval=1)

    # 构建图片嵌入
    embeder = Visualized_BGE(
        model_name_bge="bge-m3",
        model_weight="./Visualized_m3.pth",
        from_pretrained="./bge"
    )
    embeder.eval()

    image_db = ImageEmbeddingLanceDB(embedding=embeder, db_path=LANCEDB_HOST_FILE)
    image_db.add_images_from_folder(path_to_video_photo)

    # 查询图片
    db = lancedb.connect(LANCEDB_HOST_FILE)
    tbl = db.open_table("image_embeddings2")
    query_vector = embeder.encode(text=query).detach().cpu().numpy().astype(np.float32)
    query_vector = np.squeeze(np.array(query_vector)).astype(np.float32)

    docs = (
        tbl.search(query=query_vector, vector_column_name='embedding')
        .limit(4)
        .to_pandas()
    )

    first_image_name = docs['image_name'][0]
    first_image_path = os.path.join(path_to_video_photo, first_image_name)

    # 繁体转换为简体
    converter = opencc.OpenCC('t2s')
    with open(output_file_path, 'r', encoding='utf-8') as f:
        traditional_text = f.read()

    simplified_text = converter.convert(traditional_text)
    text2_path = os.path.join(vid1_dir, "vid1_text2.txt")
    with open(text2_path, 'w', encoding='utf-8') as f:
        f.write(simplified_text)

    # 文本嵌入构建
    file_path = os.path.join(vid1_dir, "vid1_text2.txt")
    loader = TextLoader(file_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.create_documents([doc.page_content for doc in document])
    text_db = TextEmbeddingLanceDB(embeder, db_path=LANCEDB_HOST_FILE)
    text_db.vectorize_and_store(docs)

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=lambda t: list(jieba.cut(t)))
    bm25_res = bm25.invoke(query)

    tbl2 = db.open_table("text_embeddings")
    vector_docs = (
        tbl2.search(query=query_vector, vector_column_name='embedding')
        .limit(4)
        .to_pandas()
    )

    # RRF计算
    def compute_rrf(rank, k=60):
        return 1 / (k + rank)

    rrf_scores = defaultdict(float)
    for rank, row in enumerate(vector_docs['page_content']):
        rrf_scores[row] += compute_rrf(rank + 1)

    for rank, result in enumerate(bm25_res):
        rrf_scores[result.page_content] += compute_rrf(rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_two_content = [content for content, score in sorted_results[:2]]
    text = " ".join(top_two_content)

    # 模型推理
    prompt_template = "用户的问题是'{query}'，以下是和问题有关的文本'{text}'，请你根据提供的文本和图像，用中文作答"
    prompt = prompt_template.format(text=text, query=query)
    b64_img = encode_image(first_image_path)

    qna_transcript_conv = prediction_guard_llava_conv.copy()
    qna_transcript_conv.append_message('user', [prompt, b64_img])

    answer = lvlm_inference_with_conversation(
        qna_transcript_conv,
        model_path='/root/autodl-fs/LVLM'
    )
    assistant_answer = answer.split("ASSISTANT:")[1].strip()
    
    yield first_image_path, text, assistant_answer
