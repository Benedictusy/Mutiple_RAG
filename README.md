# Mutiple_RAG
在这个项目中，我实现了一个多模态RAG（检索增强生成）视频问答系统，该系统能够理解视频中的视觉和文本信息，从而提供准确的答案。

该项目需要读者自行从huggingface上下载三个部分：
1、bge_m3(https://huggingface.co/BAAI/bge-m3/)

2、Visualized_m3.pth（https://huggingface.co/BAAI/bge-visualized/blob/main/Visualized_m3.pth）

3、LLaVA(https://huggingface.co/liuhaotian/llava-v1.5-7b) 

上述三个模型即该系统中用到的模型，（中英文均支持），下载完毕后，将main.py文件中

embeder = Visualized_BGE(
        model_name_bge="bge-m3",
        model_weight="./Visualized_m3.pth",
        from_pretrained="./bge"
    )
中的"model_weight"改为Visualized_m3.pth的路径，"from_pretrained"改为bge_m3文件夹的路径，再将
answer = lvlm_inference_with_conversation(
        qna_transcript_conv,
        model_path='/root/autodl-fs/LVLM'
    ),"model_path"改为LLaVA所在的文件夹。

运行demo.py，打开链接即可，可以打开任意一个bilibili视频，复制他的链接，然后提出你的问题，该系统就会给出相应的答案。（目前我还没有做多轮对话）

# 项目结构
1、mm_rag/mm_rag/vectorstores中定义了两个类
图像嵌入管理（ImageEmbeddingLanceDB）和文本嵌入管理（TextEmbeddingLanceDB）
ImageEmbeddingLanceDB
功能：
对图像文件进行编码，生成嵌入向量，并将这些向量及相关信息存储到 LanceDB 数据库中。

embedding: 图像编码模型
db_path: 数据库路径
table_name: 数据库表名
vector_key: 嵌入向量的字段名
id_key: 图像唯一 ID 的字段名
image_name_key: 图像文件名的字段名
mode: 数据库操作模式（如 append）
核心功能：
表结构定义：
表中包含以下字段：
id: 唯一标识符
image_name: 图像名称
embedding: 嵌入向量（固定维度）
图像编码： 调用嵌入模型对输入图像编码，生成 1024 维向量。
批量数据插入：
遍历文件夹中的图像，将编码后的嵌入批量插入 LanceDB 表。  
TextEmbeddingLanceDB  
功能：
对文本内容进行编码，生成嵌入向量，并将这些向量及相关信息存储到 LanceDB 数据库中。

主要构造参数：

embedding: 文本编码模型
db_path: 数据库路径
table_name: 数据库表名
vector_key: 嵌入向量的字段名
id_key: 文本唯一 ID 的字段名
text_name_key: 文本名称的字段名
content_key: 文本内容的字段名
mode: 数据库操作模式（如 append）
核心功能：
表结构定义：
表中包含以下字段：
id: 唯一标识符
text_name: 文本名称
embedding: 嵌入向量（固定维度）
page_content: 文本内容
文本编码：
调用嵌入模型对文本编码，生成 1024 维向量。
批量数据插入：
将文本内容、文本名称及编码向量打包后插入 LanceDB 表。

2、main.py文件中含有整个视频的处理流程，包括抽取视频帧、执行相似性搜索、多路召回、prompt设计等  
3、该文件通过gradio，构建前端页面，方便用户通过前端页面直接进行交互。
