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
