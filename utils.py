# Add your utilities or helper functions to this file.
import torch
import subprocess
import you_get
import io
import os
from dotenv import load_dotenv, find_dotenv
from io import StringIO, BytesIO
import textwrap
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from enum import auto, Enum
import base64
import glob
from tqdm import tqdm
from pytubefix import YouTube, Stream
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
from predictionguard import PredictionGuard
import cv2
import json
import PIL
from PIL import Image
import dataclasses
import random
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer,AutoConfig
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
from transformers import LlavaNextProcessor, LlavaForConditionalGeneration,AutoProcessor
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import (
    MessageLikeRepresentation,
)

MultimodalModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]]


def extract_frames(video_path, output_folder, interval=1):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 使用 OpenCV 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频的帧率（每秒帧数）
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # 每隔 `interval` 秒截取一帧

    frame_count = 0  # 帧计数器
    saved_count = 0  # 已保存的帧计数

    while True:
        ret, frame = cap.read()
        if not ret:  # 如果没有读取到帧，说明视频结束
            break

        # 每隔指定帧数截取一帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {saved_count} at {frame_filename}")
            saved_count += 1

        frame_count += 1

    # 释放视频资源
    cap.release()
    print("Frame extraction completed.")

import os
import subprocess

def download_video_bilibili(video_url, video_path):
    print(f'Getting video information for {video_url}')
    
    # 检查 URL 是否有效
    if not video_url.startswith('http'):
        raise ValueError("Invalid video URL")

    # 确保保存路径存在
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    # 使用 you-get 下载视频并指定输出目录
    print('Downloading video from Bilibili...')
    subprocess.run(['you-get', '-o', video_path, video_url])

    # 获取下载目录下的所有文件，并筛选出视频文件
    downloaded_files = os.listdir(video_path)
    video_files = [f for f in downloaded_files if f.endswith(('.mp4', '.flv', '.avi'))]

    # 如果没有视频文件，提示用户检查下载结果
    if not video_files:
        raise FileNotFoundError("No video file found in the specified directory. Please check the download process.")

    # 找到最新下载的视频文件
    video_files.sort(key=lambda f: os.path.getmtime(os.path.join(video_path, f)), reverse=True)
    latest_video_file = video_files[0]

    # 返回视频文件的完整路径
    filepath = os.path.join(video_path, latest_video_file)
    return filepath

def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)

def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        return default
        
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_prediction_guard_api_key():
    load_env()
    PREDICTION_GUARD_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY", None)
    if PREDICTION_GUARD_API_KEY is None:
        PREDICTION_GUARD_API_KEY = input("Please enter your Prediction Guard API Key: ")
    return PREDICTION_GUARD_API_KEY
    
PREDICTION_GUARD_URL_ENDPOINT = os.getenv("DLAI_PREDICTION_GUARD_URL_ENDPOINT", "https://dl-itdc.predictionguard.com") ###"https://proxy-dl-itdc.predictionguard.com"

# prompt templates
templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]

# function helps to prepare list image-text pairs from the first [test_size] data of a Huggingface dataset
def prepare_dataset_for_umap_visualization(hf_dataset, class_name, templates=templates, test_size=1000):
    # load Huggingface dataset (download if needed)
    dataset = load_dataset(hf_dataset, trust_remote_code=True)
    # split dataset with specific test_size
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    # get the test dataset
    test_dataset = train_test_dataset['test']
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)].format(class_name),
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs
    

def download_video(video_url, path='/tmp/'):
    print(f'Getting video information for {video_url}')
    if not video_url.startswith('http'):
        return os.path.join(path, video_url)

    filepath = glob.glob(os.path.join(path, '*.mp4'))
    if len(filepath) > 0:
        return filepath[0]

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    
    yt = YouTube(video_url, on_progress_callback=progress_callback)
    stream = yt.streams.filter(progressive=True, file_extension='mp4', res='720p').desc().first()
    if stream is None:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, stream.default_filename)
    if not os.path.exists(filepath):   
        print('Downloading video from YouTube...')
        pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
        stream.download(path)
        pbar.close()
    return filepath

def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]

    return video_url
    
# if this has transcript then download
def get_transcript_vtt(video_url, path='/tmp'):
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        return filepath

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

    return filepath
    

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# a help function that helps to convert a specific time written as a string in format `webvtt` into a time in miliseconds
def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    # get the corresponding time as total seconds 
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds
    
def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)
    
# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

# encoding image at given path or PIL Image using base64
def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        # this is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# checking whether the given string is base64 or not
def isBase64(sb):
    try:
        if isinstance(sb, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
                sb_bytes = sb
        else:
                raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False

def encode_image_from_path_or_url(image_path_or_url):
    try:
        # try to open the url to check valid url
        f = urlopen(image_path_or_url)
        # if this is an url
        return base64.b64encode(requests.get(image_path_or_url).content).decode('utf-8')
    except:
        # this is a path to image
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# helper function to compute the joint embedding of a prompt and a base64-encoded image through PredictionGuard
def bt_embedding_from_prediction_guard(prompt, base64_image):
    # 加载处理器和模型
    model_path = '/root/autodl-tmp/BridgeTower'
    # 加载本地模型和分词器
    processor = BridgeTowerProcessor.from_pretrained(model_path)
    model = BridgeTowerForContrastiveLearning.from_pretrained(model_path)

    inputs = {"text": prompt}
    
    if base64_image:
        # 解码base64图像
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # 转换为RGB格式以供处理
        inputs["images"] = image

    # 预处理输入
    processed_inputs = processor(text=[inputs['text']], images=[inputs.get('images', None)], return_tensors="pt")

    # 生成embedding
    with torch.no_grad():
        outputs = model(**processed_inputs)
        print(outputs.cross_embeds)
        print(outputs.cross_embeds.shape)
    
    embeddings = outputs.cross_embeds.flatten()
    return embeddings.tolist()  # 或者返回你想要的特定形式

    
def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_retrieved_results(results):
    print(f'There is/are {len(results)} retrieved result(s)')
    print()
    for i, res in enumerate(results):
        print(f'The caption of the {str(i+1)}-th retrieved result is:\n"{results[i].page_content}"')
        print()
        display(Image.open(results[i].metadata['metadata']['extracted_frame_path']))
        print("------------------------------------------------------------")

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history"""
    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"    

    def _get_prompt_role(self, role):
        if self.map_roles is not None and role in self.map_roles.keys():
            return self.map_roles[role]
        else:
            return role
            
    def _build_content_for_first_message_in_conversation(self, first_message: List[str]):
        content = []
        if len(first_message) != 2:
            raise TypeError("First message in Conversation needs to include a prompt and a base64-enconded image!")
        
        prompt, b64_image = first_message[0], first_message[1]
        
        # handling prompt
        if prompt is None:
            raise TypeError("API does not support None prompt yet")
        content.append({
            "type": "text",
            "text": prompt
        })
        if b64_image is None:
            raise TypeError("API does not support text only conversation yet")
            
        # handling image
        if not isBase64(b64_image):
            raise TypeError("Image in Conversation's first message must be stored under base64 encoding!")
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": b64_image,
            }
        })
        return content

    def _build_content_for_follow_up_messages_in_conversation(self, follow_up_message: List[str]):

        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError("Follow-up message in Conversation must not include an image!")
        
        # handling text prompt
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError("Follow-up message in Conversation must include exactly one text message")

        text = follow_up_message[0]
        return text
        
    def get_message(self):
        messages = self.messages
        api_messages = []
        for i, msg in enumerate(messages):
            role, message_content = msg
            if i == 0:                
                # get content for very first message in conversation
                content = self._build_content_for_first_message_in_conversation(message_content)
            else:
                # get content for follow-up message in conversation
                content = self._build_content_for_follow_up_messages_in_conversation(message_content)
                
            api_messages.append({
                "role": role,
                "content": content,
            })
        return api_messages

    # this method helps represent a multi-turn chat into as a single turn chat format
    def serialize_messages(self):
        messages = self.messages
        ret = ""
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is not None and self.system != "":
                ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                role = self._get_prompt_role(role)
                if message:
                    if isinstance(message, List):
                        # get prompt only
                        message = message[0]
                    if i == 0:
                        # do not include role at the beginning
                        ret += message
                    else:
                        ret += role + ": " + message
                    if i < len(messages) - 1:
                        # avoid including sep at the end of serialized message
                        ret += self.sep
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def append_message(self, role, message):
        if len(self.messages) == 0:
            # data verification for the very first message
            assert role == self.roles[0], f"the very first message in conversation must be from role {self.roles[0]}"
            assert len(message) == 2, f"the very first message in conversation must include both prompt and an image"
            prompt, image = message[0], message[1]
            assert prompt is not None, f"prompt must be not None"
            assert isBase64(image), f"image must be under base64 encoding"
        else:
            # data verification for follow-up message
            assert role in self.roles, f"the follow-up message must be from one of the roles {self.roles}"
            assert len(message) == 1, f"the follow-up message must consist of one text message only, no image"
            
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x,y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }

prediction_guard_llava_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="Prediction Guard LLaVA enpoint Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={
        "user": "USER", 
        "assistant": "ASSISTANT"
    }
)

# get PredictionGuard Client
def _getPredictionGuardClient():
    PREDICTION_GUARD_API_KEY = get_prediction_guard_api_key()
    client = PredictionGuard(
        api_key=PREDICTION_GUARD_API_KEY,
        url=PREDICTION_GUARD_URL_ENDPOINT,
    )
    return client

# helper function to call chat completion endpoint of PredictionGuard given a prompt and an image
def lvlm_inference(model_path: str,prompt, image, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1, top_k: int = 10):
    # prepare conversation
    model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
    ).to(0)  # 使用GPU（假设为cuda:0）

    processor = LlavaNextProcessor.from_pretrained(model_path)
    conversation2 = [
        {
            "role":"user",
            "content":[
                {"type":"text","text":prompt},
                {"type":"image"},
            ],
        },
    ]
    # print(conversation2)
 
    prompt = processor.apply_chat_template(conversation2, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    # outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    # print(output)
    # 解码生成的文本输出并返回
    return processor.decode(output[0][2:], skip_special_tokens=True)
    
    
def lvlm_inference_with_conversation(conversation, model_path: str, max_tokens: int = 200,temperature: float = 0.95, top_p: float = 0.1, top_k: int = 10):
    # 加载本地部署的LLava模型和处理器
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(0)  # 使用GPU（假设为cuda:0）

    processor = LlavaNextProcessor.from_pretrained(model_path)
    # 初始化图像和文本提示
    image = None
    prompt = ""

        # 处理每条消息的内容
    for msg in conversation.messages:
        role = msg[0]  # 获取角色，例如 'user'
        content_list = msg[1]# 获取内容列表

        # 处理每条消息的内容
        for content in content_list:
            if isinstance(content, str) and content.startswith('/9j'):  # 识别图像（假设图像是以 base64 编码的 JPEG 文件）
                # 解码 base64 图像并转换为 PIL Image 对象
                image_data = base64.b64decode(content)
                image = Image.open(BytesIO(image_data))
                image.show()
            elif isinstance(content, str):  # 处理文本部分
                prompt += content + "\n"  # 拼接文本部分作为提示

    # 如果没有图像，抛出异常
    if image is None:
        raise ValueError("No image found in conversation")

    conversation2 = [
        {
            "role":role,
            "content":[
                {"type":"text","text":prompt},
                {"type":"image"},
            ],
        },
    ]
    # print(conversation2)
 
    prompt = processor.apply_chat_template(conversation2, add_generation_prompt=True)
    # 准备输入数据（图像和对话提示）
    # print(prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(0, torch.float16)

    # 使用模型生成输出
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    # outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    # print(output)
    # 解码生成的文本输出并返回
    return processor.decode(output[0][2:], skip_special_tokens=True)
