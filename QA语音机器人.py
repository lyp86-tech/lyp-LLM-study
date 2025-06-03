import streamlit as st
import pandas as pd
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain_experimental.utilities import PythonREPL  
import os
import httpx
import base64
from pydub import AudioSegment
import tempfile
import threading
import queue
import time
import wave
import io
import pyaudio

# 设置页面布局
st.set_page_config(page_title="Excel QA Robot", layout="wide")
st.title("📊 基于数据库的对话机器人")

# 检查PyAudio是否可用
try:
    import pyaudio
    PY_AUDIO_AVAILABLE = True
except ImportError:
    PY_AUDIO_AVAILABLE = False
    st.sidebar.warning("PyAudio未安装，录音功能不可用。请使用命令`pip install pyaudio`安装。")

def init_deepseek():
    # 创建自定义HTTP客户端
    custom_client = httpx.Client(
        proxies=None,  # 显式禁用代理
        trust_env=False,  # 新增：禁止读取环境变量代理配置
        timeout=30.0,
        transport=httpx.HTTPTransport(retries=3)
    )
    
    return OpenAI(
        api_key="sk-ee72ed73b1bf4a2bbe867660fcfe52b2",
        base_url="https://api.deepseek.com/v1",
        http_client=custom_client  # 使用自定义客户端
    )
#语音识别函数
def transcribe_audio(audio_bytes):
    try:
        # 创建OpenAI客户端（使用官方API）
        openai_client = OpenAI(
            api_key="sk-hHLAlXBdnlefUZNrbr9V7okyNBVjNLc7oMzHUGfAqsW4T2Wv",  
            base_url="https://api.ipacs.top/v1"  
        )
        # 创建语音识别请求 - 使用BytesIO对象
        with io.BytesIO(audio_bytes) as audio_file:
            audio_file.name = "audio.wav"
            # 调用API并获取结构化响应
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            # 调试输出 - 查看响应结构
            #st.write("API响应类型:", type(response))
            #st.write("API响应内容:", response)
            
            # 提取转录文本 - 检查不同可能的响应结构
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'result') and hasattr(response.result, 'text'):
                return response.result.text
            elif hasattr(response, 'transcriptions') and len(response.transcriptions) > 0:
                return response.transcriptions[0].text
            else:
                st.error(f"无法解析API响应: {response}")
                return None          
    except Exception as e:
        st.error(f"语音识别失败: {str(e)}")
        return None

# 数据加载模块
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        # 新增数据预览存储
        st.session_state.preview_data = df.head(5)
        return df,st.session_state.preview_data

    except Exception as e:
        st.error(f"文件加载失败: {str(e)}")
        return None
# 录音函数
def record_audio(stop_event, audio_queue):
    # 录音参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()
    
    # 打开音频流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    
    # 开始录音
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
    
    # 停止录音
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # 将音频数据转换为WAV格式
    wav_io = io.BytesIO()
    wf = wave.open(wav_io, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # 将音频数据放入队列
    audio_queue.put(wav_io.getvalue())
        
# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 配置")
    uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx'])

    # 初始化session_state
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    if 'full_data' not in st.session_state:
        st.session_state.full_data = None
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'record_thread' not in st.session_state:
        st.session_state.record_thread = None
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if 'audio_queue' not in st.session_state:
        st.session_state.audio_queue = queue.Queue()

    # 文件处理逻辑
    if uploaded_file is not None:
        try:
            # 读取Excel文件
            df = pd.read_excel(uploaded_file)
            # 存储前5行数据到session state
            st.session_state.preview_data = df.head(5)
            st.session_state.full_data = df  # 存储完整数据
            # 显示成功提示
            st.toast("✅ 文件已成功加载", icon="✅")
            
        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")
            #清楚所有数据状态
            st.session_state.preview_data = None
            st.session_state.full_data = None
    else:
        # 当文件被删除时清除所有数据
        st.session_state.preview_data = None
        st.session_state.full_data = None
        
    # 显示预览或错误信息
    if st.session_state.preview_data is not None:
        st.subheader("📋 数据预览（前5行）")
        st.dataframe(
            st.session_state.preview_data,
            use_container_width=True,
            height=220  # 控制预览高度
        )
        st.caption(f"总行数: {len(st.session_state.full_data)} 行") # 显示完整数据的总行数
      
    else:
        st.subheader("📋 数据预览")
        st.error("🚫 数据预览失败")  # 错误提示
        st.caption("请先上传有效数据文件")  # 补充说明

# 自定义Python执行环境
class SafePythonREPL(PythonREPL):
    def __init__(self):
        super().__init__()
        self.globals = {}
        self.locals = {}
        self.globals["pd"] = __import__('pandas')
        self.globals["np"] = __import__('numpy')
        self.globals["plt"] = __import__('matplotlib.pyplot')
        self.globals["st"] = __import__('streamlit')
    
    #6月1日下午2点新增
    def run(self, code):
        try:
            # 安全执行代码
            exec(code, self.globals, self.locals)
        except Exception as e:
            return f"执行错误: {str(e)}"

#问题处理模块
def process_question(df, question, client):
    # 第一步：生成查询代码
    prompt = f"""基于数据表结构：{df.columns.tolist()}
    生成Python代码解决：{question}
    要求：
    1. 只输出代码 
    2. 不要Markdown标记 
    3. 结果保存到result变量
    4. 使用pd.DataFrame进行操作
    5. 不要添加额外解释"""

    # 调用Deepseek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    
    # 提取并清洗代码
    raw_code = response.choices[0].message.content
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()
    
    # 第二步：执行生成的代码
    try:
        repl = SafePythonREPL()  # 创建执行环境实例
        # 初始化DataFrame  
        repl.run(f"import pandas as pd")
        repl.run(f"import numpy as np")
        repl.run(f"import matplotlib.pyplot as plt")
        repl.run(f"import seaborn as sns")
        repl.run(f"import streamlit as st")
        repl.run(f"df = pd.DataFrame({json.dumps(df.to_dict(orient='records'))})") 
        # 执行用户代码
        repl.run(clean_code)
        # 增强结果获取逻辑
        result = repl.locals.get('result')
        if result is None:
            return "代码执行完成但未生成结果", raw_code, clean_code
        return result, raw_code, clean_code
    except Exception as e:
        return f"执行错误: {str(e)}", "", ""


# 主界面布局
col1, col2 = st.columns([2, 2])

with col1:
    st.header("❓ 提问区")
    # 修改录音控制部分
    st.subheader("🎤 语音输入")
    if st.session_state.recording:
        st.warning("录音中... 请说话")
    # 显示录制的音频
    if st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format="audio/wav")
     # 检查PyAudio是否可用
    if not PY_AUDIO_AVAILABLE:
        st.warning("录音功能不可用，请安装PyAudio")
    else:
        # 录音控制按钮
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            if st.button("开始录音", disabled=st.session_state.recording):
                # 重置状态
                st.session_state.audio_bytes = None
                st.session_state.transcribed_text = ""
                st.session_state.recording = True
                
                # 创建停止事件和队列
                st.session_state.stop_event.clear()
                st.session_state.audio_queue = queue.Queue()
                
                # 启动录音线程
                st.session_state.record_thread = threading.Thread(
                    target=record_audio,
                    args=(st.session_state.stop_event, st.session_state.audio_queue)
                )
                st.session_state.record_thread.start()
                st.rerun()

        with col_rec2:
            if st.button("停止录音", disabled=not st.session_state.recording):
                # 设置停止事件
                st.session_state.stop_event.set()
                
                # 等待录音线程结束
                st.session_state.record_thread.join()
                
                # 获取音频数据
                try:
                    audio_data = st.session_state.audio_queue.get(timeout=2)
                    st.session_state.audio_bytes = audio_data
                    st.session_state.recording = False
                    st.success("录音完成!")
                except queue.Empty:
                    st.error("未能获取录音数据")
    # 语音识别按钮
    if st.button("识别语音", disabled=not bool(st.session_state.audio_bytes)):
        if st.session_state.audio_bytes:
            with st.spinner("正在识别语音..."):
                transcribed = transcribe_audio(st.session_state.audio_bytes)
                if transcribed:
                    st.session_state.transcribed_text = transcribed
                    st.success("语音识别成功!")
                    st.write(f"识别结果: {transcribed}")  # 显示识别结果
                else:
                    st.warning("未能识别语音内容")
        else:
            st.warning("请先录制音频")
    #问题输入框
    question = st.text_area(
    "输入您的问题（注意表头名称一定要准确）:", 
    height=200,
    placeholder="示例问题:\n- 计算合计流量\n- 找出流量最大/最小的前五个站\n- 按时间进行统计\n- 按流量区间进行统计\n- 找变化最大的五个站\n- ……",
    key="question_input",
    value=st.session_state.transcribed_text
    )
    
    # 处理流程
    if st.button("提交问题"):
        if uploaded_file and question:
            with st.spinner("分析中..."):
                try:
                    # 直接从session_state获取完整数据
                    if st.session_state.full_data is not None:
                        client = init_deepseek()
                        result,raw_code,clean_code= process_question(st.session_state.full_data, question, client)   
                        final_code_placeholder = st.code(clean_code, language="python")
                        # 存储结果到session state
                        st.session_state.result = result
                        st.session_state.raw_code = raw_code
                        st.session_state.clean_code = clean_code
                except Exception as e:
                    st.error(f"系统错误: {str(e)}")
        elif not uploaded_file:
            st.info("请输入问题")
        else:
            st.info("请先上传Excel文件")

with col2:
    st.header("💡 回答区")
    answer_placeholder = st.empty()
    if 'result' in st.session_state:
        if isinstance(st.session_state.result, pd.DataFrame):
            st.dataframe(st.session_state.result)
        else:
            st.code(str(st.session_state.result), language="python")