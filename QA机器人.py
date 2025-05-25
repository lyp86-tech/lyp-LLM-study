import streamlit as st
import pandas as pd
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain_experimental.utilities import PythonREPL  
import os
import httpx

# 设置页面布局
st.set_page_config(page_title="Excel QA Robot", layout="wide")
st.title("📊 Excel智能问答系统")

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

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 配置")
    uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx'])

    # 初始化session_state
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = None
    if 'full_data' not in st.session_state:
        st.session_state.full_data = None

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
            height=200  # 控制预览高度
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
        self.globals["pd"] = __import__('pandas')

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
        temperature=0.1
    )
    # 提取并清洗代码
    raw_code = response.choices[0].message.content
    clean_code = raw_code.replace("```python", "").replace("```", "").strip()
    
    # 第二步：执行生成的代码
    try:
        repl = SafePythonREPL()  # 创建执行环境实例
        # 初始化DataFrame  
        repl.run(f"import pandas as pd")
        repl.run(f"df = pd.DataFrame({json.dumps(df.to_dict(orient='records'))})")  # 修复点1，缺少闭合括号
        # 执行用户代码
        repl.run(clean_code)
        # 增强结果获取逻辑（修复点2）
        result = repl.locals.get('result')
        if result is None:
            return "未生成有效结果，请确认代码包含：result = ..."
        return result
    except Exception as e:
        return f"执行错误: {str(e)}\n生成的代码：{clean_code}"


# 主界面布局
col1, col2 = st.columns([3, 3])

with col1:
    st.header("❓ 提问区")
    question = st.text_area("输入您的问题：", height=400)
  
with col2:
    st.header("💡 回答区")
    answer_placeholder = st.empty()


# 处理流程
if uploaded_file and question:
    with st.spinner("分析中..."):
        try:
            # 直接从session_state获取完整数据
            if st.session_state.full_data is not None:
                client = init_deepseek()
                result = process_question(st.session_state.full_data, question, client)
                
                if isinstance(result, pd.DataFrame):
                    answer_placeholder.dataframe(result)
                else:
                    answer_placeholder.code(str(result), language="python")
        except Exception as e:
            st.error(f"系统错误: {str(e)}")
elif uploaded_file:
    st.info("请输入问题")
else:
    st.info("请先上传Excel文件")