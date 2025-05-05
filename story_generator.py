#首先在环境中安装依赖：pip install langchain langchain-openai streamlit 

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

DEFAULT_API_KEY = "sk-ee72ed73b1bf4a2bbe867660fcfe52b2"  # 替换为有效密钥

#一、界面设置

# 设置页面标题和图标
st.set_page_config(
    page_title="刘艳平LLM学习",
    page_icon="📖",
    layout="centered"
)

# 侧边栏设置API KEY（修改为DeepSeek密钥）
with st.sidebar:
    st.title("设置")
    deepseek_api_key = DEFAULT_API_KEY
    st.success("已自动加载测试秘钥")

# 页面主标题
st.title("📖 中文故事生成器")
st.subheader("刘艳平LLM学习测试")
st.caption("输入关键词，生成一个完整的中文小故事")


# 用户输入界面
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("请输入故事关键词（用逗号分隔）:", placeholder="例如：夏天, 冰淇淋, 友谊")

with col2:
    st.write("")
    st.write("")
    generate_btn = st.button("生成故事")

#二、模型设置

# 定义提示模板
STORY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个专业的中文故事作家。根据用户提供的关键词，生成一个包含以下要素的中文故事：
    1. 有趣的开头吸引读者
    2. 有起承转合的情节发展
    3. 出人意料的结局
    
    要求：
    - 故事长度约200字
    - 使用生动形象的语言描写
    - 包含人物对话
    - 关键词：{keywords}
    
    请直接输出生成的故事内容，不要包含任何额外说明。"""
)

# 初始化模型
def get_response(keywords):
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
        timeout=30
    )
    chain = STORY_PROMPT | model
    return chain.invoke({"keywords": keywords}).content

#三、处理生成逻辑

if generate_btn:
    
    if not deepseek_api_key:
        st.error("请先输入deepseek API密钥！")
        st.stop()
    
    if not user_input.strip():
        st.error("请输入至少一个关键词")
        st.stop()
    
    with st.spinner("正在生成故事，请稍候..."):
        try:
            story = get_response(user_input)
            st.subheader("生成的故事：")
            st.markdown(f'<div style="text-align: justify; line-height: 1.6;">{story}</div>', 
                       unsafe_allow_html=True)
        except Exception as e:
            st.error(f"生成失败：{str(e)}")


