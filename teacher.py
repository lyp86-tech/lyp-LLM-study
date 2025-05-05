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
st.title("📖 小学四年级")
st.subheader("刘妙可趣味学习")
st.caption("输入关键词，生成一道题")


# 用户输入界面
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("请输入题目关键词（用逗号分隔）:", placeholder="例如：数学，长方形，面积")

with col2:
    st.write("")
    st.write("")
    generate_btn = st.button("生成题目")

#二、模型设置

# 定义提示模板
STORY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个专业的小学四年级下册老师，在江苏教学语文、数学、英语。根据用户提供的关键词，生成一个包含以下要素的小学四年题目：
    1.数学以应用题为主
    2.语文以按课文内容填空为主
    3.英语以翻译句子为主
    4.有一定难度
        
    要求：
    - 需要考虑到小学四年级的知识储备
    - 需要考虑到小学四年级的理解能力
    -以应用题、计算题、阅读理解为主
    - 关键词：{keywords}
    
    请直接输出生成的题目，不要包含任何额外说明。"""
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
    
    with st.spinner("正在生成题目，请稍候..."):
        try:
            story = get_response(user_input)
            st.subheader("生成的题目：")
            st.markdown(f'<div style="text-align: justify; line-height: 1.6;">{story}</div>', 
                       unsafe_allow_html=True)
        except Exception as e:
            st.error(f"生成失败：{str(e)}")


