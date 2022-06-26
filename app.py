import streamlit as st
from predict import Bert_summary_model
import os
import time
import json
import argparse

st.set_page_config(page_title="Demo", initial_sidebar_state="auto", layout="wide")

#返回绝对路劲
root = os.path.abspath(os.path.dirname(__file__))

def main():
    st.title("中文文本摘要生成 with Streamlit")
    st.subheader("随时随地进行自然语言处理")

    #参数配置侧栏
    st.sidebar.subheader("配置参数")
    st.sidebar.subheader("\n")
    option = st.sidebar.selectbox('选择摘要层：', ('Transformer', 'RNN', 'Classifier'))
    if option == "Transformer":
        load_from_model = os.path.join(root, 'models/bert_transformer/model_step_30000.pt')
        #load_from_model = 1
    elif option == "RNN":
        load_from_model = os.path.join(root, 'models/bert_rnn/model_step_30000.pt')
        #load_from_model = 2
    else:
        load_from_model = os.path.join(root, 'models/bert_classifier/model_step_30000.pt')
        #load_from_model = 3

    st.sidebar.subheader("\n")

    sum_num = st.sidebar.slider("自定义生成摘要的句子数目：", min_value=1, max_value=3, value=3, step=1)

    #st.sidebar.info("Cudos to the Streamlit Team")

    #Text Summarization
    st.subheader("Sentiment of Your Text")
    message = st.text_area("输入正文")

    if st.button("一键生成摘要"):
        start_message = st.empty()
        start_time = time.time()
        start_message.info("正在抽取，请等待...")
        
        message = message.replace('\n','')

        sum_model = Bert_summary_model(load_from_model)
        if len(message) > sum_model.max_process_len:
            summary = sum_model.long_predict(message,sum_num)
        else:
            summary = sum_model.predict(message,sum_num)

        end_time = time.time()
        #start_message.success("抽取完成，耗时{}s".format(end_time-start_time))
        start_message.success("抽取完成，耗时%.2fs" %(end_time-start_time))
        st.write(summary)
    else:
        st.stop()

if __name__ == '__main__':
    main()