# ====中文文本摘要生成====
# ----------------实验说明--------------
参考：https://github.com/Machine-Tom/bertsum-chinese-LAI<br>
基于论文《Fine-tune BERT for Extractive Summarization》的方法和源代码<br>
进行调整，分别在中文数据集NLPCC2017和LCSTS进行实验<br>
论文作者主页：http://nlp-yang.github.io/ 
<br><br>
# ----------------文件说明--------------
文件夹raw_data：存放数据集的JSON文件 <br>
文件夹json_data：存放进行切割&分句后的数据集的JSON文件 <br>
文件夹bert_data：存放pt文件，将JSON格式转换为pt格式方便训练（句子标注，详见后面运行说明）<br>
src/preprocess.py：对数据集进行处理的运行文件<br>
<br>
文件夹src/models：存放BERTSUM及摘要层(Classifier、Transformer、RNN)代码的文件夹<br>
src/train.py：模型训练及模型评估(ROUGE)的运行文件<br>
文件夹bertsum-chinese/models：存放分别使用三摘要层(Classifier、Transformer、RNN)训练模型的训练数据<br>
<br>
predict.py：预测摘要的运行文件<br>
app.py：使用streamlit实现的Web应用的运行文件<br>
<br>
bert-config.json:模型参数设置文件<br>
<br><br>
# ----------------运行环境--------------
* Ubuntu	18.04
* CUDA	10.1
## --------------requirement------------
* Python	3.7.7
* Pytorch	1.4.0
* pyrouge	0.1.3
* numpy	1.18.2
* emoji	1.6.1
* multiprocess	0.70.12.2
* pytorch_pretrained_bert	0.6.2
* transformers	2.11.0
* streamlit	1.7.0
<br><br>
# ----------------运行说明--------------
## ------------数据集的准备------------
中文数据集1：LCSTS2.0(A Large Scale Chinese Short Text Summarization Dataset)<br>
来源：Intelligent Computing Research Center, Harbin Institute of Technology Shenzhen Graduate School<br>
(哈尔滨工业大学深圳研究生院·智能计算研究中心)<br>
申请途径：http://icrc.hitsz.edu.cn/Article/show/139.html<br>

中文数据集2：NLPCC2017<br>
来源：中国计算机学会(CCF)主办的国际自然语言处理与中文计算会议即NLPCC中的
一项任务——中文新闻文档摘要提取，由官方提供数据集<br>
获取途径：http://tcci.ccf.org.cn/conference/2017/taskdata.php<br>
(Task 3:  Single Document Summarization)<br>
<br><br>
## -------------LCSTS的预处理---------------
STEP 1：首先下载LCSTS2.0原始数据，将LCSTS2.0/DATA目录下所有PART_*.txt文件放入
bertsum-chinese/raw_data<br>
<br>

STEP 2：将原始文件转换成JSON文件存储，在bertsum-chinese/src目录下运行：<br>
<code>python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log</code>
<br>

STEP 3：分割文件&分句<br>
①分割文件：训练集文件太大，分割成小文件便于后期训练。分割后，每个文件包含不多于16000条记录<br>
②分句：首先按照符号['。', '！', '？']分句，若得到的句数少于2句，则用['，', '；']进一步分句<br>
在bertsum-chinese/src目录下运行：<br>
<code>python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log</code>

STEP 4：句子标注<br>
找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)<br>
在bertsum-chinese/src目录下运行：<br>
<code>python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log</code>
<br><br>
## ----------NLPCC2017的预处理-----------
①将NLPCC2017数据集的格式整理成整理成bertsum-chinese/raw_data/LCSTS_test.json文件中数据对应格式<br>
②相应文件名／路径名也要做调整：-bert_data_path ../bert_data/NLPCC -log_file NLPCC_oracle<br>
③调整完后，预处理部分从STEP 3开始即可<br>
<br>

## ------------模型训练------------
提醒：在最开始的时候需要使用单个GPU以便下载BERT model<br>
将-visible_gpus 0,1,2 -gpu_ranks 0,1,2 -world_size 3 改为 -visible_gpus 0 -gpu_ranks 0 -world_size 1<br>
下载完成后你可以关闭进程然后重新使用多GPU来跑代码<br>
<br><br>
在bertsum-chinese/src目录下运行：<br>
三行代码的区别是参数-encoder中设置了不同值，分别是：Classifier、Transformer、RNN，
分别代表三个不同的摘要层<br>
①BERTSUM+Classifier model:<br>
<code>
python train_LAI.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 10000
</code>
<br><br>
②BERTSUM+Transformer model:<br>
<code>
python train_LAI.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
</code>
<br><br>
③BERTSUM+RNN model:<br>
<code>
python train_LAI.py -mode train -encoder rnn -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_rnn -lr 2e-3 -visible_gpus 1 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_rnn -use_interval true -warmup_steps 10000 -rnn_size 768 -dropout 0.1
</code>
<br><br>

提醒：如果训练过程被意外中断，可以通过以下代码从某个节点继续训练(-save_checkpoint_steps设置了定期储存模型信息)<br>
以下代码将从第20,000步储存的模型继续训练(示例-encoder 设置为Transformer、Classifier 、 RNN同理)：<br>
<code>
python train_LAI.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 1  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8 -train_from ../models/bert_transformer/model_step_20000.pt
</code>
<br><br>
注意：当数据集切换成NLPCC2017的时候，需要将/bert_data/LCSTS改成/bert_data/NLPCC(这取决于你的文件名)<br>
<br><br>
## ------------模型评估------------
在bertsum-chinese/src目录下运行：<br>
<code>
python train_LAI.py -mode test -bert_data_path ../bert_data/LCSTS -model_path MODEL_PATH -visible_gpus 1 -gpu_ranks 0 -batch_size 30000 -log_file LOG_FILE -result_path ../results/LCSTS -test_all -block_trigram False -test_from ../models/bert_transformer/model_step_30000.pt
</code>
<br><br>

注意：<br>
①MODEL_PATH 是储存checkpoints的目录<br>
②RESULT_PATH is where you want to put decoded summaries (default ../results/LCSTS)<br>
<br><br>

## ------------Streamlit实现Web应用------------
在bertsum-chinese目录下运行：<br>
<code>streamlit run app.py --browser.serverAddress '127.0.0.1'</code>
(更多资料参考Streamlit官方文档)<br>
应用的左侧是模型参数，你可以选择你喜欢的摘要层(Transformer、Classifier 、 RNN)进行中文文本摘要
且可以自定义生成摘要的句子数目，分别是一句，两句和三句。<br>
![效果图](https://i.postimg.cc/xdz0cTkJ/20220626165852.png "效果图")
<br><br>
<br><br>
# ------------实验结果------------
![结果图1](https://i.postimg.cc/Vv1MKhyg/20220626165901.png "结果图1")
[![结果图2](https://i.postimg.cc/85bszP3d/20220626165905.png)](https://postimg.cc/zbVqd5q3)
[![结果图3](https://i.postimg.cc/wxfTfPdB/20220626165910.png)](https://postimg.cc/s1Wr1TxF)
[![结果图4](https://i.postimg.cc/d0YqZcG5/20220626165914.png)](https://postimg.cc/75Kvjcsz)
