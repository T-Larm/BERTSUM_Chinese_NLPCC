# ====BERT-based Chinese Text Summarization====
# ----------------References--------------
References：https://github.com/Machine-Tom/bertsum-chinese-LAI<br>
Based on the paper 'Fine-tune BERT for Extractive Summarization' method and source code, experiments were conducted on Chinese datasets NLPCC2017 and LCSTS.<br>
Paper author's website：http://nlp-yang.github.io/ 
<br><br>
# ----------------File Description--------------
**Folder raw_data**: JSON file to store the dataset <br>
**Folder json_data**: Store the dataset after segmentation and sentence splitting in a JSON file <br>
**Folder bert_data**: Stores the pt files, converting JSON format to pt format for convenient training (sentence labeling, see the instructions later) <br>
**src/preprocess.py**: The executable file for processing the dataset <br>
<br>
**Folder src/models**: Stores the code for BERTSUM and the summarization layers (Classifier, Transformer, RNN) <br>
**src/train.py**: The executable file for model training and evaluation (ROUGE) <br>
**Folder bertsum-chinese/models**: Stores the training data for models trained with three summarization layers (Classifier, Transformer, RNN) <br>
<br>
**predict.py**: The executable file for predicting summaries<br>
**app.py**: The executable file for the Web application implemented using Streamlit<br>
<br>
**bert-config.json**: Model parameter configuration file <br>
<br><br>
# ----------------Operating environment--------------
* Ubuntu	18.04
* CUDA	10.1
## --------------Requirement------------
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
# ----------------Instructions--------------
## ------------Preparation of the Dataset------------
Chinese Dataset 1: LCSTS2.0(A Large Scale Chinese Short Text Summarization Dataset)<br>
Source：Intelligent Computing Research Center, Harbin Institute of Technology Shenzhen Graduate School<br>
(哈尔滨工业大学深圳研究生院·智能计算研究中心)<br>
Access:http://icrc.hitsz.edu.cn/Article/show/139.html<br>

Chinese Dataset 2: NLPCC2017<br>
Source: Official dataset provided by the task "Chinese News Document Summarization" at the International Conference on Natural Language Processing and Chinese Computing (NLPCC), organized by the China Computer Federation (CCF)<br>
Access: http://tcci.ccf.org.cn/conference/2017/taskdata.php (Task 3:  Single Document Summarization)<br>
<br><br>
## -------------Preprocessing of LCSTS---------------
STEP 1: First, download the LCSTS 2.0 raw data. Place all the **PART_*.txt** files from the LCSTS 2.0/DATA directory into **bertsum-chinese/raw_data**<br>

STEP 2: Convert the raw files into JSON format and store them. Run the following command in the **bertsum-chinese/src** directory:<br>
<code>python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log</code>
<br>

STEP 3: Segment the files and split sentences<br>
①File Segmentation: The training files are too large, so they are divided into smaller files for easier training. Each file contains no more than 16,000 records<br>
②Sentence Splitting: First, split the sentences using the symbols ['。', '！', '？']. If the number of sentences obtained is less than 2, further split using ['，', '；']<br>
Run the following command in the **bertsum-chinese/src** directory: <br>
<code>python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log</code>
<br>

STEP 4: Sentence Labeling<br>
Find the n sentences that are most similar to the reference summary (similarity measured by ROUGE) and label them as 1 (belong to the summary)<br>
Run the following command in the **bertsum-chinese/src** directory: <br>
<code>python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log</code>
<br><br>
## ----------Preprocessing of NLPCC2017-----------
①Organize the NLPCC2017 dataset format into the corresponding format in the **bertsum-chinese/raw_data/LCSTS_test.json** file<br>
②Adjust the corresponding file names/path names: -bert_data_path ../bert_data/NLPCC -log_file NLPCC_oracle<br>
③After making the adjustments, the preprocessing part can start from STEP 3<br>
<br>

## ------------Model Training------------
Note: At the beginning, use a single GPU to download the BERT model<br>
Change **-visible_gpus 0,1,2 -gpu_ranks 0,1,2 -world_size 3** to **-visible_gpus 0 -gpu_ranks 0 -world_size 1**to run the code with a single GPU for the purpose of downloading the BERT model<br>
After the download is complete, you can close the process and then use multiple GPUs to run the code<br>
<br><br>
In the **bertsum-chinese/src** directory, run the following commands:<br>
The difference between the three lines of code is the value set for the -encoder parameter, which is set to different values: Classifier, Transformer, RNN, representing three different summarization layers, respectively
<br><br>
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

Reminder: If the training process is unexpectedly interrupted, you can continue training from a specific checkpoint using the following code (-save_checkpoint_steps is set to save model information periodically)<br>
The code below demonstrates how to continue training from the model saved at step 20,000 (the same applies to setting -encoder as Transformer, Classifier, or RNN):<br>
<code>
python train_LAI.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/LCSTS -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 1  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 30000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8 -train_from ../models/bert_transformer/model_step_20000.pt
</code>
<br><br>
Note: When switching the dataset to NLPCC2017, you need to change /bert_data/LCSTS to /bert_data/NLPCC (or whichever is your actual file name) in the path<br>
<br><br>
## ------------Model Evaluation------------
In the **bertsum-chinese/src** directory, run the following command:<br>
<code>
python train_LAI.py -mode test -bert_data_path ../bert_data/LCSTS -model_path MODEL_PATH -visible_gpus 1 -gpu_ranks 0 -batch_size 30000 -log_file LOG_FILE -result_path ../results/LCSTS -test_all -block_trigram False -test_from ../models/bert_transformer/model_step_30000.pt
</code>
<br><br>

NOTE:<br>
①MODEL_PATH is the directory where checkpoints are stored<br>
②RESULT_PATH is where you want to put decoded summaries (default ../results/LCSTS)<br>
<br><br>

## ------------Implementing Web Application with Streamlit------------
In the bertsum-chinese directory, run the following command: <br>
<code>streamlit run app.py --browser.serverAddress '127.0.0.1'</code>
(For more information, refer to the official documentation of Streamlit)<br>


<br><br>
<br><br>
