import os
#from app import load_f_model

#返回绝对路劲
root = os.path.abspath(os.path.dirname(__file__))

bert_based_chinese = os.path.join(root, 'bert-base-chinese/')

# run device
# or cuda
device = 'cpu'

# model
max_summary_size = 128

load_from = os.path.join(root, 'models/bert_transformer/model_step_30000.pt')

#print(load_from)#路径对
vocab_path = os.path.join(bert_based_chinese, 'vocab.txt')
bert_config_path = os.path.join(bert_based_chinese, 'config.json')

# web
#iphost = '127.0.0.1'
#port = 8080