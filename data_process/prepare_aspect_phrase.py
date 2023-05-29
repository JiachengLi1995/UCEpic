"""
    Recover ref sentences from token ids

    ref_sent.json: list of sentences
"""
# load files

import os
import sys 
import json
from uctopic import UCTopicTool

from tqdm import tqdm
from nltk import tokenize

dataset = sys.argv[1] if len(sys.argv) > 1 else "beer"
gpu_id = sys.argv[2] if len(sys.argv) > 2 else "0"
print(f"generating sentences for {dataset} ...")

train = [json.loads(line) for line in open(os.path.join(dataset, "train.json"))]
dev = [json.loads(line) for line in open(os.path.join(dataset, "dev.json"))]
test = [json.loads(line) for line in open(os.path.join(dataset, "test.json"))]

data = train + dev + test

# pre-process 

sent_dict = dict()

for line in tqdm(data):
    user_id = line["user_id"]
    item_id = line["item_id"]

    text = line["text"]

    sentences = tokenize.sent_tokenize(text)
    sent_dict[str((user_id, item_id))] = sentences
    
# save 
json.dump(sent_dict, open(os.path.join(dataset, "sent_dict.json"), "w", encoding='utf8'))

ref_sent = []

for sents in sent_dict.values():
    ref_sent += sents

topic_tool = UCTopicTool('JiachengLi/uctopic-base', device=f'cuda:{gpu_id}')
output_data, topic_phrase_dict = topic_tool.topic_mining(ref_sent, n_clusters=[4, 16])
output_dict = {line[0]: line[1] for line in output_data}

json.dump(output_dict, open(os.path.join(dataset, "sent2topic.json"), "w"))