from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import random

from transformers import RobertaTokenizer, RobertaConfig
import torch.nn.functional as F

from utils import boolean_string
from model import UCEpic
from dataset import construct_phrase_reference, construct_aspects, construct_reference
from nltk.corpus import stopwords
import numpy as np

PREVENT_FACTOR = 0.1
PROMOTE_FACTOR = 1.1
PREVENT_LIST = ['<unk>', '"',"(",")","-","[","]","'","&", "/"]  
STOP_LIST = set([s.lower() for s in stopwords.words('english')]) | set(['</s>', '<pad>', '<s>', 'the', 'of', 'and', 'in', 'a', 'to', 'was', 'is', '"', 'for', 'on', 'as', 'with', 'by', 'he', "'s", 'at', 'that', 'from', 'it', 'his', 'an', 'which', 's', '.', ',', '(', ')',"'", '%'])
REDUCE_LIST = set(["'",'s','.',","]) 

REF_LEN = 128

REF = None

'''
Reference should be tokenized and ids. Don't have bos or eos tokens.
'''

InputFeatures = namedtuple("InputFeatures", "input_ids input_masks decoder_input_ids_s decoder_attention_mask_s \
                                            decoder_input_ids_m decoder_attention_mask_m ins_labels lm_labels")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def sample_insertion(insert_pos, k):
    non_zero = insert_pos.nonzero()[:, 1].cpu().detach().tolist()
    if len(non_zero)>0:
        selected_idxs = random.sample(non_zero, k=min(k, len(non_zero)))
        pos_mask = torch.zeros_like(insert_pos).to(insert_pos.device)
        pos_mask[:, selected_idxs] = 1
        insert_pos = pos_mask * insert_pos

    return insert_pos

# api for conditional generation

class Generator(object):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer 
        self.args = args

        self.prevent = None
        if args.prevent:
            self.prevent = []
            for x in PREVENT_LIST:
                self.prevent+= tokenizer.encode(x)[1:-1]

        if args.reduce_stop:
            reduce_l = REDUCE_LIST |  STOP_LIST
        self.reduce = None
        if args.prevent:
            self.reduce = [tokenizer.convert_tokens_to_ids(x) for x in reduce_l]

    def __call__(self, gt=None, item_keywords=None, ref_input_array=None, ref_mask_array=None, aspects=None, aspects_mask=None):
        
        source = item_keywords# if item_keywords is not None else [p[-1] for p in rule_based_noun_phrase(gt)]
        
        if self.args.control == 'soft':
            source = []
        else:
            if self.args.key_num != -1:
                source = source[:self.args.key_num]
        
        res = self._insert(source, ref_input_array=ref_input_array, ref_mask_array=ref_mask_array, aspects=aspects, aspects_mask=aspects_mask)
        return res


    def _random(self, item_keywords, k=5):
        if len(item_keywords) == 0:
            return []
        k_ = random.randint(1, min(k, len(item_keywords)))
        keywords = random.choices(item_keywords, k=k_)
        source = [self.tokenizer.decode(k).strip() for k in keywords]
        return source

    def _insert(self, source, is_decoded=True, force=0, ref_input_array=None, ref_mask_array=None, aspects=None, aspects_mask=None):
        """
        :source list: if need_encode, source is the list of keyphrases to generate a sentence, e.g. ['good morning', 'love'] 
        :source list: if not need_encode, source is the token ids, e.g. [0, 8396, 662, 17693, 2]
        :mask list: mask is the position of protected token, i.e., not allowed to append new tokens after this position, 
                    e.g. [1, 4], means we cannot append tokens after [8396], so we cannot break 'good' (and 'morning'), and 
                    cannot append tokens after '</s>'.
        """

        ref_input_array = torch.LongTensor(ref_input_array).unsqueeze(dim=0).to(self.args.device)
        ref_mask_array = torch.LongTensor(ref_mask_array).unsqueeze(dim=0).to(self.args.device)

        aspects = torch.LongTensor(aspects).unsqueeze(dim=0).to(self.args.device)
        aspects_mask = torch.LongTensor(aspects_mask).unsqueeze(dim=0).to(self.args.device)

        input_ids, mask = self._encode(source)
        
        loop_num = 0

        first_stage = True

        with torch.no_grad():

            while input_ids.size(-1) <= self.args.max_seq_len and loop_num < 10:

                if not first_stage or self.args.control == 'hard':
                    aspects[:, 0] = 64
                    aspects_mask[:, 0] = 1

                first_stage = False
                
                attention_mask = torch.ones([1, input_ids.size(1)]).to(self.args.device)
                outputs = self.model.ins_forward(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                ref=ref_input_array,
                                                ref_mask=ref_mask_array,
                                                aspects=aspects,
                                                aspects_mask=aspects_mask)

                insert_pos = outputs.ins_logits.argmax(dim=-1) + force # TODO: never use force!
                insert_pos[..., mask] = 0

                if torch.count_nonzero(insert_pos) == 0:
                    break
                
                #insert_pos = sample_insertion(insert_pos, k=1)
                # insertion augmentation
                input_ids_, mask = self._insertion_aug(input_ids, mask, insert_pos)

                # predict insertion tokens
                attention_mask = torch.ones([1, input_ids_.size(1)]).to(self.args.device)
                outputs = self.model.lm_forward(input_ids=input_ids_, 
                                                attention_mask=attention_mask,
                                                ref=ref_input_array,
                                                ref_mask=ref_mask_array,
                                                aspects=aspects,
                                                aspects_mask=aspects_mask)

                lm_logits = outputs.lm_logits

                if self.prevent is not None:
                    #for p in self.prevent:
                    lm_logits[:,:,self.prevent] = -10#lm_logits[:,:,self.prevent] * PREVENT_FACTOR
                
                # if self.reduce is not None:
                #     reduce_factor = min(float(loop_num) / args.reduce_decay, 1.0) 
                #     #for p in self.reduce:
                #     lm_logits[:,:,self.reduce] = lm_logits[:,:,self.reduce] * reduce_factor

                if self.args.lessrepeat:
                    #for p in input_ids_.cpu().numpy()[0]:
                    p_list = input_ids_[0]
                    lm_logits[:,:,p_list] = lm_logits[:,:,p_list] * 0.1

                insert_token = lm_logits.argmax(dim=-1) * (input_ids_ == self.tokenizer.mask_token_id)
                input_ids = insert_token + (input_ids_ * (input_ids_ != self.tokenizer.mask_token_id))

                # loop counter
                loop_num += 1 

                # print(f"TURN [{loop_num}]: {self.tokenizer.decode(input_ids.flatten().tolist())}" )

        if is_decoded:
            return self.tokenizer.decode(input_ids.flatten().tolist()[1:-1])
        return input_ids


    def _insertion_aug(self, input_ids, mask, insert_pos):
        i = 0
        input_ids_, mask_ = [], []
        for j, (input_id, pos) in enumerate(zip(input_ids.flatten().tolist(), insert_pos.flatten().tolist())):
            if i < len(mask) and mask[i] == j:
                i += 1
                mask_.append(len(input_ids_)) 
            input_ids_.extend([input_id] + [self.tokenizer.mask_token_id] * pos)
        return torch.LongTensor(input_ids_).unsqueeze(dim=0).to(self.args.device), mask_

    def _encode(self, source):
        mask = []
        try: 
            if len(source):
                if isinstance(source[0], str): # list of phrases
                    input_ids = []
                    offset = 1 # will add <s> as the starting token
                    for i, phrase in enumerate(source):
                        phrase = ' '+phrase
                        phrase_ids = self.tokenizer.encode(phrase)[1:-1] # exclude <s> and </s>
                        if len(phrase_ids) > 1:
                            mask.extend([offset + i for i in range(len(phrase_ids)-1)]) # 
                        offset += len(phrase_ids)
                        input_ids.extend(phrase_ids)
                    b, e = self.tokenizer.encode("")
                    input_ids = [b] + input_ids + [e]
                else:
                    input_ids = source # list of token ids (with <s> and </s> idss)
            else: 
                input_ids = self.tokenizer.encode("") # input is nothing
        except TypeError:
            print("Please input a list of keyphrases / token ids !")            

        mask.append(len(input_ids) - 1)
        input_tensor = torch.LongTensor(input_ids).unsqueeze(dim=0)

        return input_tensor.to(self.args.device), mask

class FileEvaluator(object):
    def __init__(self, dir_path, output_path, generator, tokenizer, output="generated_end2end"):

        self.ref = {
            "item": json.load(open(os.path.join(dir_path, "ref_item.json"))),
            "user": json.load(open(os.path.join(dir_path, "ref_user.json"))),
            "sent_dict": json.load(open(os.path.join(dir_path, "sent_dict.json"), "r")),
            "sent2topic": json.load(open(os.path.join(dir_path, "sent2topic.json"), "r")),
            "sent2order": json.load(open(os.path.join(dir_path, 'sent2order.json'), "r")),
            "sent2vec": np.load(os.path.join(dir_path, 'sent2vec.npy')),
            "review": json.load(open(os.path.join(dir_path, "ref_review.json"), "r")),
            "item2phrase": json.load(open(os.path.join(dir_path, "item_phrases.json"), "r"))
        }
        
        self.data_file = [json.loads(line) for line in open(os.path.join(dir_path, f"test_robust.json"))]#[:500]
        self.output_file = open(os.path.join(output_path, f"{output}.json"), "w")
        self.generator = generator
        self.tokenizer = tokenizer
        self.args = args

    def _aspect(self, user, item):

        aspects = np.zeros([1], dtype=np.int32)
        aspects_mask = np.zeros([1], dtype=np.int32)
        aspects_mask[0] = 1
        phrase_mention = ''

        sentences = self.ref["sent_dict"][str((user, item))]

        aspects_list = []

        for sent in sentences:
            if sent in self.ref["sent2topic"]:
                aspects_list += [[sent[ele[0]:ele[1]], int(ele[2])] for ele in self.ref["sent2topic"][sent]]

        if len(aspects_list) > 0:
            sampled_aspect = aspects_list[random.randint(0, len(aspects_list)-1)]

            phrase_mention, phrase_aspect = sampled_aspect
            aspects[0] = phrase_aspect

        return aspects, aspects_mask, phrase_mention

    def postprocess(self, res):

        res = res.replace('\n', ' ')
        res = res.replace('\"', '')
        res = res.replace('\t"', ' ')

        return res.strip()

    def hard_constraints(self, item):

        phrases = self.ref["item2phrase"][item]
        
        phrase = phrases[0][0]

        #phrase = phrases[random.randint(0, len(phrases)-1)][0]

        return [phrase]

    def evaluate(self):
        cnt = 0
        for i, l in enumerate(tqdm(self.data_file, ncols=50)): # TODO: debug
            

            user, item = l['user_id'], l['item_id']
            
            #ref_input_array, ref_mask_array = construct_reference(user=user, item=item, ref=self.ref, tokenizer=tokenizer)
            #ref_input_array, ref_mask_array = construct_reference_embeddings(user, item, self.args.max_seq_length, self.ref["sent2order"], self.ref["sent2vec"], self.ref)
            ref_input_array, ref_mask_array = construct_phrase_reference(user=user, item=item, ref=self.ref, tokenizer=self.tokenizer)
            #aspects, aspects_mask = construct_aspects(user, item, self.args.num_aspects, self.ref)
            aspects, aspects_mask, phrase_mention = self._aspect(user, item)

            gt = self.tokenizer.decode(l['insertion_data'][0]['source'][1:-1])

            #self.hard_constraints(item)
            res = self.generator(gt, l[str(self.args.phrase_key)], ref_input_array, ref_mask_array, aspects, aspects_mask)#self.item_keywords[item]) #TODO: debug item not in item_keywords

            res = {"aspect_id": int(aspects[0]), "aspect": phrase_mention, "gen": self.postprocess(res), "gt": gt}
            self.output_file.write(f"{json.dumps(res)}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--keyfile', 
                        type=Path, required=False)
    parser.add_argument("--train_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--pregenerated_data', 
                        type=Path, required=False, 
                        default=None
                        )
    parser.add_argument("--checkpoint", 
                        type=str, required=True, 
                        help="Pretrained model checkpoint, e.g. pretrained_model/wiki/checkpoint-14840")
    parser.add_argument("--bert_model", 
                        type=str, default='roberta-base', 
                        help="RoBerta pre-trained model or trained RefineModel, e.g. roberta-base")
    parser.add_argument("--do_lower_case", 
                        type=boolean_string, 
                        default=False, 
                        )
    parser.add_argument("--reduce_memory",                        
                        type=boolean_string, 
                        default=False, 
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", 
                        type=boolean_string, 
                        default=False, 
                        help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--type",
                        default="greedy",
                        type=str,
                        choices=['greedy','sampling'],
                        help="greedy: greedy generation. sampling: top-k sampling")
    parser.add_argument('--noi_decay',
                        type=int,
                        default=1,
                        help="round number to decay NOI prob") 
    parser.add_argument('--reduce_decay',
                        type=int,
                        default=1,
                        help="round number to decay reduce prob") 
    parser.add_argument('--epoch', type=int,
                        default=59,
                        help="use the last epoch for validation") 
    parser.add_argument('--eval', type=boolean_string,
                        default='false',
                        help="whether to eval the data on pre_generated data") 
    parser.add_argument('--n_test',
                        type=int,
                        default=5000,
                        help="number of test examples")
    parser.add_argument('--prevent', 
                        type=boolean_string, 
                        default=True,
                        help="avoid generating several words")
    parser.add_argument('--reduce_stop',
                        type=boolean_string, 
                        default=True, 
                        help="reduce stopwords")    
    parser.add_argument('--lessrepeat',
                        type=boolean_string, 
                        default=True, 
                        help="reduce repetition (only for tokenwise)")
    parser.add_argument('--sep',
                         type=str, default=" ", help="token to seperate keywords")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help="max sequence length") 
    parser.add_argument("--no_ins_at_first", 
                        type=boolean_string, 
                        default=False, 
                        help="Do not insert at the begining of the text")
    parser.add_argument("--max_ins", 
                        type=int, 
                        default=7, 
                        help="maximum placeholder number")
    parser.add_argument("--max_seq_len",
                        type=int, 
                        default=32, 
                        help="maximum sentence length")    
    parser.add_argument("--keywords",
                        type=str, 
                        default='tagging', 
                        help="selected from `tagging` or `generation`")   
    parser.add_argument("--extraction",
                        type=str, 
                        default='random', 
                        help="selected from `tagging` or `generation` or `gt` or `random`")
    parser.add_argument('--key_num',
                        type=int,
                        default=-1,
                        help="number of keyphrases used for generation, -1 is to use all.") 
    parser.add_argument("--phrase_key",
                        type=str, 
                        default='yake', 
                        help="selected from `tagging` or `generation` or `gt` or `random`")
    parser.add_argument("--num_aspects", type=int, default=100, help="Number of aspects")
    parser.add_argument("--control", type=str, default="soft", help="soft or hard control")
    args = parser.parse_args()


    # Set device
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    args.device = device
    #device = "cpu"

    # Set logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger = logging.getLogger(__name__)

    # Set seed
    set_seed(args)

    args.output_mode = "classification"

    # Set tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, use_fast=False)
    checkpoint = torch.load(os.path.join(args.checkpoint, 'checkpoint.bin'), map_location=args.device)
    
    config = RobertaConfig.from_pretrained(args.bert_model)
    config.num_labels = args.max_ins if 'max_ins' not in checkpoint else checkpoint['max_ins']
    config.num_aspects = args.num_aspects
    model = UCEpic.from_pretrained(args.bert_model, config=config)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    print(args)

    generator = Generator(model=model, tokenizer=tokenizer, args=args)

    if args.eval and args.pregenerated_data is not None:
        print(f"Evaluation on {args.pregenerated_data}")
        # evaluate on file
        file_eval = FileEvaluator(args.pregenerated_data, args.checkpoint, generator, tokenizer, output=f"generated_{args.phrase_key}_bert_{args.control}_{args.key_num}")
        file_eval.evaluate()