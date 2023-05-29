from argparse import ArgumentParser
import os
import torch
import logging

from model import UCEpicForPretraining
from utils import boolean_string, random_seed

from transformers import RobertaTokenizer, RobertaConfig

# api for conditional generation

class Generator(object):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer 
        self.args = args 

    def __call__(self, source, mask=None, is_decoded=True, force=0):
        """
        :source list: if need_encode, source is the list of keyphrases to generate a sentence, e.g. ['good morning', 'love'] 
        :source list: if not need_encode, source is the token ids, e.g. [0, 8396, 662, 17693, 2]
        :mask list: mask is the position of protected token, i.e., not allowed to append new tokens after this position, 
                    e.g. [1, 4], means we cannot append tokens after [8396], so we cannot break 'good' (and 'morning'), and 
                    cannot append tokens after '</s>'.
        """
        input_ids, new_mask = self._encode(source)
        mask = new_mask if mask is None else new_mask
        loop_num = 0

        while input_ids.size(-1) <= self.args.max_seq_len and loop_num < 10:
            # predict insertion positions and numbers 
            outputs = self.model.ins_forward(input_ids)
            insert_pos = outputs.ins_logits.argmax(dim=-1) + force # TODO: never use force!
            insert_pos[..., mask] = 0
            
            # insertion augmentation
            input_ids_, mask = self._insertion_aug(input_ids, mask, insert_pos)

            # predict insertion tokens
            outputs = self.model.lm_forward(input_ids_)
            insert_token = outputs.lm_logits.argmax(dim=-1) * (input_ids_ == self.tokenizer.mask_token_id)
            input_ids = insert_token + (input_ids_ * (input_ids_ != self.tokenizer.mask_token_id))

            # loop counter
            loop_num += 1 

            print(f"TURN [{loop_num}]: {self.tokenizer.decode(input_ids.flatten().tolist())}" )

        if is_decoded:
            return self.tokenizer.decode(input_ids.flatten().tolist())
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", 
                        type=str, required=True, 
                        help="Pretrained model checkpoint, e.g. pretrained_model/wiki/checkpoint-14840")
    parser.add_argument("--bert_model", 
                        type=str, default='roberta-base', 
                        help="RoBerta pre-trained model or trained RefineModel, e.g. roberta-base")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", 
                        type=boolean_string, 
                        default=False, 
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--max_ins", 
                        type=int, 
                        default=7, 
                        help="maximum placeholder number")
    parser.add_argument("--max_seq_len",
                        type=int, 
                        default=32, 
                        help="maximum sentence length")    
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

    # Set logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger = logging.getLogger(__name__)

    # Set seed
    random_seed(args.seed)
    args.output_mode = "classification"

    # Set tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, use_fast=False)
    checkpoint = torch.load(os.path.join(args.checkpoint, 'checkpoint.bin'))

    config = RobertaConfig.from_pretrained(args.bert_model)
    config.num_labels = args.max_ins if 'max_ins' not in checkpoint else checkpoint['max_ins']
    model = UCEpicForPretraining.from_pretrained(args.bert_model, config=config)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    print(args)

    generator = Generator(model=model, tokenizer=tokenizer, args=args)
    
    import pdb; pdb.set_trace()
    
    print(generator(source=["morning"]))