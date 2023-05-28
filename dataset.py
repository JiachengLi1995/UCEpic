import os
import json
import random
import numpy as np
import torch

from collections import namedtuple
from argparse import ArgumentParser
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

ExtractionFeatures = namedtuple(
    "ExtractionFeatures", "input_ids input_masks decoder_input_ids decoder_attention_mask tag_labels")

InsertionFeatures = namedtuple("InsertionFeatures", "input_ids input_masks aspects aspects_mask decoder_input_ids_s decoder_attention_mask_s \
                                                    decoder_input_ids_m decoder_attention_mask_m ins_labels lm_labels")

REF_LEN = 128


def _debug_tagging(ids, tags, tokenizer):
    source = []
    phrase = []
    for i, t in zip(ids, tags):
        if t == 1:
            if len(phrase):
                source.append(phrase)
            phrase = [i]
        elif t == 2:
            phrase.append(i)
    if len(phrase):
        source.append(phrase)
    return [tokenizer.decode(s) for s in source]


def construct_reference(user, item, ref=None, tokenizer=None):

    item_tokens = []
    for u in ref['item'][item]:
        if u != user:
            item_tokens.extend(ref['review'][ref['item'][item][u]])
        if len(item_tokens) >= (REF_LEN * 2 // 3):
            break

    user_tokens = []

    for i in ref['user'][user]:
        if i != item:
            user_tokens.extend(ref['review'][ref['user'][user][i]])
        if len(user_tokens) >= (REF_LEN * 2 // 3):
            break

    item_tokens = [tokenizer.bos_token_id] + item_tokens + \
        [tokenizer.sep_token_id]  # add special token

    if len(user_tokens) + len(item_tokens) < REF_LEN:
        tokens = np.array(item_tokens + user_tokens + [tokenizer.pad_token_id] * (
            REF_LEN - len(user_tokens) - len(item_tokens)))
    else:
        tokens = np.array(item_tokens + user_tokens)[: REF_LEN]

    mask = np.zeros_like(tokens, dtype=np.int32)
    mask[: (len(user_tokens) + len(item_tokens))] = 1
    return tokens, mask

def construct_phrase_reference(user, item, ref=None, tokenizer=None):
    
    user_phrases = []

    for i in ref['user'][user]:
        if i != item:
            sentences = ref["sent_dict"][str((user, i))]
            for sent in sentences:
                if sent in ref["sent2topic"]:
                    user_phrases += [sent[ele[0]:ele[1]] for ele in ref["sent2topic"][sent]]
            
        if len(user_phrases) > REF_LEN // 2: break

    item_phrases = []
    for u in ref['item'][item]:
        if u != user:
            sentences = ref["sent_dict"][str((u, item))]
            for sent in sentences:
                if sent in ref["sent2topic"]:
                    item_phrases += [sent[ele[0]:ele[1]] for ele in ref["sent2topic"][sent]]

        if len(item_phrases) > REF_LEN // 2: break

    user_phrase_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(user_phrases)))
    item_phrase_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(item_phrases)))

    phrase_tokens = user_phrase_tokens[:REF_LEN//2] + item_phrase_tokens[:REF_LEN//2]

    if len(phrase_tokens) < REF_LEN:
        tokens = np.array(phrase_tokens + [tokenizer.pad_token_id] * (REF_LEN - len(phrase_tokens)))
    else:
        tokens = np.array(phrase_tokens)[:REF_LEN]

    mask = np.zeros_like(tokens, dtype=np.int32)
    mask[: len(tokens)] = 1
    return tokens, mask

def construct_reference_embeddings(user, item, max_seq_length, sent2order, sent2vec, ref):

    ref_input_array = np.zeros([max_seq_length, 768])
    ref_input_mask = np.zeros([max_seq_length], dtype=np.int32)
    ref_input_mask[0] = 1 ## if all zeros, multiatt will output nan

    point = 0
    for i in ref['user'][user]:
        if i!=item:
            start, end = sent2order[str((user, i))]
            sent_embeds = sent2vec[start:end]
            sent_num = end - start
            ref_input_array[point:point+sent_num] = sent_embeds[:max_seq_length-point]
            ref_input_mask[point:point+sent_num] = 1
            point += sent_num

        if point >= (max_seq_length * 2 // 3):
            break

    for u in ref['item'][item]:
        if u != user:
            start, end = sent2order[str((u, item))]
            sent_embeds = sent2vec[start:end]
            sent_num = end - start
            ref_input_array[point:point+sent_num] = sent_embeds[:max_seq_length-point]
            ref_input_mask[point:point+sent_num] = 1
            point += sent_num

        if point >= max_seq_length:
            break

    return ref_input_array, ref_input_mask

def construct_aspects(user, item, max_seq_length, ref):

    aspects = np.zeros([max_seq_length], dtype=np.int32)
    aspects_mask = np.zeros([max_seq_length], dtype=np.int32)
    aspects_mask[0] = 1 ## if all zeros, multiatt will output nan

    sentences = ref["sent_dict"][str((user, item))]

    aspects_list = []

    for sent in sentences:

        if sent in ref["sent2topic"]:
            aspects_list += [int(ele[2]) for ele in ref["sent2topic"][sent]]

    aspects_list = list(set(aspects_list))#aspects_list[:1]

    aspects[:len(aspects_list)] = aspects_list[:max_seq_length]
    aspects_mask[:len(aspects_list)] = 1

    return aspects, aspects_mask


def convert_example_to_insertion_features(example, tokenizer, max_seq_length, args, ref, sent2order, sent2vec):

    # ref_input_array, ref_mask_array = construct_reference(
    #     user=example["userid"], item=example["itemid"], ref=ref, tokenizer=tokenizer)
    # ref_input_array, ref_mask_array = construct_reference_embeddings(
    #     example["userid"], example["itemid"], max_seq_length, sent2order, sent2vec, ref)

    ref_input_array, ref_mask_array = construct_phrase_reference(user=example["userid"], item=example["itemid"], ref=ref, tokenizer=tokenizer)

    aspects, aspects_mask = None, None#construct_aspects(example["userid"], example["itemid"], args.num_aspects, ref)

    source_ids = example["source"]
    ins_label = [args.max_ins-1 if i >= args.max_ins else i for i in example["ins_label"]]

    ins_label = [-100 if i == 0 and args.zero_ins_mask_prob >
                 random.random() else i for i in ins_label]

    mid_ids = example["mid"]
    lm_label = example["lm_label"]

    if len(source_ids) > max_seq_length:
        source_ids = source_ids[:max_seq_length]
        ins_label = ins_label[:max_seq_length]

    if len(mid_ids) > max_seq_length:
        mid_ids = mid_ids[:max_seq_length]
        lm_label = lm_label[:max_seq_length]

    # The preprocessed data should be already truncated
    assert len(source_ids) == len(ins_label) <= max_seq_length
    assert len(mid_ids) == len(lm_label) <= max_seq_length

    # Source sequences

    source_array = np.full(max_seq_length, dtype=np.int32,
                           fill_value=tokenizer.pad_token_id)
    source_array[:len(source_ids)] = source_ids

    source_mask_array = np.zeros(max_seq_length, dtype=np.int32)
    source_mask_array[:len(source_ids)] = 1

    ins_label_array = np.full(max_seq_length, dtype=np.int32, fill_value=-100)
    ins_label_array[:len(ins_label)] = ins_label

    # Mid sequences

    mid_array = np.full(max_seq_length, dtype=np.int32,
                        fill_value=tokenizer.pad_token_id)
    mid_array[:len(mid_ids)] = mid_ids

    mid_mask_array = np.zeros(max_seq_length, dtype=np.int32)
    mid_mask_array[:len(mid_ids)] = 1

    lm_label_array = np.full(max_seq_length, dtype=np.int32, fill_value=-100)
    lm_label_array[:len(lm_label)] = lm_label

    features = InsertionFeatures(input_ids=ref_input_array,
                                 input_masks=ref_mask_array,
                                 aspects=aspects,
                                 aspects_mask=aspects_mask,
                                 decoder_input_ids_s=source_array,
                                 decoder_attention_mask_s=source_mask_array,
                                 decoder_input_ids_m=mid_array,
                                 decoder_attention_mask_m=mid_mask_array,
                                 ins_labels=ins_label_array,
                                 lm_labels=lm_label_array)
    return features


def convert_example_to_extraction_features(example, tokenizer, max_seq_length, ref=None, mode='train', args=None):

    user = example["userid"]
    item = example["itemid"]

    # user tokens as reference

    user_tokens = []

    for i in ref['user'][user]:
        if i == item:
            continue
        tmp_ref = ref['review'][ref['user'][user][i]]

        if len(user_tokens) + len(tmp_ref) >= max_seq_length - 2:
            user_tokens.extend(
                tmp_ref[-(max_seq_length - 2 - len(user_tokens)):])
            break
        user_tokens.extend(tmp_ref)

    user_tokens = [tokenizer.bos_token_id] + \
        user_tokens + [tokenizer.sep_token_id]

    # item tokens as tagging input

    item_tags = np.zeros(len(example['token_ids']), dtype=np.int32)  # 0 -> "O"

    # item tagging ground truth
    item_tags_gt = list(item_tags)

    # tagging propagation

    # concat with other item reviews

    item_tokens_gt = example['token_ids']
    item_token_cnt = len(item_tokens_gt)
    item_token_list, item_tokens = [], []

    for u in ref['item'][item]:
        if u == user:
            continue

        tmp_ref = ref['review'][ref['item'][item][u]]

        if item_token_cnt + len(tmp_ref) >= max_seq_length - 2:
            tmp_ref = tmp_ref[-(max_seq_length - 2 - item_token_cnt):]
            item_token_list.append(tmp_ref)
            item_token_cnt += len(tmp_ref)
            break
        item_token_list.append(tmp_ref)
        item_token_cnt += len(tmp_ref)

    # add gt into random position

    insertp = random.randint(0, len(item_token_list))
    item_tags = []
    for i, tokens in enumerate(item_token_list):
        item_tokens.extend(tokens)
        item_tags.extend([0] * len(tokens))
        if i == insertp:
            item_tokens.extend(item_tokens_gt)
            item_tags.extend(item_tags_gt)

    if insertp == len(item_token_list):
        item_tokens.extend(item_tokens_gt)
        item_tags.extend(item_tags_gt)

    # label propagation

    for b, e in example['phrase_pos']:
        phrase = item_tokens_gt[b:e+1]
        tag = [1] + [2] * (e-b)
        length = len(phrase)
        for s in range(len(item_tokens)-length):
            phrase_test = item_tokens[s: s+length]
            if phrase_test == phrase:
                item_tags[s: s+length] = tag

    # 'O' masking

    if mode == 'train':
        item_tags = [-100 if i == 0 and args.zero_tag_mask_prob >
                     random.random() else i for i in item_tags]

    # item tokens and item tags

    item_tokens = [tokenizer.bos_token_id] + \
        item_tokens + [tokenizer.sep_token_id]
    item_tags = [-100] + item_tags + [-100]

    # padding as numpy array

    source_array = np.full(max_seq_length, dtype=np.int32,
                           fill_value=tokenizer.pad_token_id)
    source_array[:len(item_tokens)] = item_tokens

    source_mask_array = np.zeros(max_seq_length, dtype=np.int32)
    source_mask_array[: len(item_tokens)] = 1

    ref_input_array = np.full(
        max_seq_length, dtype=np.int32, fill_value=tokenizer.pad_token_id)
    ref_input_array[:len(user_tokens)] = user_tokens

    ref_mask_array = np.zeros(max_seq_length, dtype=np.int32)
    ref_mask_array[: len(user_tokens)] = 1

    item_tags_array = np.full(max_seq_length, dtype=np.int32, fill_value=-100)
    item_tags_array[:len(item_tags)] = item_tags

    features = ExtractionFeatures(input_ids=ref_input_array,
                                  input_masks=ref_mask_array,
                                  decoder_input_ids=source_array,
                                  decoder_attention_mask=source_mask_array,
                                  tag_labels=item_tags_array)
    return features


class FinetuningDataset(Dataset):
    def __init__(self, path, mode='train', tokenizer=None, args=None):

        # arguments
        self.path = path
        self.mode = mode  # train or dev or test
        self.tokenizer = tokenizer
        self.args = args

        # data loading
        self.data = [json.loads(l) for l in open(
            os.path.join(path, f"{mode}.json"), "r")]

        # ref loading
        user_ref = json.load(open(os.path.join(path, "ref_user.json"), "r"))
        self.user_num = len(user_ref)
        item_ref = json.load(open(os.path.join(path, "ref_item.json"), "r"))
        self.item_num = len(item_ref)
        review_ref = json.load(
            open(os.path.join(path, "ref_review.json"), "r"))
        sent_dict = json.load(open(os.path.join(path, "sent_dict.json"), "r"))
        sent2topic = json.load(open(os.path.join(path, "sent2topic.json"), "r"))

        self.sent2order = json.load(open(os.path.join(path, 'sent2order.json'), "r"))
        self.sent2vec = np.load(os.path.join(path, 'sent2vec.npy'))

        # ranking ref loading
        if self.args.keywords == 'ranking':
            self.item_keywords = json.load(
                open(os.path.join(path, "ranking", "ref_item_keywords.json"), "r"))
            self.item_keywords = {
                i: set([tuple(k) for k in self.item_keywords[i]]) for i in self.item_keywords}
            self.keyword_max_len = args.keyword_max_len
            self.user2id = dict(
                zip(list(user_ref.keys()), range(self.user_num)))

        self.ref = {
            "item": item_ref,
            "user": user_ref,
            "sent_dict": sent_dict,
            "sent2topic": sent2topic,
            "review": review_ref
        }

    def __getitem__(self, index):
        # data sample
        sample = self.data[index]

        insertion_feature = self._insert(sample)

        return (
            torch.LongTensor(insertion_feature.input_ids),
            torch.LongTensor(insertion_feature.input_masks),
            torch.LongTensor(insertion_feature.aspects),
            torch.LongTensor(insertion_feature.aspects_mask),
            torch.LongTensor(insertion_feature.decoder_input_ids_s),
            torch.LongTensor(insertion_feature.decoder_attention_mask_s),
            torch.LongTensor(insertion_feature.decoder_input_ids_m),
            torch.LongTensor(insertion_feature.decoder_attention_mask_m),
            torch.LongTensor(insertion_feature.ins_labels),
            torch.LongTensor(insertion_feature.lm_labels),
        )

    def _extract(self, sample):

        user = sample['user_id']
        item = sample['item_id']

        example = sample['tagging_data']
        example.update({'userid': user, 'itemid': item})

        features = convert_example_to_extraction_features(
            example, self.tokenizer, self.args.max_len, ref=self.ref, mode=self.mode, args=self.args)

        return features

    def _aspect(self, user, item):

        aspects = np.zeros([1], dtype=np.int32)
        aspects_mask = np.zeros([1], dtype=np.int32)
        aspects_mask[0] = 1

        first_stage_data = {"source": [0, 2], "mid": [0, 2], "lm_label": [-100, -100], "ins_label": [0, -100]}

        sentences = self.ref["sent_dict"][str((user, item))]

        aspects_list = []

        for sent in sentences:
            if sent in self.ref["sent2topic"]:
                aspects_list += [[sent[ele[0]:ele[1]], int(ele[2])] for ele in self.ref["sent2topic"][sent]]

        if len(aspects_list) > 0:
            sampled_aspect = aspects_list[random.randint(0, len(aspects_list)-1)]

            phrase_mention, phrase_aspect = sampled_aspect
            phrase_tokens = self.tokenizer.encode(phrase_mention)[1:-1]
            first_stage_data["lm_label"] = [-100]+phrase_tokens+[-100]
            first_stage_data["ins_label"][0] = len(phrase_tokens)
            first_stage_data["mid"] = [0] + [self.tokenizer.mask_token_id]*len(phrase_tokens) + [2]
            aspects[0] = phrase_aspect

        return first_stage_data, aspects, aspects_mask

    def _insert(self, sample):

        user = sample['user_id']
        item = sample['item_id']
        insertions = sample['insertion_data']

        first_stage_data, aspects, aspects_mask = self._aspect(user, item)

        default_aspects = np.zeros([1], dtype=np.int32)
        default_aspects_mask = np.zeros([1], dtype=np.int32)

        default_aspects[0] = 64
        default_aspects_mask[0] = 1

        example_candidates = [[first_stage_data, aspects, aspects_mask]]

        for _ in range(3):

            example_candidates.append([insertions[random.randint(0, len(insertions)-1)], default_aspects, default_aspects_mask])

        example_tuple = example_candidates[random.randint(0, len(example_candidates)-1)].copy()

        example, aspects, aspects_mask = example_tuple
        
        example.update({'userid': user, 'itemid': item})

        features = convert_example_to_insertion_features(
            example, self.tokenizer, self.args.max_len, args=self.args, ref=self.ref, sent2order=self.sent2order, sent2vec=self.sent2vec)

        new_features = InsertionFeatures(input_ids=features.input_ids,
                                        input_masks=features.input_masks,
                                        aspects=aspects,
                                        aspects_mask=aspects_mask,
                                        decoder_input_ids_s=features.decoder_input_ids_s,
                                        decoder_attention_mask_s=features.decoder_attention_mask_s,
                                        decoder_input_ids_m=features.decoder_input_ids_m,
                                        decoder_attention_mask_m=features.decoder_attention_mask_m,
                                        ins_labels=features.ins_labels,
                                        lm_labels=features.lm_labels)

        return new_features

    def _rank(self, sample):

        user = sample['user_id']
        item = sample['item_id']
        keywords = self.__extract_phrase(
            token_ids=sample['tagging_data']['token_ids'],
            phrase_pos=sample['tagging_data']['phrase_pos']
        )
        pos_set = set([tuple(k) for k in keywords])
        neg_set = list(self.item_keywords[item] - pos_set)

        cands = []

        # pos
        pos_keyword = list(random.choice(keywords)) if len(keywords) else []
        pos_keyword = pos_keyword[:self.keyword_max_len] + (
            self.keyword_max_len - len(pos_keyword)) * [self.tokenizer.pad_token_id]
        cands.append(pos_keyword)

        # neg
        for _ in range(1):  # can be more
            neg_keyword = list(random.choice(neg_set)) if len(neg_set) else []
            neg_keyword = neg_keyword[:self.keyword_max_len] + (
                self.keyword_max_len - len(neg_keyword)) * [self.tokenizer.pad_token_id]
            cands.append(neg_keyword)

        # import pdb; pdb.set_trace()

        return self.user2id[user], cands

    def __extract_phrase(self, token_ids, phrase_pos):
        phrases = []
        for b, e in phrase_pos:
            phrase = tuple(token_ids[b:e+1])
            phrases.append(phrase)
        return phrases

    def _keywords(self, sample):

        user = sample['user_id']
        item = sample['item_id']
        example = sample['phrase_data']
        example.update({'userid': user, 'itemid': item})

        features = convert_example_to_insertion_features(
            example, self.tokenizer, self.args.max_len, args=self.args, ref=self.ref)

        return features

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--input_path', type=str, required=False)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--mlm_prob', type=float, default=0.2)
    parser.add_argument('--sub_prob', type=float, default=0.1)
    parser.add_argument('--mask_prob', type=float, default=0.8)
    parser.add_argument('--min_mask', type=int, default=2)
    parser.add_argument('--zero_ins_mask_prob', type=float, default=0.9)

    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(
        'roberta-base', use_fast=False)

    dataset = FinetuningDataset("data/yelp", tokenizer=tokenizer, args=args)
    dataset = FinetuningDataset(
        "data/yelp", mode='dev', tokenizer=tokenizer, args=args)
