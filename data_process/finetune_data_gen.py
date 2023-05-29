from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import json
import os
from collections import defaultdict
from multiprocessing import Pool
import spacy
from processor import SentenceProcessor

from transformers import RobertaTokenizerFast

NLP = spacy.load('en_core_web_sm', disable=['ner'])

'''
Tokenize and recognize phrases
Split data into train, dev, test
Construct reference
Save all files
'''

parser = ArgumentParser()
parser.add_argument('--input_path', 
                    type=Path, required=True, 
                    default=None)
parser.add_argument('--output_dir',
                    type=Path, required=True,
                    default=None)
parser.add_argument('--item_key',
                    type=str, required=False,
                    default='business_id')
parser.add_argument('--user_key',
                    type=str, required=False,
                    default='user_id')
parser.add_argument('--text_key',
                    type=str, required=False,
                    default='text')
parser.add_argument('--max_len',
                    type=int, required=False,
                    default=64)
parser.add_argument('--min_len',
                    type=int, required=False,
                    default=5)
parser.add_argument('--max_ins',
                    type=int, required=False,
                    default=13)
parser.add_argument('--mlm_prob', 
                    type=float, default=0.2)
parser.add_argument('--sub_prob', 
                    type=float, default=0.1)
parser.add_argument('--mask_prob', 
                    type=float, default=0.8)
parser.add_argument('--min_mask', 
                    type=int, default=2)
parser.add_argument('--max_dataline',
                    type=int, required=False,
                    default=1000000)
ARGS = parser.parse_args()
TOKENIZER = RobertaTokenizerFast.from_pretrained('roberta-base')
SENTENCE_PROCESSOR = SentenceProcessor(
    TOKENIZER, ARGS.mlm_prob, ARGS.sub_prob, ARGS.mask_prob, ARGS.min_mask)


class Processor:
    @staticmethod
    def process_data(data):
        processed_data = []
        pool = Pool(processes=16)
        pool_func = pool.imap(func=Processor._preprocess, iterable=data)
        doc_tuples = list(tqdm(pool_func, total=len(
            data), ncols=100, desc=f'Process data and extract phrases'))
        for data_line in doc_tuples:
            if data_line is not None:
                processed_data.append(data_line)
        pool.close()
        pool.join()

        user_item_dict = defaultdict(set)

        for line in tqdm(processed_data, ncols=100, desc='Construct user dict'):
            user_item_dict[line['user_id']].add(line['item_id'])

        return processed_data, user_item_dict

    @staticmethod
    def _preprocess(line):

        text = line[ARGS.text_key]
        user = line[ARGS.user_key]
        item = line[ARGS.item_key]

        token_ids = TOKENIZER.convert_tokens_to_ids(
            TOKENIZER.tokenize(text, add_special_tokens=False))

        if ARGS.min_len > len(token_ids) or len(token_ids) > ARGS.max_len:
            return None

        doc = NLP(text)

        phrases = Processor._rule_base_phrase(doc)

        phrase_ids = [TOKENIZER.convert_tokens_to_ids(
            TOKENIZER.tokenize(p, add_special_tokens=False)) for p in phrases]

        phrase_pos = Processor.list_find_pos(token_ids, phrase_ids)
        insertion_data = Processor._insertion_data_gen(text)
        phrase_data = Processor._phrase_data_gen(phrase_ids)

        tagging_data = {'token_ids': token_ids, 'phrase_pos': phrase_pos}

        data = {'user_id': user, 'item_id': item}
        data['tagging_data'] = tagging_data
        data['insertion_data'] = insertion_data
        data['phrase_data'] = phrase_data
        data['text'] = text
        data['phrases'] = [p.strip() for p in phrases]

        return data

    @staticmethod
    def _phrase_data_gen(phrase_ids):

        flatten_phrase_ids = []
        for line in phrase_ids:
            flatten_phrase_ids += line

        flatten_phrase_ids = flatten_phrase_ids[:ARGS.max_ins]

        source = [TOKENIZER.bos_token_id, TOKENIZER.eos_token_id]
        ins_label = [len(flatten_phrase_ids), -100]
        mid = [TOKENIZER.bos_token_id] + [TOKENIZER.mask_token_id] * \
            len(flatten_phrase_ids) + [TOKENIZER.eos_token_id]
        lm_label = [TOKENIZER.bos_token_id] + \
            flatten_phrase_ids + [TOKENIZER.eos_token_id]

        return {'source': source, 'mid': mid, 'lm_label': lm_label, 'ins_label': ins_label}

    @staticmethod
    def _insertion_data_gen(text):

        results = []
        local_k = 0
        res = SENTENCE_PROCESSOR(text)
        res['k'] = local_k
        if max(res['ins_label']) <= ARGS.max_ins:
            results.append(res)

        while len(res['source']) > ARGS.min_len:
            local_k += 1
            res = SENTENCE_PROCESSOR(res['source'])
            res['k'] = local_k
            if max(res['ins_label']) <= ARGS.max_ins:
                results.append(res)
        return results

    @staticmethod
    def _rule_base_phrase(doc):

        definite_articles = {'a', 'the', 'an', 'this', 'those', 'that', 'these',
                                  'my', 'his', 'her', 'your', 'their', 'our'}
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                left_p = '(' in chunk.text
                right_p = ')' in chunk.text
                if left_p == right_p:
                    ps = chunk.text
                    if ps.split(" ")[0].lower() in definite_articles:
                        new_ps = " ".join(ps.split(" ")[1:])
                        start_char = chunk.start_char + len(ps) - len(new_ps)
                        assert doc.text[start_char:chunk.end_char] == new_ps
                        phrases.append(' '+new_ps)
                    else:
                        if chunk.start_char == 0:
                            phrases.append(chunk.text)
                        else:
                            # if phrase is not the start, add space for roberta tokenizer
                            phrases.append(' '+chunk.text)
            else:
                if doc[chunk.start].pos_ != 'PRON':
                    if chunk.start_char == 0:
                        phrases.append(chunk.text)
                    else:
                        # if phrase is not the start, add space for roberta tokenizer
                        phrases.append(' '+chunk.text)
        return phrases

    @staticmethod
    def list_find_pos(token_ids, phrase_ids):

        if len(phrase_ids) == 0:
            return []

        results = []
        cur_token_pos = 0
        cur_phrase_pos = 0
        while cur_token_pos < len(token_ids) and cur_phrase_pos < len(phrase_ids):

            while cur_token_pos < len(token_ids) and token_ids[cur_token_pos] != phrase_ids[cur_phrase_pos][0]:
                cur_token_pos += 1

            phrase_length = len(phrase_ids[cur_phrase_pos])
            if token_ids[cur_token_pos:cur_token_pos+phrase_length] == phrase_ids[cur_phrase_pos]:
                results.append((cur_token_pos, cur_token_pos+phrase_length-1))
                cur_phrase_pos += 1
                cur_token_pos += phrase_length
            else:
                cur_token_pos += 1
        return results


if __name__ == "__main__":

    data = []

    with open(ARGS.input_path, encoding='utf8') as f:
        for line in tqdm(f, ncols=100, desc='Reading dataset'):
            #line = json.loads(line)
            line = eval(line)
            data.append(line)
            if len(data) >= ARGS.max_dataline:
                break

    processed_data, user_item_dict = Processor.process_data(data)

    print(f'The length of processed data: {len(processed_data)}')

    train_set = set()
    dev_set = set()
    test_set = set()

    for user, item_set in user_item_dict.items():

        item_list = list(item_set)

        if len(item_list) == 1:
            train_set.add((user, item_list[0]))
        elif len(item_list) == 2:
            train_set.add((user, item_list[0]))
            dev_set.add((user, item_list[1]))
        else:
            for item in item_list[:-2]:
                train_set.add((user, item))
            dev_set.add((user, item_list[-2]))
            test_set.add((user, item_list[-1]))

    # used to make sure user and item in dev and test must appear in train
    train_user_set = set([pair[0] for pair in train_set])
    train_item_set = set([pair[1] for pair in train_set])

    train_data = []
    dev_data = []
    test_data = []

    pair_set = set()  # used to check each (user, item) pair appears only once

    for line in processed_data:

        user = line['user_id']
        item = line['item_id']

        if (user, item) in pair_set:
            continue

        pair_set.add((user, item))
        if (user, item) in train_set:
            train_data.append(line)
        elif (user, item) in dev_set:
            if user in train_user_set and item in train_item_set:
                dev_data.append(line)
        else:
            if user in train_user_set and item in train_item_set:
                test_data.append(line)

    print(
        f'Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}')

    # Prepare data for baselines

    output_dir = os.path.join(ARGS.output_dir, 'baseline')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf8') as f:
        for line in tqdm(train_data, ncols=100, desc='Writing training data'):
            new_line = {'user_id': line['user_id'], 'item_id': line['item_id'],
                        'text': line['text'], 'phrases': line['phrases']}
            f.write(json.dumps(new_line)+'\n')

    with open(os.path.join(output_dir, 'dev.json'), 'w', encoding='utf8') as f:
        for line in tqdm(dev_data, ncols=100, desc='Writing dev data'):
            new_line = {'user_id': line['user_id'], 'item_id': line['item_id'],
                        'text': line['text'], 'phrases': line['phrases']}
            f.write(json.dumps(new_line)+'\n')

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf8') as f:
        for line in tqdm(test_data, ncols=100, desc='Writing test data'):
            new_line = {'user_id': line['user_id'], 'item_id': line['item_id'],
                        'text': line['text'], 'phrases': line['phrases']}
            f.write(json.dumps(new_line)+'\n')

    # Construct ref from training data

    ref = []
    ref_user = defaultdict(dict)
    ref_item = defaultdict(dict)

    output_dir = os.path.join(ARGS.output_dir, 'finetune')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for num_lines, data_line in enumerate(train_data):

        ref.append(data_line['tagging_data']['token_ids'])
        ref_user[data_line['user_id']][data_line['item_id']] = num_lines
        ref_item[data_line['item_id']][data_line['user_id']] = num_lines

    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf8') as f:
        for line in tqdm(train_data, ncols=100, desc='Writing training data'):
            f.write(json.dumps(line)+'\n')

    with open(os.path.join(output_dir, 'dev.json'), 'w', encoding='utf8') as f:
        for line in tqdm(dev_data, ncols=100, desc='Writing dev data'):
            f.write(json.dumps(line)+'\n')

    with open(os.path.join(output_dir, 'test.json'), 'w', encoding='utf8') as f:
        for line in tqdm(test_data, ncols=100, desc='Writing test data'):
            f.write(json.dumps(line)+'\n')

    with open(os.path.join(output_dir, 'ref_review.json'), 'w', encoding='utf8') as f:
        json.dump(ref, f)

    with open(os.path.join(output_dir, 'ref_user.json'), 'w', encoding='utf8') as f:
        json.dump(ref_user, f)

    with open(os.path.join(output_dir, 'ref_item.json'), 'w', encoding='utf8') as f:
        json.dump(ref_item, f)
