import json
import os
from dataclasses import dataclass
from typing import List, Tuple
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import multiprocessing as mp
import torch
import unicodedata
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from transformers import RobertaTokenizer


@dataclass
class SentenceProcessor:

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float
    substitute_probability: float
    mask_probability: float
    min_mask: int

    def __call__(self, sentence):
        '''
        input: sentence
        output: {source: "", mid: "",  lm_label: "", ins_label: [0, 0, 0, ..., 6]}
        '''
        if isinstance(sentence, str):
            tokens = [self.tokenizer.bos_token] + \
                self.tokenizer.tokenize(sentence) + [self.tokenizer.eos_token]
            input_ids = [self.tokenizer.convert_tokens_to_ids(
                token) for token in tokens]

            return {"source": input_ids, "mid": input_ids,  "lm_label": [-100]*len(input_ids), "ins_label": [0]*len(input_ids)}
        elif isinstance(sentence, list):
            assert sentence[0] == self.tokenizer.bos_token_id and sentence[-1] == self.tokenizer.eos_token_id and len(
                sentence) > 2
            tokens = [self.tokenizer.convert_ids_to_tokens(
                token) for token in sentence]
            input_ids = sentence

            if len(sentence) <= self.min_mask + 2:  # +2 for <s> and </s>

                mid = [self.tokenizer.bos_token_id] + [self.tokenizer.mask_token_id] * \
                    len(sentence[1:-1]) + [self.tokenizer.eos_token_id]
                lm_label = [-100] + sentence[1:-1] + [-100]
                ins_label = [len(sentence[1:-1]), 0]
                return {"source": [sentence[0], sentence[-1]], "mid": mid,  "lm_label": lm_label, "ins_label": ins_label}

        else:
            assert 0, 'Sentence should be a string or a list of tokens.'

        mask_labels = self._whole_word_mask(tokens)
        inputs, labels = self.mask_tokens(torch.tensor(input_ids).view(
            1, -1), torch.tensor(mask_labels).view(1, -1))

        mid = inputs.tolist()
        lm_label = labels.tolist()

        source = [input_id for input_id in mid if input_id !=
                  self.tokenizer.mask_token_id]
        ins_pos = inputs.eq(self.tokenizer.mask_token_id).long().tolist()

        ins_label = []
        for ele in ins_pos:
            if ele == 0:
                ins_label.append(0)
            else:
                ins_label[-1] += 1

        return {"source": source, "mid": mid,  "lm_label": lm_label, "ins_label": ins_label}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):

        cand_indexes = []

        for (i, token) in enumerate(input_tokens):

            if token == self.tokenizer.bos_token or token == self.tokenizer.eos_token:
                continue

            if self._is_subword(token) and len(cand_indexes) > 0:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(
            1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _is_subword(self, token: str):
        if (
            not self.tokenizer.convert_tokens_to_string(token).startswith(" ")
            and not self._is_punctuation(token[0])
        ):
            return True

        return False

    @staticmethod
    def _is_punctuation(char: str):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, self.mask_probability)).bool() & masked_indices

        while indices_replaced.long().sum().item() < self.min_mask and labels.shape[1] > 2:
            mask_idx = random.randint(1, labels.shape[1]-2)
            labels[:, mask_idx] = inputs[:, mask_idx]
            indices_replaced[:, mask_idx] = True

        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, self.substitute_probability/(
            1-self.mask_probability))).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs.view(-1), labels.view(-1)


parser = ArgumentParser()

parser.add_argument('--input_path', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--max_len', type=int, default=64)
parser.add_argument('--mlm_prob', type=float, default=0.2)
parser.add_argument('--sub_prob', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.8)
parser.add_argument('--min_mask', type=int, default=2)
parser.add_argument('--text_token', type=str, default='text')

ARGS = parser.parse_args()

if not os.path.exists(ARGS.output_dir):
    os.makedirs(ARGS.output_dir)

TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
PROCESSOR = SentenceProcessor(
    TOKENIZER, ARGS.mlm_prob, ARGS.sub_prob, ARGS.mask_prob, ARGS.min_mask)


def _process(line):

    text = line[ARGS.text_token]

    token_ids = TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(text))

    if ARGS.min_len > len(token_ids) or len(token_ids) > ARGS.max_len:
        return None

    results = []
    local_k = 0
    res = PROCESSOR(text)
    res.update({
        'k': local_k}
    )
    results.append(res)
    while len(res['source']) > 2:
        local_k += 1
        res = PROCESSOR(res['source'])
        res.update({
            'k': local_k}
        )
        results.append(res)

    return results


def unwrap_self(cls, kw1, kw2):
    cls.process_wrapper(kw1, kw2)


def unwrap_self_listener(cls):
    cls.listener()


class MultiProcessor(object):
    def __init__(self, fname, cores, epoch_lines, total):
        self.fname = fname
        self.cores = cores
        self.total = total

        filesize = os.path.getsize(fname)
        if filesize > 0:
            if filesize < cores:
                self.size = filesize
                self.cores = 1
            else:
                self.size = int(filesize/cores)
        else:
            self.size = 0

        manager = mp.Manager()
        self.queue = manager.Queue(maxsize=1000000)

        self.epoch_lines = epoch_lines

        self.local_max_len = 0
        self.local_max_ins = 0
        self.local_lines = 0

        self.epoch_num = 0
        self.num_lines = 0

    def process(self, line):
        results = _process(line)
        self.queue.put(results)

    def process_wrapper(self, chunkStart, chunkSize):

        with open(self.fname) as f:
            f.seek(chunkStart)
            lines = f.read(chunkSize).splitlines()
            for line in lines:
                line = json.loads(line)
                self.process(line)

    def chunkify(self):
        if self.size <= 0:
            return
        fileEnd = os.path.getsize(self.fname)
        with open(self.fname, 'rb') as f:
            chunkEnd = f.tell()
            while True:
                chunkStart = chunkEnd
                f.seek(self.size, 1)
                f.readline()
                chunkEnd = f.tell()
                yield chunkStart, chunkEnd - chunkStart
                if chunkEnd > fileEnd:
                    break

    def listener(self):
        """listen other process sending data 
        """
        pbar = tqdm(total=self.total, desc='Processing data', ncols=100)
        fw = open(os.path.join(ARGS.output_dir,
                  f"epoch_{self.epoch_num}.json"), "w")
        while 1:
            results = self.queue.get()
            if results == 'kill':
                break
            pbar.update(1)
            self.num_lines += 1
            if results is not None:
                for line in results:
                    self.local_lines += 1
                    self.local_max_len = max(
                        self.local_max_len, len(line['lm_label']))
                    self.local_max_ins = max(
                        self.local_max_ins, max(line['ins_label']))
                    fw.write(f"{json.dumps(line)}\n")

            if self.num_lines == self.epoch_lines:
                fw.close()
                json.dump({'num_training_examples': self.local_lines, 'max_seq_len': self.local_max_len, 'max_ins_num': self.local_max_ins},
                          open(os.path.join(ARGS.output_dir, f"epoch_{self.epoch_num}_metrics.json"), "w"))
                self.local_lines = 0
                self.local_max_len = 0
                self.local_max_ins = 0
                self.num_lines = 0
                self.epoch_num += 1
                fw = open(os.path.join(ARGS.output_dir,
                          f"epoch_{self.epoch_num}.json"), "w")

        fw.close()
        json.dump({'num_training_examples': self.local_lines, 'max_seq_len': self.local_max_len, 'max_ins_num': self.local_max_ins},
                  open(os.path.join(ARGS.output_dir, f"epoch_{self.epoch_num}_metrics.json"), "w"))

    def run(self):
        # init objects
        pool = mp.Pool(self.cores + 1)
        jobs = []

        # start queue for writing file
        watcher = pool.apply_async(unwrap_self_listener, (self,))

        # create jobs
        for chunkStart, chunkSize in self.chunkify():
            jobs.append(pool.apply_async(
                unwrap_self, (self, chunkStart, chunkSize)))

        # wait for all jobs to finish
        for job in jobs:
            job.get()

        # clean up
        # waiting untill queue push data to file
        self.queue.put('kill')
        watcher.get()

        pool.close()


if __name__ == "__main__":

    data = []
    with open(ARGS.input_path, "r") as f:
        for line in tqdm(f, desc='Reading data'):
            line = json.loads(line)
            data.append(line)

    epoch_lines = len(data) // ARGS.epochs
    total = len(data)
    del data
    p = MultiProcessor(fname=ARGS.input_path, cores=ARGS.num_workers,
                       epoch_lines=epoch_lines, total=total)
    p.run()
