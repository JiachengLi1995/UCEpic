import json
import os
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from transformers import RobertaTokenizer

from processor import SentenceProcessor



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
        
        while 1:
            results = self.queue.get()
            if results == 'kill':
                break

            if not os.path.exists(os.path.join(ARGS.output_dir, f"epoch_{self.epoch_num}.json")):
                fw = open(os.path.join(ARGS.output_dir,
                          f"epoch_{self.epoch_num}.json"), "w")
            
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

        fw.close()
        if self.local_lines > 0:
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
