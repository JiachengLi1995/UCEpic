from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime as dt
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report

from transformers import RobertaTokenizer, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from model import UCEpicForPretraining
from utils import random_seed, last_commit_msg, save_dependencies

NUM_PAD = 3

InputFeatures = namedtuple(
    "InputFeatures", "input_ids_s attention_mask_s input_ids_m attention_mask_m ins_labels lm_labels")


def convert_example_to_features(example, tokenizer, max_seq_length, args=None):
    '''
    data_line: {source: "", mid: "",  lm_label: "", ins_label: "[0, 0, 0, ..., 6]", itemid: "", userid: ""}
    e.g.: (Should be token ids)
    source: '<s> Good . </s>'
    ins_label: [0, 1, 0, 0]
    mid: '<s> Good <mask> . </s>'
    lm_label: [-100, -100, 'Morning', -100, -100]
    '''

    source_ids = example["source"]
    ins_label = example["ins_label"]

    ins_label = [-100 if i == 0 and args.zero_ins_mask_prob >
                 random.random() else i for i in example["ins_label"]]

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

    source_array = np.full(max_seq_length, dtype=np.int,
                           fill_value=tokenizer.pad_token_id)
    source_array[:len(source_ids)] = source_ids

    source_mask_array = np.zeros(max_seq_length, dtype=np.bool)
    source_mask_array[:len(source_ids)] = 1

    ins_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-100)
    ins_label_array[:len(ins_label)] = ins_label

    # Mid sequences

    mid_array = np.full(max_seq_length, dtype=np.int,
                        fill_value=tokenizer.pad_token_id)
    mid_array[:len(mid_ids)] = mid_ids

    mid_mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mid_mask_array[:len(mid_ids)] = 1

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-100)
    lm_label_array[:len(lm_label)] = lm_label

    features = InputFeatures(input_ids_s=source_array,
                             attention_mask_s=source_mask_array,
                             ins_labels=ins_label_array,
                             input_ids_m=mid_array,
                             attention_mask_m=mid_mask_array,
                             lm_labels=lm_label_array,
                             )
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, args=None):
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = os.path.join(
            training_path, f"epoch_{self.data_epoch}.json")
        metrics_file = os.path.join(
            training_path, f"epoch_{self.data_epoch}_metrics.json")
        assert os.path.exists(data_file) and os.path.exists(metrics_file)
        metrics = json.load(open(metrics_file))
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids_s = np.memmap(filename=self.working_dir/'input_ids_s.memmap',
                                    mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            attention_mask_s = np.memmap(filename=self.working_dir/'attention_mask_s.memmap',
                                         shape=(num_samples, seq_len), mode='w+', dtype=np.bool)

            input_ids_m = np.memmap(filename=self.working_dir/'input_ids_m.memmap',
                                    mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            attention_mask_m = np.memmap(filename=self.working_dir/'attention_mask_m.memmap',
                                         shape=(num_samples, seq_len), mode='w+', dtype=np.bool)

            ins_labels = np.memmap(filename=self.working_dir/'ins_labels.memmap',
                                   shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_labels = np.memmap(filename=self.working_dir/'lm_labels.memmap',
                                  shape=(num_samples, seq_len), mode='w+', dtype=np.int32)

        else:
            input_ids_s = np.zeros(
                shape=(num_samples, seq_len), dtype=np.int32)
            attention_mask_s = np.zeros(
                shape=(num_samples, seq_len), dtype=np.bool)
            input_ids_m = np.zeros(
                shape=(num_samples, seq_len), dtype=np.int32)
            attention_mask_m = np.zeros(
                shape=(num_samples, seq_len), dtype=np.bool)
            ins_labels = np.full(shape=(num_samples, seq_len),
                                 dtype=np.int32, fill_value=-100)
            lm_labels = np.full(shape=(num_samples, seq_len),
                                dtype=np.int32, fill_value=-100)

        logging.info(f"Loading training examples for epoch {epoch}")
        with open(data_file) as f:
            for i, line in enumerate(tqdm(f, ncols=100)):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(
                    example, tokenizer, seq_len, args=args)
                input_ids_s[i] = features.input_ids_s
                attention_mask_s[i] = features.attention_mask_s
                input_ids_m[i] = features.input_ids_m
                attention_mask_m[i] = features.attention_mask_m
                ins_labels[i] = features.ins_labels
                lm_labels[i] = features.lm_labels

        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids_s = input_ids_s
        self.attention_mask_s = attention_mask_s
        self.input_ids_m = input_ids_m
        self.attention_mask_m = attention_mask_m
        self.ins_labels = ins_labels
        self.lm_labels = lm_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids_s[item].astype(np.int64)),
                torch.tensor(self.attention_mask_s[item].astype(np.int64)),
                torch.tensor(self.ins_labels[item].astype(np.int64)),
                torch.tensor(self.input_ids_m[item].astype(np.int64)),
                torch.tensor(self.attention_mask_m[item].astype(np.int64)),
                torch.tensor(self.lm_labels[item].astype(np.int64)),
                )


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--last_checkpoint',
                        type=str,
                        default=None,
                        help="Restor training from the last checkpoint.")
    parser.add_argument('--zero_ins_mask_prob',
                        type=float,
                        default=0.9,
                        help="zero insertion masking probability")
    parser.add_argument('--from_scratch', action='store_true',
                        help='do not load prtrain model, only random initialize')
    parser.add_argument("--output_step", type=int,
                        default=1000, help="Number of step to save model")

    args = parser.parse_args()

    samples_per_epoch = []
    num_data_epochs = args.epochs
    max_ins = 0
    for i in range(args.epochs):
        epoch_file = os.path.join(args.pregenerated_data, f"epoch_{i}.json")
        metrics_file = os.path.join(
            args.pregenerated_data, f"epoch_{i}_metrics.json")
        if os.path.exists(epoch_file) and os.path.exists(metrics_file):
            metrics = json.load(open(metrics_file))
            samples_per_epoch.append(metrics['num_training_examples'])
            max_ins = max(max_ins, metrics['max_ins_num']+1)
        else:
            if i == 0:
                exit("No training data was found!")
            print(
                f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break

    # logging folder
    branch, commit = last_commit_msg()
    args.output_dir = os.path.join('checkpoints', branch, commit, args.output_dir,
                                   f'seed_{args.seed}_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "args.log"), "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    save_dependencies(args.output_dir)

    logging.basicConfig(filename=os.path.join(args.output_dir, "train_log.txt"),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [
                            -1, 0] else logging.WARN
                        )
    logger = logging.getLogger(__name__)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    random_seed(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    tokenizer = RobertaTokenizer.from_pretrained(
        args.bert_model, use_fast=False)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model

    if args.from_scratch:
        config = RobertaConfig()
        model = UCEpicForPretraining()
    else:
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.num_labels = max_ins
        model = UCEpicForPretraining.from_pretrained(args.bert_model, config=config)

    model.to(device)

    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)

    if args.last_checkpoint is not None:
        logging.info(f'Restore checkpoint from {args.last_checkpoint}.')
        model_checkpoint = torch.load(os.path.join(
            args.last_checkpoint, 'checkpoint.bin'), map_location='cuda:{}'.format(args.local_rank))

        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        max_ins = model_checkpoint['max_ins']
        args = model_checkpoint['training_args']

    global_step = 0
    eval_dataset = None
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in tqdm(range(args.epochs-1), ncols=100):  # last epoch is used for dev
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, args=args)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(
            epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        metrics = {
            "lm_correct": 0,
            "lm_total": 1e-10,
            "ins_pred": [],
            "ins_true": [],
            "lm_loss": [],
            "ins_loss": []
        }
        for step, batch in enumerate(tqdm(train_dataloader, ncols=100, desc='Training.')):
            batch = tuple(t.to(device) for t in batch)

            input_ids_s, attention_mask_s, ins_labels, input_ids_m, attention_mask_m, lm_labels = batch

            inputs = {'input_ids_s': input_ids_s, 'attention_mask_s': attention_mask_s, 'ins_labels': ins_labels,
                      'input_ids_m': input_ids_m, 'attention_mask_m': attention_mask_m, 'lm_labels': lm_labels}

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            loss = outputs.loss

            metrics["lm_correct"] += outputs.lm_correct.sum().cpu().item()
            metrics["lm_total"] += outputs.lm_total.sum().cpu().item()
            metrics["ins_pred"] += outputs.ins_pred.cpu().tolist()
            metrics["ins_true"] += outputs.ins_true.cpu().tolist()
            metrics["lm_loss"].append(
                outputs.masked_lm_loss.mean().detach().cpu().item())
            metrics["ins_loss"].append(outputs.ins_loss.mean().detach().cpu().item())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids_s.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.fp16:

                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                    optimizer.zero_grad()

                    if optimizer_was_run:
                        scheduler.step()

                    global_step += 1

                else:

                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if global_step % args.output_step == 0 and args.local_rank in [-1, 0]:
                lm_acc = metrics["lm_correct"] / metrics["lm_total"]
                lm_loss = sum(metrics["lm_loss"]) / len(metrics["lm_loss"])
                ins_loss = sum(metrics["ins_loss"]) / len(metrics["ins_loss"])
                ins_metrics = classification_report(
                    metrics["ins_true"], metrics["ins_pred"])

                logger.info(
                    f"Training: LM Loss:{lm_loss}, INS Loss: {ins_loss}, LM Accuracy: {lm_acc}, Insertion Metrics: {ins_metrics}.")

        if eval_dataset is None:
            eval_dataset = PregeneratedDataset(epoch=args.epochs-1, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                               num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, args=args)

        eval(eval_dataset, model, args, device, logger)

        if args.local_rank in [-1, 0]:
            # Save model checkpoint

            output_dir = os.path.join(
                args.output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, 'module') else model
            # model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_args': args,
                'max_ins': max_ins
            }, os.path.join(output_dir, 'checkpoint.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        logger.info("PROGRESS: {}%".format(
            round(100 * (epoch + 1) / args.epochs, 4)))
        logger.info("EVALERR: {}%".format(tr_loss))

    # Save a trained model
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_args': args,
            'max_ins': max_ins
        }, os.path.join(args.output_dir, 'checkpoint.bin'))


def eval(eval_dataset, model, args, device, logger):

    model.eval()

    if args.local_rank == -1:
        train_sampler = RandomSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(
        eval_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    metrics = {
        "lm_correct": 0,
        "lm_total": 1e-10,
        "ins_pred": [],
        "ins_true": [],
    }

    with torch.no_grad():
        tqdm_loader = tqdm(train_dataloader, ncols=100)
        for step, batch in enumerate(tqdm_loader):
            batch = tuple(t.to(device) for t in batch)

            input_ids_s, attention_mask_s, ins_labels, input_ids_m, attention_mask_m, lm_labels = batch

            inputs = {'input_ids_s': input_ids_s, 'attention_mask_s': attention_mask_s, 'ins_labels': ins_labels,
                      'input_ids_m': input_ids_m, 'attention_mask_m': attention_mask_m, 'lm_labels': lm_labels}

            outputs = model(**inputs)

            metrics["lm_correct"] += outputs.lm_correct.sum().cpu().item()
            metrics["lm_total"] += outputs.lm_total.sum().cpu().item()
            metrics["ins_pred"] += outputs.ins_pred.cpu().tolist()
            metrics["ins_true"] += outputs.ins_true.cpu().tolist()

            tqdm_loader.set_description(
                f"Evaluation: lm_acc: {metrics['lm_correct'] / metrics['lm_total'] : .4f}")

    lm_acc = metrics["lm_correct"] / metrics["lm_total"]
    ins_metrics = classification_report(
        metrics["ins_true"], metrics["ins_pred"])
    logger.info(
        f"Evaluation: LM Accuracy: {lm_acc}, Insertion Metrics: {ins_metrics}.")


if __name__ == '__main__':
    main()
