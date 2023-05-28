from argparse import ArgumentParser
import os
import torch
import logging
import json

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime as dt

from torch.cuda.amp import autocast
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from model import UCEpic
from dataset import FinetuningDataset
from utils import random_seed, last_commit_msg, save_dependencies

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--bert_model", type=str, default='roberta-base', help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--keywords", type=str, default='end2end', help="Method to keywords.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--max_len", type=int, default=256, help="Max length of reference.")
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
    parser.add_argument('--zero_tag_mask_prob',
                        type=float,
                        default=0.9,
                        help="zero tagging masking probability")
    parser.add_argument('--from_scratch', action='store_true', help='do not load prtrain model, only random initialize')
    parser.add_argument("--output_step", type=int, default=1000, help="Number of step to save model")
    parser.add_argument("--num_aspects", type=int, default=100, help="Number of aspects")

    args = parser.parse_args()

    # logging folder
    branch, commit = last_commit_msg()
    args.output_dir = os.path.join('checkpoints', branch, commit, args.output_dir, f'seed_{args.seed}_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "args.log"), "w") as f:
        f.write(json.dumps(vars(args), indent=2)) 
    save_dependencies(args.output_dir)
    
    ## Prepare args
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))


    ## Prepare logger
    logging.basicConfig(filename = os.path.join(args.output_dir, "train_log.txt"),
                        filemode= 'a',
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN
                        )
    logger = logging.getLogger(__name__)


    ## Prepare device    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    # Prepare model
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, use_fast=False)
    if args.local_rank == -1:
        model_checkpoint = torch.load(os.path.join(args.pretrained_model, 'checkpoint.bin'), map_location=device)
    else:
        model_checkpoint = torch.load(os.path.join(args.pretrained_model, 'checkpoint.bin'), map_location='cuda:{}'.format(args.local_rank))

    args.max_ins = model_checkpoint['max_ins']
    config = RobertaConfig.from_pretrained(args.bert_model)
    config.num_labels = args.max_ins
    config.num_aspects = args.num_aspects
    model = UCEpic.from_pretrained(args.bert_model, config=config)

    if not args.from_scratch:
        model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
        logger.warning('Load pre-trained checkpoint!')
    else:
        logger.warning('Training from scratch!')

    model.to(device)

    if args.n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    ## Prepare dataset
    train_dataset = FinetuningDataset(args.path, mode='train', tokenizer=tokenizer, args=args)
    eval_dataset = FinetuningDataset(args.path, mode='dev', tokenizer=tokenizer, args=args)

    total_train_examples = len(train_dataset.data)

    num_train_optimization_steps = int(total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps =num_train_optimization_steps)

    global_step = 0
    local_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in tqdm(range(args.epochs), ncols=100): # last epoch is used for dev
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        metrics = {
            "lm_correct":0,
            "lm_total":1e-10,
            "ins_pred":[],
            "ins_true":[],
            "lm_loss":[],
            "ins_loss":[]
        }

        for step, batch in enumerate(tqdm(train_dataloader, ncols=100, desc='Training.')):
            batch = tuple(t.to(device) for t in batch)

            ref, ref_mask, aspects, aspects_mask, ins_decoder_input_ids_s, ins_decoder_attention_mask_s, ins_decoder_input_ids_m, \
            ins_decoder_attention_mask_m, ins_ins_labels, ins_lm_labels = batch

            inputs = {'input_ids_s':ins_decoder_input_ids_s, 'attention_mask_s': ins_decoder_attention_mask_s, 'ins_labels': ins_ins_labels,\
                      'input_ids_m': ins_decoder_input_ids_m, 'attention_mask_m': ins_decoder_attention_mask_m, 'lm_labels': ins_lm_labels,\
                      'ref': ref, 'ref_mask': ref_mask, 'aspects': aspects, 'aspects_mask': aspects_mask}

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            
            loss = outputs.loss

            metrics["lm_correct"] += outputs.lm_correct
            metrics["lm_total"] += outputs.lm_total
            metrics["ins_pred"] += outputs.ins_pred
            metrics["ins_true"] += outputs.ins_true
            metrics["lm_loss"].append(outputs.masked_lm_loss.detach().cpu().item())
            metrics["ins_loss"].append(outputs.ins_loss.detach().cpu().item())

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += ins_decoder_input_ids_s.size(0)
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

            if local_step % args.output_step == 0 and args.local_rank in [-1, 0]:
                
                lm_acc = metrics["lm_correct"] / metrics["lm_total"]
                lm_loss = sum(metrics["lm_loss"]) / len(metrics["lm_loss"])
                ins_loss = sum(metrics["ins_loss"]) / len(metrics["ins_loss"])
                ins_metrics = classification_report(metrics["ins_true"], metrics["ins_pred"])

                
                logger.info(f"Training: LM Loss:{lm_loss}, INS Loss: {ins_loss}, LM Accuracy: {lm_acc}, \n \
                                Insertion Metrics: {ins_metrics}.")

            local_step += 1

        eval(eval_dataset, model, args, device, logger)

        if args.local_rank in [-1, 0]:
            # Save model checkpoint

            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            tokenizer.save_pretrained(output_dir)

            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'training_args': args,
                'max_ins': args.max_ins
            }, os.path.join(output_dir, 'checkpoint.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        logger.info("PROGRESS: {}%".format(round(100 * (epoch + 1) / args.epochs, 4)))
        logger.info("EVALERR: {}%".format(tr_loss))


    # Save a trained model
    if  args.local_rank == -1 or torch.distributed.get_rank() == 0 :
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        tokenizer.save_pretrained(args.output_dir)

        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'training_args': args,
            'max_ins': args.max_ins
        }, os.path.join(args.output_dir, 'checkpoint.bin'))


def eval(eval_dataset, model, args, device, logger):

    model.eval()

    if args.local_rank == -1:
        train_sampler = RandomSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(eval_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    metrics = {
            "lm_correct":0,
            "lm_total":1e-10,
            "ins_pred":[],
            "ins_true":[],
            "lm_loss":[],
            "ins_loss":[]
        }

    with torch.no_grad():
        tqdm_loader = tqdm(train_dataloader, ncols=100)
        for step, batch in enumerate(tqdm_loader):
            batch = tuple(t.to(device) for t in batch)

            ref, ref_mask, aspects, aspects_mask, ins_decoder_input_ids_s, ins_decoder_attention_mask_s, ins_decoder_input_ids_m, \
            ins_decoder_attention_mask_m, ins_ins_labels, ins_lm_labels = batch

            inputs = {'input_ids_s':ins_decoder_input_ids_s, 'attention_mask_s': ins_decoder_attention_mask_s, 'ins_labels': ins_ins_labels,\
                      'input_ids_m': ins_decoder_input_ids_m, 'attention_mask_m': ins_decoder_attention_mask_m, 'lm_labels': ins_lm_labels,\
                      'ref': ref, 'ref_mask': ref_mask, 'aspects': aspects, 'aspects_mask': aspects_mask}

            outputs = model(**inputs)
            
            metrics["lm_correct"] += outputs.lm_correct
            metrics["lm_total"] += outputs.lm_total
            metrics["ins_pred"] += outputs.ins_pred
            metrics["ins_true"] += outputs.ins_true

            tqdm_loader.set_description(f"Evaluation: lm_acc: {metrics['lm_correct'] / metrics['lm_total'] : .4f}")
    
    lm_acc = metrics["lm_correct"] / metrics["lm_total"]
    ins_metrics = classification_report(metrics["ins_true"], metrics["ins_pred"])

    logger.info(f"Evaluation: LM Accuracy: {lm_acc}, \n \
                            Insertion Metrics: {ins_metrics}.")

    model.train()

if __name__ == '__main__':
    main()
