from typing import List, Tuple
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import torch
import unicodedata
import random



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
            not self.tokenizer.convert_tokens_to_string([token]).startswith(" ")
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