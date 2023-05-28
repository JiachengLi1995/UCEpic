import datasets
import json
import spacy
from tqdm import tqdm
import sys
from nltk.translate.bleu_score import corpus_bleu

FILE_PATH = 'data/beer/generated_gt.json' if len(sys.argv) <= 1 else sys.argv[1]
NLP = spacy.load('en_core_web_sm', disable=['ner'])

def n_grams(tokens, N):
    if len(tokens) < N:
        return []
    grams = []
    for i in range(len(tokens)-N+1):
        grams.append(' '.join(tokens[i:i+N]))
    return grams

def mean(A):
    return sum(A)/len(A)

class Distinct:
    def __init__(self):
        self.sentences = []
    
    def add(self, sentence):
        '''
        sentence: a list of tokens
        '''
        self.sentences.append(sentence)

    def compute(self, type='inter'):

        if type == 'inter':
            total = 0
            distinct_set = set()

            for sentence in self.sentences:
                distinct_set |= set(sentence)
                total += len(sentence)
            return len(distinct_set) / total
        
        else:
        
            return sum(self.distinct_n(sentence) for sentence in self.sentences) / len(self.sentences)

    def distinct_n(self, sentence):
        '''
        sentence: a list of tokens
        '''
        if len(sentence) == 0:
            return 0.0
        distinct_ngrams = set(sentence)
        return len(distinct_ngrams) / len(sentence)

bleu_1 = datasets.load_metric('bleu')
bleu_2 = datasets.load_metric('bleu')
bleu_3 = datasets.load_metric('bleu')
bleu_4 = datasets.load_metric('bleu')
distinct_1 = Distinct()
distinct_2 = Distinct()
distinct_1_gt = Distinct()
distinct_2_gt = Distinct()
meteor = datasets.load_metric('meteor')
rouge = datasets.load_metric('rouge')
bertscore = datasets.load_metric('bertscore')

data = []
num_lines = sum(1 for _ in open(FILE_PATH))
results = []
gens = []
gts = []

with open(FILE_PATH, encoding='utf8') as f:
    for line in tqdm(f, ncols=100, total=num_lines):

        line = json.loads(line)

        gt = line['gt']
        gen = line['gen']

        # 1-gram
        gt_1 = [token.text.lower() for token in NLP(gt) if not token.is_punct]
        gen_1 = [token.text.lower() for token in NLP(gen) if not token.is_punct]

        # 2-gram
        gt_2 = n_grams(gt_1, 2)
        gen_2 = n_grams(gen_1, 2)

        # 3-gram
        gt_3 = n_grams(gt_1, 3)
        gen_3 = n_grams(gen_1, 3)

        # 4-gram
        gt_4 = n_grams(gt_1, 4)
        gen_4 = n_grams(gen_1, 4)
        
        # bleu
        gens.append(gen_1)
        gts.append([gt_1])

        # distinct
        distinct_1.add(gen_1)
        distinct_1_gt.add(gt_1)

        distinct_2.add(gen_2)
        distinct_2_gt.add(gt_2)

        gen_sentence = ' '.join(gen_1)
        gt_sentence = ' '.join(gt_1)
        # meteor
        meteor.add(prediction=gen_sentence, reference=gt_sentence)
        # rouge
        rouge.add(prediction=gen_sentence, reference=gt_sentence)
        data.append({'hpy': gen_sentence, 'ref': gt_sentence})
        # bertscore
        bertscore.add(prediction=gen_sentence, reference=gt_sentence)

print('Computing BLEU scores:')

print('BLEU-1: ', corpus_bleu(gts, gens, weights=(1, 0, 0, 0)))
print('BLEU-2: ', corpus_bleu(gts, gens, weights=(0, 1, 0, 0)))
print('BLEU-3: ', corpus_bleu(gts, gens, weights=(0, 0, 1, 0)))
print('BLEU-4: ', corpus_bleu(gts, gens, weights=(0, 0, 0, 1)))

print('Computing Distinct scores:')
print('DISTINCT-1:\t', 'Prediction:\t', distinct_1.compute(), '\tGT:\t', distinct_1_gt.compute())
print('DISTINCT-2:\t', 'Prediction:\t', distinct_2.compute(), '\tGT:\t', distinct_2_gt.compute())

print('Computing Meteor:')
print('Meteor: ', meteor.compute()['meteor'])

print('Computing ROUGE:')
print('ROUGE-L: ', rouge.compute()['rougeL'].mid.fmeasure)

print('BERT scores:')
print('BERT score (roberta):', mean(bertscore.compute(lang='en')['f1']))