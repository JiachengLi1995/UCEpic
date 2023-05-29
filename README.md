# UCEpic

This repository contains the code of model UCEpic for <strong> Lexically Constrained </strong> Explanation Generation in Recommendation.

Our KDD 2023 paper [UCEpic: Unifying Aspect Planning and Lexical Constraints for Generating Explanations in Recommendation](https://arxiv.org/abs/2209.13885).

# Quick Links

  - [Overview](#overview)
  - [Dependencies](#dependencies)
  - [Pretraining](#pretraining)
  - [Pretrained Model](#pretrained-model)
  - [Finetuning](#finetuning)
  - [Contact](#contact)
  - [Citation](#citation)

# Overview
We propose a model, UCEpic, that generates high-quality personalized explanations for recommendation results by unifying aspect planning and lexical constraints in
an insertion-based generation manner. Methodologically, to ensure text generation quality and robustness to various lexical constraints, we first pre-train a non-personalized text generator via our proposed robust insertion process. Then, to
obtain personalized explanations under this framework of insertion-based generation, we design a method of incorporating aspect planning and personalized references into the insertion process. Hence, UCEpic unifies aspect planning and lexical constraints into one framework and generates explanations for recommendations under different settings.

# Dependencies
We train and test the model using the following main dependencies:
- Python 3.10.10
- PyTorch 2.0.0
- Transformers 4.28.0
- Spacy 3.5.1

# Pretraining
## Dataset
The pretraining aims to train a non-personalized text generator via our proposed robust insertion process. The pretraining data is the Wikipedia corpus. We provide a sample data in ``data/wiki_sample.json`` and the full dataset can be downloaded from [here](https://drive.google.com/file/d/1fSnPJu_ewcx5gv7f26YDdI6RgWxwgLaI/view?usp=share_link).

## Training
To pretrain the model, we need to first pre-process dataset. The pre-processing will tokenize text and then randomly sample tokens in multiple rounds. The sampled tokens will be used as the insertion positions and inserted tokens. The pre-processing can be done by running the following command:
```bash
bash scripts/pretrain_data_process.sh
```
The data will be splitted into ``--epochs`` files and each file will be used for one epoch. The last epoch is used for validation.

Then, we can start the pretraining by running the following command:
```bash
bash scripts/pretrain.sh
```
In ``scripts/pretrain.sh``, we provide two sample scripts for single-GPU training and multi-GPU training respectively.

## Inference
The pretrained model is a non-personalized text generator. It can be used for general text generation given keywords or phrases. We provide an example in ``scripts/pretrain_inference.sh``. Run the script and input ``print(generator(source=["UCSD student", "machine learning"]))`` in the python console. You will get the following output:
```
(Pdb) print(generator(source=["UCSD student", "machine learning"]))
TURN [1]: <s> a UCSD student in machine learning.</s>
TURN [2]: <s>He is a UCSD student specializing in machine learning and.</s>
TURN [3]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [4]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [5]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [6]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [7]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [8]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [9]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
TURN [10]: <s>He is a UCSD student, specializing in machine learning and statistics.</s>
<s>He is a UCSD student, specializing in machine learning and statistics.</s>
```
The output shows all intermediate results during insertion. From the output, we can see that the model can generate text with the given keywords or phrases. The generated text can guarantee to include all keywords and prhases.


# Pretrained Model
We provide two versions of pretrained models, i.e., ``wiki_ckpt`` and ``review_ckpt``. The backbone of these two models is ``roberta-base``.
The ``wiki_ckpt`` is trained on Wikipedia corpus, while the ``review_ckpt`` is trained on [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and [Google Local](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) review corpus.

|              Model              | Note|
|:-------------------------------|------|
|[wiki_ckpt](https://drive.google.com/file/d/1WcyqBnuk_tV8LvNaDTYrHQw-UKCigqH7/view?usp=share_link)| UCEpic model (without references and aspects) on Wikipedia corpus.|
|[review_ckpt](https://drive.google.com/file/d/1SFCHho7hJAV2m_kUxYEdT5A_Ge27JqdP/view?usp=share_link)| UCEpic model (without references and aspects) on Amazon and Google Local review corpus.|

# Finetuning
In the finetuning stage, we aim to finetune the pretrained model to generate personalized explanations for recommendation results. This process will train the model to incorporate aspect planning and personalized references into the insertion process. 

## Dataset
In this repor, we provide a sample Yelp data for finetuning in ``data/yelp/data.json``. The data pre-processing for finetuning have two steps:
- Sample insertion positions and tokens. The sampled tokens will be used as the insertion positions and inserted tokens.
- Construct references and extract keyphrases using the [UCTopic](https://github.com/JiachengLi1995/UCTopic).

We provide a sample script to process ``data/yelp/data.json`` in ``scripts/finetune_data_process.sh``. Run the script by command:
```bash
bash scripts/finetune_data_process.sh
```
Then, construct the references and extract keyphrases using ``UCTopic`` using the following command:
```bash
bash scripts/phrase_aspect_prepare.sh
```
For the ``UCTopic`` installation, please refer to [UCTopic](https://github.com/JiachengLi1995/UCTopic) github repo.

## Training
After the data pre-processing, we can start the finetuning by running the following command:
```bash
bash scripts/finetune.sh yelp/finetune
```

## Inference and Evaluation
After the finetuning, we can use the model to generate explanations for recommendation results. We provide a sample script in ``scripts/evaluation.sh``. Run the script by command:
```bash
bash scripts/evaluation.sh
```
You need specify the following arguments in the script:
- ``DATA``: the path of the pre-processed data from ``finetune_data_process.sh`` and ``phrase_aspect_prepare.sh``.
- ``CKPT``: the path of the finetuned model (folder).
- ``EXT``: the key of given keywords or phrases in the JSON file.
- ``NUM``: the number of keywords or phrases used for generation (set ``-1`` to use all).
- ``CONTROL``: the control mode for generation. ``hard`` means the model will generate text with the given keywords or phrases. ``soft`` means the model will generate text with the given aspect (aspect-planning).

In the same script, we also provide the evaluation code (``compute_scores.py``) for the generated explanations. The evaluation code will compute the all metrics used in the paper for the generated explanations. 

# Contact
If you have any questions related to the code or the paper, feel free to email Jiacheng (`j9li@eng.ucsd.edu`).

# Citation

Please cite our paper if you use UCEpic in your work:

```bibtex
@inproceedings{li23unifying,
  title = "UCEpic: Unifying Aspect Planning and Lexical Constraints for Generating Explanations in Recommendation",
  author = "Jiacheng Li and Zhankui He and Jingbo Shang and Julian McAuley",
  year = "2023",
  booktitle = "KDD"
}
```