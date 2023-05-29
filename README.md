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
The pretraining aims to train a non-personalized text generator via our proposed robust insertion process. The pretraining data is the Wikipedia corpus. 

# Pretrained Model
We provide two versions of pretrained models, i.e., ``wiki_ckpt`` and ``review_ckpt``. The backbone of these two models is ``roberta-base``.
The ``wiki_ckpt`` is trained on Wikipedia corpus, while the ``review_ckpt`` is trained on [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and [Google Local](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) review corpus.

|              Model              | Note|
|:-------------------------------|------|
|[wiki_ckpt](https://drive.google.com/file/d/1WcyqBnuk_tV8LvNaDTYrHQw-UKCigqH7/view?usp=share_link)| UCEpic model (without references and aspects) on Wikipedia corpus.|
|[review_ckpt]()| UCEpic model (without references and aspects) on Amazon and Google Local review corpus.|

# Finetuning

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