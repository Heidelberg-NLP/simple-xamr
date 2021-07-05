# Translate, then Parse! A strong baseline for Cross-Lingual AMR Parsing
The goal of our project is to combine a state-of-the-art (sota) NMT system and a sota AMR parser in order to perform multilingual AMR parsing.

For more information, read our paper: [Translate, then Parse! A strong baseline for Cross-Lingual AMR Parsing](https://arxiv.org/abs/2106.04565)

Contributors:

   - Sarah Uhrig suhrig@cl.uni-heidelberg.de
   - Yoalli Rezepka García rezepkagarcia@cl.uni-heidelberg.de

Based on these useful papers and projects:

   - [XL-AMR: Enabling Cross-Lingual AMR Parsing with Transfer Learning Techniques](https://www.aclweb.org/anthology/2020.emnlp-main.195.pdf)
   - [XL-AMR code](https://github.com/SapienzaNLP/xl-amr)
   - [AMR Parser](https://github.com/sheng-z/stog)
   - [Smatch (semantic match) tool](https://github.com/snowblink14/smatch)
   - [amrlib](https://github.com/bjascob/amrlib)
   - [AMR 2.0](https://catalog.ldc.upenn.edu/LDC2017T10)
   - [SOTA NMT System](https://github.com/UKPLab/EasyNMT)
   - [OPUS-MT (Tiedemann&Thottingal, 2020)](https://github.com/Helsinki-NLP/Opus-MT)
   - [Europarl Parallel Corpus](https://www.statmt.org/europarl/)
  
## Setup
Download [amrlib Parse T5 STOG model](https://github.com/bjascob/amrlib-models) and place the extracted files in the correct folder as stated in the description (``amrlib/data/model_stog``).

Linux & Mac users:
```pip install -r requirements.txt```

Windows users:
   * ```pip install -r requirements.txt```
   * ```pip uninstall easynmt```
   * ```pip install --no-deps easynmt```

## Run in Colab
```
!git clone https://github.com/Yrgarcia/Multilingual-AMR-Parsing
import os
os.chdir('/content/Multilingual-AMR-Parsing/')
!wget https://github.com/bjascob/amrlib-models/releases/download/model_parse_t5-v0_1_0/model_parse_t5-v0_1_0.tar.gz
!tar -xvzf model_parse_t5-v0_1_0.tar.gz
!pip3 install -r requirements.txt
```
Set the `AMR_mode_dir` variable to `/content/Multilingual-AMR-Parsing/model_parse_t5-v0_1_0` in  `__main__.py`.

### Translation
Set `translate = True` in `__main__.py`.

### AMR parsing
Set `parseamr = True` in `__main__.py`.

### Evaluation
Set `evaluate = True` in `__main__.py`. Set `unify_files = True` in `__main__.py` **only** if you are evaluating on the freshly parsed files. The files contained in `AMRgraphs` folder in this project are already truncated and unified. 

Run:

```!python __main__.py```

## Datasets and Models
We employ the [LDC2020T07](https://catalog.ldc.upenn.edu/LDC2020T07) dataset, a multilingual dataset which "contains translations of the test split sentences from [LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10), a total of 5,484 sentences or 1,371 sentences per language." Languages: Italian, Spanish, German, Mandarin Chinese.

### Step 1: Machine Translation
With [EasyNMT](https://github.com/UKPLab/EasyNMT), we translate the Italian, Spanish, German, and Mandarin Chinese sentences from LDC2020T07 into English. EasyNMT provides a wrapper for pretrained machine translation models. We use University of Helsinki's [Opus-MT models](https://github.com/Helsinki-NLP/OPUS-MT-train). We evaluate translation quality with the help of BLEU score and the pretrained SentenceBERT model, by computing cosine similarity between translation sentence embeddings and gold sentence embeddings.

|lang|BLEU|σ<sub>BLEU</sub> |cosine sim.|σ<sub>cos</sub> |
|----|----|----|----|----|
|DE  |0.41|0.07|0.93|0.01|
|ES  |0.49|0.10|0.95|0.01|
|IT  |0.46|0.07|0.94|0.01|
|ZH  |0.23|0.07|0.88|0.03|
|mean|0.40|0.10|0.92|0.03|

BLEU and cosine similarity results for the neural machine translations from German, Spanish, Italian, and Mandarin Chinese into English. Scores are averaged across datasets for the four languages (standard dev. between datasets). Mean scores are averages across languages (standard dev. between languages).

### Step 2: AMR Parsing
We apply [amrlib](https://github.com/bjascob/amrlib) to parse AMRs from the English sentences, using the Sentence to Graph (StoG(2)) parsing function.

## Evaluation
We evaluate the resulting AMRs, comparing them to the gold AMR graphs provided in the LDC2017T10 AMR 2.0 dataset. We apply amrlib's evaluation metric API for Smatch, an evaluation metric for semantic feature structures.

## Experiments/Results

With exception of ZH, all our results (Trans+AMR) turned out to be very promising with regards to Smatch score as well as scores for different subtasks.


|Model  |AMREAGER  |  |  |  | XL-AMR trans+|  |  |  | Trans +AMR|  |  |  |
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
| Language       | DE                            | ES                                 | IT                            | ZH   | DE   | ES   | IT   | ZH   | DE            | ES            | IT            | ZH   |
| SMATCH       | 39.1                          | 42.1                               | 43.2                          | 34.6 | 53.0 | 58.0 | 58.1 | 43.1 | **67.6**   | **72.3** | **70.7** | **60.2** |
| Unlabeled    | 45.0                          | 46.6                               | 48.5                          | 41.1 | 57.7 | 63.0 | 63.4 | 48.9 | **71.9** | **76.5** | **75.1** | **65.4** |
| No WSD       | 39.2                          | 42.2                               | 42.5                          | 34.7 | 53.2 | 58.4 | 58.4 | 43.2 | **67.9** | **72.7** | **71.1** | **60.4** |
| Reentrancies | 18.6                          | 27.2                               | 25.7                          | 15.9 | 39.9 | 46.6 | 46.1 | 34.7 | **55.8** | **60.9** | **58.2** | **47.5** |
| Concepts     | 44.9                          | 53.3                               | 52.3                          | 39.9 | 58.0 | 65.9 | 64.7 | 48.0 | **71.4** | **78.1** | **75.6** | **63.3** |
| Named Ent.   | 63.1                          | 65.7                               | 67.7                          | 67.9 | 66.0 | 66.2 | 70.0 | 60.6 | **86.3** | **86.6** | **87.6** | **84.2** |
| Wikification | 49.9                          | 44.5                               | 50.6                          | 46.8 | 60.9 | 63.1 | 67.0 | 54.5 | 0.0           | 0.0           | 0.0           | 0.0  |
| Negation     | 18.6                          | 19.8                               | 22.3                          | 6.8  | 11.7 | 23.4 | 29.2 | 12.8 | **49.0** | **59.5** | **55.7** | **38.5**  |
| SRL          | 29.4                          | 35.9                               | 34.3                          | 27.2 | 47.9 | 55.2 | 54.7 | 41.3 | **61.7** | **68.0** | **65.8** | **54.1** |
