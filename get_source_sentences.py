# -*- coding: utf-8 -*-
"""
Short helper script to extract the English source sentences from the original
AMR 2.0 dataset (to use them as a gold standard for translation evaluation).

@author: s-uhr
"""

import os

path = "abstract_meaning_representation_amr_2.0/data/amrs/split/test/"
files = os.listdir(path)
print(files)

for file in files:
    with open(path + file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        sentences = [line for line in lines if line[:7]=="# ::snt"]
        sentences = [sentence[8:] for sentence in sentences]
        
    with open("amr_2-four_translations/english_source_sentences/" + file[:-4] + "_source.txt", "w", encoding="utf-8") as fw:
        for sentence in sentences:
            fw.write(sentence)