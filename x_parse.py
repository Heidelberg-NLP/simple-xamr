# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 09:54:36 2021

@author: s-uhr
"""

from amr_parser import sent_to_graph, graph_to_sent, read_file, save_graphs, evaluate_smatch
from nmt_english import Translator
import multiprocessing
import os
import locale
import sys

def truncate_files(filename):
    # cut off the last 5 lines to omit the obsolete AMR graph
    with open(filename, mode='r+', encoding=preferred_encoding) as f:
        lines = f.readlines()[:-5]
        print(lines)
        f.truncate()
        f.seek(0)
        f.writelines(lines)

if __name__ == '__main__':
    arguments = sys.argv
    
    source_language = arguments[2]
    file_to_translate = arguments[4]
    
    gold_amrs = "amr_2-four_translations/AMR/" + file_to_translate[29:-17] + ".txt"
    
    if not os.path.exists('translations'):
        os.makedirs('translations')
        
    if not os.path.exists('AMRgraphs'):
        os.makedirs('AMRgraphs')
    
    amr_gsii = False
    
    print("\nParsing file", file_to_translate, "from", source_language + ".\n")
    
    multiprocessing.freeze_support()
    preferred_encoding = locale.getpreferredencoding()
    preferred_encoding = 'utf-8'
    
    english_source_sentences = os.listdir("amr_2-four_translations/english_source_sentences")  # where the original english sentence files are stored
    translations = sorted(os.listdir("translations"))  # where the translated files are stored
    amr_graphs = sorted(os.listdir("AMRgraphs"))  # where to store the parsed AMR graphs

    categories = ["bolt", "consensus", "dfa", "proxy", "xinhua"]  # categories included in the dataset for each language
    languages = ["DE", "ES", "IT", "ZH"]  # languages included in the dataset
    
    # Translate file and save it to translations folder
    translator = Translator()
    translator.load_sentences(file_to_translate)
    translator.translate(source_language=source_language)
    translator.save_translation("translations/" + file_to_translate[29:-4] + "_nmt.txt")
    translation_file = "translations/" + file_to_translate[29:-4] + "_nmt.txt"

    # parse translated file to AMR graphs save it to AMRgraphs folder
    sentences = read_file(translation_file)
    graphs = sent_to_graph(sentences, device="cpu") #TODO REMOVE CPU
    new_path = "AMRgraphs/" + translation_file[13:-8] + "_AMR.txt"
    save_graphs(graphs, path=new_path)
    truncate_files(new_path) # TODO apparently not working
    
    evaluate_smatch(gold_amrs, new_path)
    