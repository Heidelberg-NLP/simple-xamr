# -*- coding: utf-8 -*-
"""
Translator class to load a pretrained machine translation model, apply it to
a list of sentences, save the translated sentences and compare them to a list
of gold standard English sentences with BLEU and the cosine similarity of 
Sentence-BERT Embeddings.

@author: s-uhr
"""

from easynmt import EasyNMT
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import os

class Translator(object):

    def __init__(self, model_name='opus-mt', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = EasyNMT(self.model_name, device=self.device)
        # self.sentences_to_translate
        # self.gold_sentences
        # self.translation
        # self.sentence_embeddings_gold
        # self.sentence_embeddings_translation
    
    def load_sentences(self, to_translate_path, gold_path="None"):
        """
        Load sentences to translate and English gold sentenecs and store them
        as class variables.

        Parameters
        ----------
        to_translate_path : String
            Absolute or relative path to file with sentences to translate.
        gold_path : String
            Absolute or relative path to file with English gold sentences (for
            evaluation).

        Returns
        -------
        list
            List of sentences to translate.

        """
        self.to_translate_path = to_translate_path
        with open(self.to_translate_path, mode="r", encoding="utf-8") as fr:
            self.sentences_to_translate = fr.read().split("\n")
        
        print("Sentences to translate loaded from", self.to_translate_path)
        
        if gold_path != "None":
            with open(gold_path, mode="r", encoding='utf-8') as fr:
                self.gold_sentences = fr.read().split("\n")
                
            print("Gold sentences loaded from", gold_path)
        
        # print(self.sentences_to_translate)
        # print(self.gold_sentences)
        
        return self.sentences_to_translate

        

    def translate(self, source_language, target_language='en'):
        """
        Translate sentences_to_translate to the target language and store
        them in class variable "translation".

        Parameters
        ----------
        target_language : String, optional
            Iso-Code of the language to translate to. The default is 'en'.

        Returns
        -------
        list
            List of translated sentences.

        """
        print("... translating to target language:", target_language)
        self.translation = self.model.translate(self.sentences_to_translate,
                                                target_lang=target_language,
                                                source_lang=source_language)
        
        #print(self.translation)
        return self.translation
    
    
    def save_translation(self, path):
        """
        Save translation to path, one sentence per line.

        Parameters
        ----------
        path : String
            Absolute or relative file path to save translation to.

        Returns
        -------
        None.

        """
        with open(path, mode="w", encoding='utf-8') as fw:
            for sentence in self.translation:
                fw.write(sentence)
                fw.write("\n")
        
        print("Translations saved to", path)
        
        return None
    
    
    def evaluate_bleu(self):
        """
        Computes the pairwise bleu score for each of the English gold sentences
        and the translated English sentences and returns the mean bleu score.

        Returns
        -------
        bleu : float
            Mean of the sentences' bleu scores.

        """
        bleu_scores = []
        
        for original, translation in zip(self.gold_sentences, self.translation):
            
            bleu = sentence_bleu([original.split()], translation.split(), weights=(0.5, 0.5))
            bleu_scores.append(bleu)
            
        bleu = np.mean(bleu_scores)
        standard_dev = np.std(bleu_scores)
        
        bleu_eval = "Bleu Score (mean of all sentences): "+"{:2.4f}".format(bleu)+"; σ = "+"{:2.4f}".format(standard_dev)
        print(bleu_eval)
        
        with open("translation_evaluation.txt", "a", encoding='utf-8') as fa:
            fa.write("\n\n## " + self.to_translate_path)
            fa.write("\n" + bleu_eval + "\n")
        
        return bleu
    
    
    def create_sentence_embeddings(self):
        """
        Creates Sentence-BERT embeddings for each of the English gold sentences
        and the translated sentences (helper function for cosine similarity
        evaluation).

        Returns
        -------
        self.sentence_embeddings_gold
            Sentence embeddings of the English gold sentences.
        self.sentence_embeddings_translation
            Sentence embeddings of the English translations.

        """
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        print("... creating sentence embeddings")
        
        self.sentence_embeddings_gold = sbert_model.encode(self.gold_sentences,
                                                           show_progress_bar=False,
                                                           device=self.device)
        self.sentence_embeddings_translation = sbert_model.encode(self.translation,
                                                                  show_progress_bar=False,
                                                                  device=self.device)
        
        return self.sentence_embeddings_gold, self.sentence_embeddings_translation
    
    
    def evaluate_cosine_similarity(self):
        """
        Computes pairwise cosine similarity between gold and translation sentence
        embeddings and returns the mean of all values.

        Returns
        -------
        cosine_mean : float
            The mean of all pairwise cosine similarities.

        """
        self.create_sentence_embeddings()
        
        cosine_scores = []
        i=1
        
        for original, translation in zip(self.sentence_embeddings_gold, self.sentence_embeddings_translation):
            sim = 1 - distance.cosine(original, translation)
            #print(i, "similarity = ", sim)
            i+=1
            cosine_scores.append(sim)
        
        cosine_mean = np.mean(cosine_scores)
        standard_dev = np.std(cosine_scores)
        
        cosine_eval = "Cosine similarity gold––translation (mean of all sentences): " + "{:2.4f}".format(cosine_mean) + "; σ = " + "{:2.4f}".format(standard_dev)
        print(cosine_eval, "\n")
        
        with open("translation_evaluation.txt", "a", encoding='utf-8') as fa:
            fa.write(cosine_eval)
            
        return cosine_mean



if __name__ == "__main__":
    
    """TRANSLATION"""
    
    # english_source_sentences = os.listdir("amr_2-four_translations/english_source_sentences")
    # files_to_translate = os.listdir("amr_2-four_translations/data")
    
    # source_languages = ['de', 'es', 'it', 'zh']
    
    # for source_language in source_languages:
    #     if source_language == 'de':
    #         german = [0, 4, 8, 12, 16]
    #         files_to_translate = [files_to_translate[i] for i in german]
    #         print("GERMAN - files:", files_to_translate, "\n")
            
    #     elif source_language == 'es':
    #         spanish= [1, 5, 9, 13, 17]
    #         files_to_translate = [files_to_translate[i] for i in spanish]
    #         print("SPANISH - files:", files_to_translate)
           
    #     elif source_language == 'it':    
    #         italian = [2, 6, 10, 14, 18]
    #         files_to_translate = [files_to_translate[i] for i in italian]
    #         print("ITALIAN - files:", files_to_translate)
            
    #     elif source_language == 'zh':   
    #         mandarin = [3, 7, 11, 15, 19]
    #         files_to_translate = [files_to_translate[i] for i in mandarin]
    #         print("MANDARIN - files:", files_to_translate)
        
        
    #     for file_to_translate, source_sentence in zip(files_to_translate, english_source_sentences):
    #         translator = Translator()
    #         translator.load_sentences("amr_2-four_translations/data/" + file_to_translate, "amr_2-four_translations/english_source_sentences/" + source_sentence)
    #         translator.translate(source_language=source_language)
    #         translator.save_translation("translations/" + file_to_translate[:-4] + "_nmt.txt")
        
    #         translator.evaluate_bleu()
    #         translator.evaluate_cosine_similarity()
            
    #         with open("translation_evaluation.txt", "a", encoding='utf-8') as fa:
    #             fa.write("\n---")
            
    """BACKTRANSLATION"""
    
    # source_sentences = sorted(os.listdir("amr_2-four_translations/data/"))
    # files_to_translate = sorted(os.listdir("translations/"))
    
    # target_languages = ['es', 'it', 'zh'] #'de', 
    
    # for target_language in target_languages:
    #     if target_language == 'de':
    #         german = [0, 4, 8, 12, 16]
    #         files_to_translate_back = [files_to_translate[i] for i in german]
    #         nmt_source_sentences = [source_sentences[i] for i in german]
    #         print("GERMAN - files to translate:", files_to_translate_back, "\nGold files:", nmt_source_sentences)
            
    #     elif target_language == 'es':
    #         spanish= [1, 5, 9, 13, 17]
    #         files_to_translate_back = [files_to_translate[i] for i in spanish]
    #         nmt_source_sentences = [source_sentences[i] for i in spanish]
    #         print("SPANISH - files:", files_to_translate_back)
           
    #     elif target_language == 'it':    
    #         italian = [2, 6, 10, 14, 18]
    #         files_to_translate_back = [files_to_translate[i] for i in italian]
    #         nmt_source_sentences = [source_sentences[i] for i in italian]
    #         print("ITALIAN - files:", files_to_translate_back)
            
    #     elif target_language == 'zh':   
    #         mandarin = [3, 7, 11, 15, 19]
    #         files_to_translate_back = [files_to_translate[i] for i in mandarin]
    #         nmt_source_sentences = [source_sentences[i] for i in mandarin]
    #         print("MANDARIN - files:", files_to_translate_back)
        
        
    #     for file_to_translate_back, source_sentence in zip(files_to_translate_back, nmt_source_sentences):
    #         translator = Translator()
    #         translator.load_sentences("translations/" + file_to_translate_back, "amr_2-four_translations/data/" + source_sentence)
    #         translator.translate(source_language="en", target_language=target_language)
    #         translator.save_translation("backtranslations/" + file_to_translate_back[:-4] + "_backtranslated.txt")
        
    #         translator.evaluate_bleu()
    #         translator.evaluate_cosine_similarity()
            
    #         with open("translation_evaluation.txt", "a", encoding='utf-8') as fa:
    #             fa.write("\n---")
            