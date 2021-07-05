from amr_parser import sent_to_graph, graph_to_sent, read_file, save_graphs, evaluate_smatch
from nmt_english import Translator
import multiprocessing
import os
import locale


def truncate_files(filename):
    # cut off the last 5 lines to omit the obsolete AMR graph
    with open(filename, mode='r+', encoding=preferred_encoding) as f:
        lines = f.readlines()[:-5]
        f.truncate()
        f.seek(0)
        f.writelines(lines)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    preferred_encoding = locale.getpreferredencoding()
    preferred_encoding = 'utf-8'
    english_source_sentences = os.listdir("amr_2-four_translations/english_source_sentences")  # where the original english sentence files are stored
    files_to_translate = os.listdir("amr_2-four_translations/data")  # where the original source language files are stored
    translations = sorted(os.listdir("translations"))  # where the translated files are stored
    gold_amr_dir = os.listdir("amr_2-four_translations/AMR")  # where the gold AMR graphs are stored
    amr_graphs = sorted(os.listdir("AMRgraphs"))  # where to store the parsed AMR graphs

    categories = ["bolt", "consensus", "dfa", "proxy", "xinhua"]  # categories included in the dataset for each language
    languages = ["DE", "ES", "IT", "ZH"]  # languages included in the dataset

    AMR_model_dir = None  # Only for running on colab: enter path to AMR model dir
    gold_amrs_unified = "amr_2-four_translations/AMR/GOLD_AMR_unified.txt"

    translate = False
    parseamr = False
    amr_gsii = True
    evaluate = True
    unify_files = False

    if translate:
        source_languages = ['de', 'es', 'it', 'zh']
    
        for source_language in source_languages:
            if source_language == 'de':
                german = [0, 4, 8, 12, 16]
                files_to_translate = [files_to_translate[i] for i in german]
                print("GERMAN - files:", files_to_translate, "\n")
                
            elif source_language == 'es':
                spanish= [1, 5, 9, 13, 17]
                files_to_translate = [files_to_translate[i] for i in spanish]
                print("SPANISH - files:", files_to_translate)
               
            elif source_language == 'it':    
                italian = [2, 6, 10, 14, 18]
                files_to_translate = [files_to_translate[i] for i in italian]
                print("ITALIAN - files:", files_to_translate)
                
            elif source_language == 'zh':   
                mandarin = [3, 7, 11, 15, 19]
                files_to_translate = [files_to_translate[i] for i in mandarin]
                print("MANDARIN - files:", files_to_translate)
            
            
            for file_to_translate, source_sentence in zip(files_to_translate, english_source_sentences):
                translator = Translator()
                translator.load_sentences("amr_2-four_translations/data/" + file_to_translate, "amr_2-four_translations/english_source_sentences/" + source_sentence)
                translator.translate(source_language=source_language)
                translator.save_translation("translations/" + file_to_translate[:-4] + "_nmt.txt")
            
                translator.evaluate_bleu()
                translator.evaluate_cosine_similarity()
                
                with open("translation_evaluation.txt", "a", encoding='utf-8') as fa:
                    fa.write("\n---")

    if parseamr:
        if amr_gsii:
            amr_graphs = sorted(os.listdir("AMRgraphs_GSII"))  # where to store the parsed AMR graphs
            # AMR_model_dir = None #for standard model
            AMR_model_dir = "model_parse_gsii-v0_1_0/"  # Only for running on colab or for non-standard parser: enter path to AMR model dir
            # parse all translated files to AMR graphs and store them in AMRgraphs folder
            sentences = read_file("translations/amr-release-2.0-amrs-test-proxy.sentences.ES_nmt.txt")  # + translations[i])
            graphs = sent_to_graph(sentences, AMR_model_dir, verbose=False)
            new_path = "AMRgraphs_GSII/amr-release-2.0-amrs-test-proxy.sentences.ES_AMR.txt"
            save_graphs(graphs, path=new_path)

        else:
            for i in range(len(translations)):
                # parse all translated files to AMR graphs and store them in AMRgraphs folder
                sentences = read_file("translations/" + translations[i])
                graphs = sent_to_graph(sentences, AMR_model_dir)
                new_path = "AMRgraphs/" + translations[i][:-8] + "_AMR.txt"
                save_graphs(graphs, path=new_path)

        if evaluate:
            # unify categories per language into one file and truncate parsed files before unification:
            if amr_gsii:
                # unify categories per language into one file and truncate parsed files before unification:
                if unify_files:
                    # for file in amr_graphs:
                    #     # truncate all files
                    #     truncate_files('AMRgraphss_GSII/'+file)
                    unification_dict = {"DE": [], "ES": [], "IT": [], "ZH": []}
                    for lang in languages:
                        for cat in categories:
                            # order files per language
                            filename = "amr-release-2.0-amrs-test-" + cat + ".sentences." + lang + "_AMR.txt"
                            unification_dict[lang].append(filename)
                    for lang in languages:
                        # unify parsed AMR graph files into one file per language
                        f0 = open("AMRgraphs_GSII/Unified-test-sentences." + lang + "_AMR.txt", 'a+',
                                  encoding=preferred_encoding)
                        for k in range(len(categories)):
                            f1 = open("AMRgraphs_GSII/" + unification_dict[lang][k], 'r', encoding=preferred_encoding)
                            f0.write(f1.read())
                            f1.close()

                    # # unify gold data into one file:
                    # gamr = open("amr_2-four_translations/AMR/GOLD_AMR_unified.txt", 'a+', encoding=preferred_encoding)
                    # for i in range(len(gold_amr_dir)):
                    #     f1 = open("amr_2-four_translations/AMR/" + gold_amr_dir[i], 'r', encoding=preferred_encoding)
                    #     gamr.write(f1.read())
                    #     f1.close()

                for lang in languages:
                    print("Smatch for " + lang + ": \n")
                    evaluate_smatch(gold_amrs_unified, "AMRgraphs_GSII/Unified-test-sentences." + lang + "_AMR.txt")
            else:
                if unify_files:
                    for file in amr_graphs:
                        # truncate all files
                        truncate_files('AMRgraphs/'+file)
                    unification_dict = {"DE": [], "ES": [], "IT": [], "ZH": []}
                    for lang in languages:
                        for cat in categories:
                            # order files per language
                            filename = "amr-release-2.0-amrs-test-" + cat + ".sentences." + lang + "_AMR.txt"
                            unification_dict[lang].append(filename)
                    for lang in languages:
                        # unify parsed AMR graph files into one file per language
                        f0 = open("AMRgraphs/Unified-test-sentences." + lang + "_AMR.txt", 'a+', encoding=preferred_encoding)
                        for k in range(len(categories)):
                            f1 = open("AMRgraphs/" + unification_dict[lang][k], 'r', encoding=preferred_encoding)
                            f0.write(f1.read())
                            f1.close()
                    # unify gold data into one file:
                    gamr = open("amr_2-four_translations/AMR/GOLD_AMR_unified.txt", 'a+', encoding=preferred_encoding)
                    for i in range(len(gold_amr_dir)):
                        f1 = open("amr_2-four_translations/AMR/" + gold_amr_dir[i], 'r', encoding=preferred_encoding)
                        gamr.write(f1.read())
                        f1.close()

                for lang in languages:
                    print("Smatch for " + lang + ": \n")
                    evaluate_smatch(gold_amrs_unified, "AMRgraphs/Unified-test-sentences." + lang + "_AMR.txt")