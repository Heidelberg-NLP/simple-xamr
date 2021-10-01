# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:46:32 2021

@author: Yoalli R.G.
"""
import amrlib
from amrlib.evaluate.smatch_enhanced import compute_scores
import locale
import multiprocessing
import os

def read_file(filename):
    """
    Load english sentences into a list.

    Parameters
    ----------
    filename : String
        Path to the file containing the sentences.

    Returns
    -------
    lines
        List of english sentences.

    """
    with open(filename, encoding=locale.getpreferredencoding(), mode='r') as sf:
        lines = sf.readlines()
    return lines


def sent_to_graph(sent_list, path_to_model=None, verbose=False, device=None):
    """
    Parse english sentence to AMR graph.

    Parameters
    ----------
    sent_list : List
        List of english sentences to be parsed.
    verbose : Boolean, optional
        For printing the parsed graphs.

    Returns
    -------
    graphs
        AMR graph objects.

    """
    print("Parsing sentences to AMR...")
    stog = amrlib.load_stog_model(model_dir=path_to_model, device=device)
    print("Model loaded.")
    graphs = stog.parse_sents(sent_list)
    if verbose:
        for graph in graphs:
            print(graph)

    return graphs


def graph_to_sent(graphs, path_to_model=None, verbose=False):
    """
    Parse AMR graphs to english sentences.

    Parameters
    ----------
    graphs : Object
        Graph objects to be parsed.
    verbose : Boolean, optional
        For printing the parsed sentences.

    Returns
    -------
    sents
        List of sentences.

    """
    print("Parsing AMR graphs to sentences...")
    gtos = amrlib.load_gtos_model(model_dir=path_to_model)
    sents, _ = gtos.generate(graphs)
    if verbose:
        for sent in sents:
            print(sent)
    print("Done parsing.")
    return sents


def save_graphs(graphs, path):
    """
    Save graphs to path where each graph is separated by a new line.

    Parameters
    ----------
    graphs : Object
    path : String
        Absolute or relative file path to save graphs to.

    Returns
    -------
    None.

    """
    with open(path, mode="w", encoding='utf-8') as gr:
        for graph in graphs:
            if graph == None:
                gr.write("# ::snt\n\n(t / thing \n \t :ARG1-of (r / resemble-01))")
                gr.write("\n\n")
            else:
                gr.write(graph)
                gr.write("\n\n")
    print("Graphs saved to", path)


def evaluate_smatch(gold_path, pred_path):
    """
    Compute SMATCH score for predicted AMR graphs based on gold graphs.

    Parameters
    ----------
    gold_path : Object
    pred_path : String
        Absolute or relative file path to gold graphs and predicted graphs.

    Returns
    -------
    None.

    """
    scores = compute_scores(pred_path, gold_path)
    print("SMATCH scores: " + str(scores))

#
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     preferred_encoding = locale.getpreferredencoding()
#     preferred_encoding = 'utf-8'
#     english_source_sentences = os.listdir("amr_2-four_translations/english_source_sentences")  # where the original english sentence files are stored
#     files_to_translate = os.listdir("amr_2-four_translations/data")  # where the original source language files are stored
#     translations = sorted(os.listdir("translations/"))  # where the translated files are stored
#     gold_amr_dir = os.listdir("amr_2-four_translations/AMR")  # where the gold AMR graphs are stored
#     amr_graphs = sorted(os.listdir("AMRgraphs_GSII"))  # where to store the parsed AMR graphs
#
#     categories = ["bolt", "consensus", "dfa", "proxy", "xinhua"]  # categories included in the dataset for each language
#     languages = ["DE", "ES", "IT", "ZH"]  # languages included in the dataset
#
#     #AMR_model_dir = None #for standard model
#     AMR_model_dir = "model_parse_gsii-v0_1_0/"  # Only for running on colab or for non-standard parser: enter path to AMR model dir
#     gold_amrs_unified = "amr_2-four_translations/AMR/GOLD_AMR_unified.txt"
#
#     translate = False
#     parseamr = False
#     evaluate = True
#     unify_files = False
#
#     if parseamr:
#         # parse all translated files to AMR graphs and store them in AMRgraphs folder
#         sentences = read_file("translations/amr-release-2.0-amrs-test-proxy.sentences.ES_nmt.txt")# + translations[i])
#         graphs = sent_to_graph(sentences, AMR_model_dir, verbose=False)
#         new_path = "AMRgraphs_GSII/amr-release-2.0-amrs-test-proxy.sentences.ES_AMR.txt"
#         save_graphs(graphs, path=new_path)
#
#
#     # unify categories per language into one file and truncate parsed files before unification:
#     if unify_files:
#
#         def truncate_files(filename):
#             # cut off the last 5 lines to omit the obsolete AMR graph
#             with open(filename, mode='r+', encoding=preferred_encoding) as f:
#                 lines = f.readlines()[:-5]
#                 f.truncate()
#                 f.seek(0)
#                 f.writelines(lines)
#
#         # for file in amr_graphs:
#         #     # truncate all files
#         #     truncate_files('AMRgraphss_GSII/'+file)
#
#         unification_dict = {"DE": [], "ES": [], "IT": [], "ZH": []}
#         for lang in languages:
#             for cat in categories:
#                 # order files per language
#                 filename = "amr-release-2.0-amrs-test-" + cat + ".sentences." + lang + "_AMR.txt"
#                 unification_dict[lang].append(filename)
#         for lang in languages:
#             # unify parsed AMR graph files into one file per language
#             f0 = open("AMRgraphs_GSII/Unified-test-sentences." + lang + "_AMR.txt", 'a+', encoding=preferred_encoding)
#             for k in range(len(categories)):
#                 f1 = open("AMRgraphs_GSII/" + unification_dict[lang][k], 'r', encoding=preferred_encoding)
#                 f0.write(f1.read())
#                 f1.close()
#
#         # # unify gold data into one file:
#         # gamr = open("amr_2-four_translations/AMR/GOLD_AMR_unified.txt", 'a+', encoding=preferred_encoding)
#         # for i in range(len(gold_amr_dir)):
#         #     f1 = open("amr_2-four_translations/AMR/" + gold_amr_dir[i], 'r', encoding=preferred_encoding)
#         #     gamr.write(f1.read())
#         #     f1.close()
#
#     if evaluate:
#
#         for lang in languages:
#             print("Smatch for " + lang + ": \n")
#             evaluate_smatch(gold_amrs_unified, "AMRgraphs_GSII/Unified-test-sentences." + lang + "_AMR.txt")