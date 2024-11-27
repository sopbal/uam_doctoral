#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:50:13 2021
author: Jakub Barylski, Sophia Baldysz
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
from umap import UMAP

from gensim.models import Word2Vec
from nltk import bigrams, collocations
from pandas import DataFrame, Series, isnull, option_context
from plotly.express import scatter_3d
from plotly.graph_objs import Figure
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import fisher_exact

Measures = collocations.BigramAssocMeasures

def parse_command_line_arguments():
    """
    Requires a gff file for embedding and and a path for the output directory where the files will be stored. 
    :return:
    """

    parser = argparse.ArgumentParser(description=Path(__file__).name)

    parser.add_argument("-gff", dest="gff_path", required=False, default=[], action='append',
                        help="gff formatted file(s) with additional (external) annotations", metavar="")

    parser.add_argument("-out", dest="out_path", required=False,
                        default='',
                        help="name of the output directory [default: \"{input_name}.sif\"]", metavar="")

    parser.add_argument("-dontembed", dest="out_path", required=False,
                        default='',
                        help="name of the output directory [default: \"{input_name}.sif\"]", metavar="")

    args = parser.parse_args()

    if not args.gff_path or not args.out_path:
        parser.print_help()
        raise AttributeError('one of the required argument is missing')

    return Path(args.gff_path), Path(args.out_path)


def parse_gff(gff_path: Path) -> Tuple[DataFrame,
                                       List[Tuple[str]],
                                       DataFrame,
                                       List[List[str]]]:
    """
    Requires a gff file, where each line represents an HMM and a protein that shows similarity to that HMM
    :param gff_path: path to gff file
    :return: counts, bigram_list, one_hot_frame, train_sentences
    """
    print(f'Parsing GFF at: {gff_path.as_posix()}')
    protein_dict = defaultdict(list)
    instance_counts = defaultdict(int)
    with gff_path.open() as gff:
        comments = 0
        for i, line in enumerate(gff):
            if line.strip() and not line.startswith('#'):

                ls = line.strip().split('\t')
                protein = ls[0]
                start = ls[3]
                attributes = ls[-1].split(';')
                attributes = [pair.split('=') for pair in attributes]
                attributes = {k: v for k, v in attributes}
                hmm = attributes['HMM']

                record = {'start': start, 'model': hmm}
                protein_dict[protein].append(record)
                instance_counts[hmm] += 1
            else:
                comments += 1
    print(f'Found {i} lines including {comments} comment lines')

    print(f'Sorting annotations')
    for protein in protein_dict:
        protein_dict[protein].sort(key=lambda hmm_hit: hmm_hit['start'])
        protein_dict[protein] = [hmm_hit['model'] for hmm_hit in protein_dict[protein]]
    print(f'Done')

    print(f'Generating bigrams')
    bigram_list = []
    for protein, hmm_hists in protein_dict.items():
        if len(protein) > 1:
            bigram_list.extend([n for n in bigrams(hmm_hists)])
    print(f'Done')

    print(f'Generating presence-absence matrix')
    one_hot_frame = DataFrame(0,
                              columns=tuple(instance_counts.keys()),
                              index=tuple(protein_dict.keys()))
    for protein, models in protein_dict.items():
        for model in models:
            one_hot_frame.loc[protein, model] = 1

    protein_counts = {m: sum(one_hot_frame[m]) for m in one_hot_frame.columns}

    counts = DataFrame(columns=['instance', 'protein'], index=instance_counts.keys())
    for model in instance_counts.keys():
        counts.loc[model, 'instance'] = instance_counts[model]
        counts.loc[model, 'protein'] = protein_counts[model]

    train_sentences = list(protein_dict.values())

    return counts, bigram_list, one_hot_frame, train_sentences

def create_network(counts: DataFrame,
                   bigram_list: List[Tuple[str]],
                   one_hot_frame: DataFrame,
                   out_path: Path,
                   attraction_threshold: float = 0.1,
                   fisher_significance: float = 1e-2):
    """
    :param bigram_list: requires a bigram list created in the parse gff file, where for each hmm all the proteins are assigned to the list
    :param attraction_threshold: indicates the minimal jaccard association value, default: 0.1
    :param fisher_significance: assesses the statistical significance results of the co-occurence of domains in proteins
    :param out_path: path to directory where the results of the embedding will be stored
    :param counts: a dataframe where each model has an assigned protein instance counts which show similarity to that model  
    :type counts: DataFrame
    :param one_hot_frame: a dataframe for the proteins and their instances
    :type one_hot_frame: DataFrame
    :return: network file with sif extension to import to e.g. Cytoscape, an edges file in tsv format and a node file in tsv format 
    """

    edge_frame = DataFrame(columns=['pearson', 'jaccard', 'freq', 'abs_num', 'fisher', 'f_pval_adj', 'attraction'])
    edge_frame.index.name = 'name'

    used_models = []
    unique_bigrams = set(bigram_list)
    print(f'Calculating BIGRAM association measures')
    for n, pair in enumerate(unique_bigrams):
        for model in pair:
            if model not in used_models:
                used_models.append(model)

        edge_id = f'{pair[0]} (bigram) {pair[1]}'
        edge_frame.loc[edge_id] = Series(name=edge_id)

        bigram_contingency_marginals = bigram_contingency(pair, bigram_list)

        cor = fisher(bigram_contingency_marginals)
        edge_frame.loc[edge_id]['fisher'] = cor  # TODO this 1 - f i is highly


        jaccard = Measures.jaccard(*bigram_contingency_marginals)
        edge_frame.loc[edge_id]['attraction'] = edge_frame.loc[edge_id]['jaccard'] = jaccard

        freq = Measures.raw_freq(*bigram_contingency_marginals)
        edge_frame.loc[edge_id]['freq'] = freq
        abs_num = bigram_contingency_marginals[0]
        edge_frame.loc[edge_id]['abs_num'] = abs_num

    orphan_models = {model for model in counts.index if model not in used_models}

    print(f'Calculating order-independent node correlations')
    correlations = one_hot_frame.corr(method='pearson')

    tmp_models = list(correlations.columns)
    while tmp_models:
        model0 = tmp_models.pop()
        for model1 in tmp_models:

            edge_id = f'{model0} (pearson) {model1}'
            edge_frame.loc[edge_id] = Series(name=edge_id)

            concurrence_contingency_marginals = coocur_countingency(model0,
                                                                    model1,
                                                                    one_hot_frame)
            cor = fisher(concurrence_contingency_marginals)
            edge_frame.loc[edge_id]['fisher'] = cor
            correlation = correlations.loc[model0, model1]


            edge_frame.loc[edge_id]['attraction'] = edge_frame.loc[edge_id]['pearson'] = correlation

            freq = concurrence_contingency_marginals[0] / concurrence_contingency_marginals[2]
            edge_frame.loc[edge_id]['freq'] = freq
            abs_num = concurrence_contingency_marginals[0]
            edge_frame.loc[edge_id]['abs_num'] = abs_num

    edge_frame['f_pval_adj'] = multipletests(edge_frame['fisher'],
                                             method='bonferroni')[1]

    edge_frame.to_excel('test.xls')

    edge_frame = edge_frame.loc[(edge_frame['f_pval_adj'] < fisher_significance) & (edge_frame['attraction'] > attraction_threshold)]

    net_file = out_path.joinpath('Network.sif')
    net_lines = [edge_id.replace('(', '').replace(')', '') for edge_id in edge_frame.index]
    for model in orphan_models:
        net_lines.append(model)
    with net_file.open('w') as net_f:
        net_f.write('\n'.join(net_lines))
    node_file = out_path.joinpath('Node_weights.tsv')
    counts.to_csv(node_file.as_posix(), sep='\t')
    edge_file = out_path.joinpath('Edge_weights.tsv')
    edge_frame.to_csv(edge_file.as_posix(), sep='\t')


def bigram_contingency(counted_bigram,
                       bigram_iterable):
    """
    :param counted_bigram: a bigram for 2 models (model pair)
    :param bigram_iterable: list of bigrams, indicating through which proteins these models show bigram association
    :return:number of times a bigram occurs, number of times one model shows bigram association with the other (2 values, bi-directional), total length of bigram iterable (occurences)
    """
    ii = bigram_iterable.count(counted_bigram)
    ix = len([b for b in bigram_iterable if b[0] == counted_bigram[0]])
    xi = len([b for b in bigram_iterable if b[1] == counted_bigram[1]])
    xx = len(bigram_iterable)
    return ii, (ix, xi), xx


def coocur_countingency(model0,
                        model1,
                        one_hot_frame: DataFrame):
    """
    :param model0: one of the models tested whether it co-occurs with another model irrespective of the order (order-independent)
    :param model1: the other model tested whether it co-occurs with another model irrespective of the order (order-independent)
    :param one_hot_frame: one_hot dataframe on protein occurence within models 
    :return:number of times a pair occurs, number of times one model shows association with the other (2 values, bi-directional), total number of occurences 
    """
    ii = one_hot_frame.loc[(one_hot_frame[model0] == 1) & (one_hot_frame[model1] == 1)].shape[0]
    ix = one_hot_frame.loc[one_hot_frame[model0] == 1].shape[0]
    xi = one_hot_frame.loc[one_hot_frame[model1] == 1].shape[0]
    xx = one_hot_frame.shape[0]
    return ii, (ix, xi), xx


def fisher(marginals):
    """Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less
    sensitive to small counts than PMI or Chi Sq, but also more expensive
    to compute. Requires scipy. NLTK CORRECTION
    """

    n_ii, n_io, n_oi, n_oo = _contingency(*marginals)

    (odds, pvalue) = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative="greater")
    return pvalue


def _contingency(n_ii, n_ix_xi_tuple, n_xx):
    """Calculates values of a bigram contingency table from marginal values."""
    (n_ix, n_xi) = n_ix_xi_tuple
    n_oi = n_xi - n_ii
    n_io = n_ix - n_ii
    return n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io


def build_architecture_embedding(model_sentences: List[List[str]],
                                 counts: DataFrame,
                                 window,
                                 out_path: Path,
                                 translator: Dict[str, str],
                                 vecsize):
    """
    :param model_sentences: list of model sentences, used as input for the Word2Vec model
    :param counts: dataframe with protein count instances for models 
    :return: visualization of the embedding transformed by UMAP to be represented in a 3D space if the number of dimensions is greater than that 
    """
    print('Calculating embedding')
    model = Word2Vec(model_sentences,
                     vector_size=vecsize,
                     window=window,  # todo optimize
                     min_count=10,
                     workers=36)
    model.train(model_sentences,
                total_examples=len(model_sentences),
                epochs=20)

    vectors = [model.wv[w] for w in model.wv.key_to_index]
    if model.vector_size > 3:
        dr = UMAP(n_components=3)
        vectors = dr.fit_transform(vectors)
    vectors = {w: v for w, v in zip(model.wv.key_to_index, vectors)}
    model.save(out_path.joinpath(f"Embedding.win_{window}_d{vecsize}.word2vec").as_posix())

    visualisation_frame = DataFrame.from_dict(vectors,
                                              orient='index',
                                              columns=['x', 'y', 'z'])
    visualisation_frame.index.name = 'model'
    visualisation_frame['instances'] = [counts.loc[m, 'instance'] for m in visualisation_frame.index]
    visualisation_frame['proteins'] = [counts.loc[m, 'protein'] for m in visualisation_frame.index]
    visualisation_frame['description'] = [translator[m] for m in visualisation_frame.index]
    visualisation_frame['db'] = [m.split(':')[0] for m in visualisation_frame.index]

    size = 50
    s_ratio = max(visualisation_frame['proteins']) / max(visualisation_frame['instances'])

    core = scatter_3d(visualisation_frame,
                      x='x', y='y', z='z',
                      color='db',
                      size_max=size * s_ratio,
                      size='proteins')  # generate th 3d plot
    envelope = scatter_3d(visualisation_frame,
                          # text=visualisation_frame.index,
                          x='x', y='y', z='z',
                          color='db',
                          size='instances',
                          opacity=0.25,
                          size_max=size,
                          hover_data=['description', 'instances', 'proteins'])
    fig = Figure(data=core.data + envelope.data)
    fig.write_html(out_path.joinpath(f"Embedding.win_{window}_d{vecsize}.html").as_posix())


def main(gff_path: Path,
         out_dir_path: Path,
         translator_path: Path):
    """
    :param gff_path: path to the gff file containing models and proteins to which they show similarity 
    :param out_dir_path: path to the output directory where the results will be stored
    :return: network for bigram association and cooccurence, node and edge tsv files, visualizations of the embedding, model file of the embedding 
    """
    hmm_counts, bigrams_found, one_hot, sentences = parse_gff(gff_path)

    with translator_path.open('rb') as handle:
        translator = pickle.load(handle)

    # create_network(counts=hmm_counts,
    #                bigram_list=bigrams_found,
    #                one_hot_frame=one_hot,
    #                out_path=out_dir_path)
    for window in (1, 2, 3, 4):
        for d in (3, 4):
            build_architecture_embedding(model_sentences=sentences,
                                         counts=hmm_counts,
                                         translator=translator,
                                         out_path=out_dir_path,
                                         window=window,
                                         vecsize=d)


if __name__ == "__main__":
    # protein_gff, out_folder = parse_command_line_arguments()
    protein_gff = Path(.../example.gff)
    out_folder = Path(.../example_output_dir)  
    main_translator_path = Path(.../example_description.pkl) #pkl file containing descriptions on the HMM models used in this script 
    out_folder.mkdir(exist_ok=True, parents=True)
    main(gff_path=protein_gff,
         out_dir_path=out_folder,
         translator_path=main_translator_path)
