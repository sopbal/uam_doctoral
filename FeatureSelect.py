#!/usr/bin/env python3
# coding: utf-8

# base python utilities
from collections import defaultdict
from pathlib import Path
from System_utils import Parallel, PathLike
from Learning_utils import ScalerLike, ClassifierLike, \
    Selector, train, evaluate, \
    load_representation, param_bounds, params_verbatim, draw_curves
from joblib import dump

# classes used for type hinting
from typing import Any, Callable, Collection, Dict, Hashable, List, Tuple, Type, Union

# bae math, chart, table and random functions
import numpy as np
import pandas as pd

# Bayesian optimization workflow
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

# ML model architectures and feature selection methods available for training
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, \
    SelectFromModel, _from_model, SelectKBest


def bayes(train_test_dir: Path,
          full_data_dir: Path,
          replicates: Collection[str],
          classifiertype: Type[ClassifierLike],
          selectortype: Callable,
          out_dir: Path):
    """
    :param train_test_dir: path to directory with train and test datasets 
    :param datasets: train and test files containing the feature representations used in the model training and testing 
    :param classifiertype: type of classifier used in the process, e.g. MLPC
    :return: optimized process and two results files, the parameter report (the best model parameters) and the feature report (most significant features)
    """

    def procedure_wrapper(**kwargs):  # model_maxn
        """
        :param selected_parameters: depending on the type of classifier selected, specific parameters to a specific classifier 
        :return: F1 mean of the F1 scores for all of the assessed datasets 
        """
        metric_summary = {}
        parameters = params_verbatim(kwargs,
                                     discrete[classifiertype],
                                     cathegorical[classifiertype],
                                     override[classifiertype])
        metric_summary.update(parameters)

        jobs = Parallel(iteration,
                        [(parameters, dataset) for dataset in train_and_test_data])

        used_metrics = ('f1', 'accuracy', 'sensitivity', 'auroc', 'auprc', 'precision', 'recall')
        metric_dict = defaultdict(list)
        classifier_archive = []
        for partial_result in jobs.result:
            partial_result = list(partial_result)
            classifier_archive.append(partial_result.pop())
            for metric_name, metric_value in zip(used_metrics, partial_result):
                metric_dict[metric_name].append(metric_value)
        for metric_name, metric_values in metric_dict.items():
            for stat_function in (np.mean, np.std):
                metric_summary[f'{metric_name}_{stat_function.__name__}'] = stat_function(metric_values)

        f1_mean = metric_summary['f1_mean']
        if not ALL_RESULTS or f1_mean > ALL_RESULTS[0]['f1_mean']:
            ALL_RESULTS.insert(0, metric_summary)
            metric_str = '; '.join(
                [f'{m}: {v:.3f}' if isinstance(v, float) else f'{m}: {v}' for m, v in metric_summary.items()])
            print(f'Better result found:\n{metric_str}')
            draw_curves(classifier_archive, metric_summary, out_dir.joinpath('Best_classifier'))
            for i, (classifier, selector, _, _) in enumerate(classifier_archive, 1):
                model_path = out_dir.joinpath(f'Best_classifier.dataset_{i}.jpkl')
                with model_path.open('wb') as model_file:
                    dump(classifier, model_file)
                selector_path = out_dir.joinpath(f'Best_selector.dataset_{i}.jpkl')
                with selector_path.open('wb') as model_file:
                    dump(selector, model_file)
        else:
            ALL_RESULTS.append(metric_summary)
        return f1_mean

    def iteration(input_data):
        parameters, dataset = input_data
        trainfile, trainlabels, testfile, testlabels = dataset
        train_vectors, train_labels = load_representation(trainfile, trainlabels)
        classifier, selector = train(train_vectors,
                                     train_labels,
                                     classifiertype,
                                     parameters,
                                     feature_selector)
        test_vectors, test_labels = load_representation(testfile, testlabels)
        train_names, test_names = tuple(train_vectors.columns), tuple(test_vectors.columns)
        assert train_names == test_names == feature_selector.f_names, \
            f'Headers not matched \nTRAIN: {train_names}\nTEST: \n{test_names}\nSELECTOR: {feature_selector.f_names}'
        test_vectors = feature_selector.transform(test_vectors)
        f1_max, accuracy, sensitivity, auroc, auprc, precission, recall = evaluate(classifier, test_vectors,
                                                                                   test_labels)

        return f1_max, accuracy, sensitivity, auroc, auprc, precission, recall, \
               (classifier, selector, test_vectors, test_labels)

    # run the optimisation procedure

    continous = {SVC: {'C': (1e-4, 1e4), 'gamma': (1e-4, 1e4)},
                 MLPClassifier: {'alpha': (0.5, 5)},
                 RandomForestClassifier: {},
                 XGBClassifier: {'gamma': (0, 1000),
                                 'eta': (0.01, 0.99)},
                 GaussianNB: {'var_smoothing': (1e-15, 1e-5)}}

    discrete = {SVC: {},
                MLPClassifier: {},
                RandomForestClassifier: {'max_depth': (3, 10),
                                         'n_estimators': (10, 100)},
                XGBClassifier: {'gamma': (0, 1000),
                                'eta': (0.01, 0.99)},
                GaussianNB: {}}

    cathegorical = {SVC: {'kernel': ('linear', 'poly', 'rbf', 'sigmoid')},
                    MLPClassifier: {  # 'solver': ('lbfgs', 'sgd', 'adam'),
                        # 'activation': ('identity', 'logistic', 'tanh', 'relu'),
                        # 'hidden_layer_sizes': (
                        #     (100, 25, 5),
                        #     (100, 50, 25),
                        #     (200, 50, 10),
                        #     (50, 100, 10),
                        #     (200, 20),
                        #     (300, 30),
                        #     (100, 50),
                        #     (250,),
                        #     (150,),
                        #     (50,),
                        #     (20,))
                    },
                    RandomForestClassifier: {},
                    XGBClassifier: {},
                    GaussianNB: {}}

    override = {SVC: {'probability': True},
                MLPClassifier: {'max_iter': int(1e6),
                                'solver': 'sgd',  # TODO only for feature optimisation
                                'activation': 'relu',  # TODO only for feature optimisation
                                'hidden_layer_sizes': (250,)
                                },
                RandomForestClassifier: {},
                XGBClassifier: {},
                GaussianNB: {}}

    full_representation_files = []
    for representation in representations:
        full_representation_files.append(full_data_dir.joinpath(f'{representation}.txt'))
    all_labels_file = full_data_dir.joinpath('labels.txt')
    print('Loading data...')
    full_vectors, full_labels = load_representation(full_representation_files, all_labels_file)
    print('Calibrating feature selector...')
    feature_selector = Selector(selectortype, full_vectors, full_labels)

    train_and_test_data = read_datasets(train_test_dir, representations, replicates)

    # discrete[classifiertype]['k_features'] = (3, full_vectors.shape[1])
    discrete[classifiertype]['k_features'] = (50, 500)

    ALL_RESULTS = []

    pbounds = param_bounds(continous[classifiertype],
                           discrete[classifiertype],
                           cathegorical[classifiertype])

    print('Starting optimisation process...')
    optimizer = BayesianOptimization(f=procedure_wrapper,
                                     pbounds=pbounds,
                                     random_state=1342,
                                     verbose=2)
    logger = JSONLogger(path=out_dir.joinpath('log.json').as_posix())
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=40,
                       n_iter=60)

    formatted_results = pd.DataFrame.from_records(ALL_RESULTS)
    formatted_results.sort_values('f1_mean', inplace=True, ascending=False)
    formatted_results = formatted_results.reindex()
    parameter_report_file = out_dir.joinpath('parameter_report.xlsx')
    formatted_results.to_excel(parameter_report_file.as_posix())

    feature_report_file = out_dir.joinpath('feature_report.xlsx')

    best_params = params_verbatim(optimizer.max['params'],
                                  discrete[classifiertype],
                                  cathegorical[classifiertype],
                                  override[classifiertype])

    feature_selector.save_report(full_vectors, best_params['k_features'], feature_report_file)

    best_result = optimizer.max['target']
    print(f'BEST RESULT FOUND: {best_result}')
    best_param_string = '; '.join([f'{p}: {v}' for p, v in best_params.items()])
    print(f'Parameters: {best_param_string}')


def read_datasets(master_dir: Path, representations: Collection[str], replicates: Collection[str]):
    datasets = []
    for replicate in replicates:
        train_vec_files = []
        test_vec_files = []
        for representation in representations:
            data_subir = master_dir.joinpath(f'{replicate}__{representation}')
            train_vec_files.append(data_subir.joinpath('trainset.txt'))
            test_vec_files.append(data_subir.joinpath('testset.txt'))
        train_labels = data_subir.joinpath('trainlabels.txt')
        test_labels = data_subir.joinpath('testlabels.txt')
        datasets.append((train_vec_files, train_labels, test_vec_files, test_labels))
    return datasets


train_and_test_directory = Path('.../example_representation_directory')
full_data_directory = Path('.../example_full_dataset_directory') #contains label and representation files 

classifier_architecture = MLPClassifier
selection_methods = (RandomForestClassifier,)

used_replicates = ('1', '2', '3')

for selection_method in selection_methods:
    out_directory = Path(
        f'...examplefolder/ParaleLysins_DROP500_WithSelector_{selection_method.__name__}_{classifier_architecture.__name__}.2.0')

    representations = ('fizchem', 'PAAC', 'CKSAAP')

    subdir = out_directory.joinpath('.'.join(representations))
    subdir.mkdir(parents=True)

    bayes(train_and_test_directory,
          full_data_directory,
          used_replicates,
          classifier_architecture,
          selection_method,
          subdir)
