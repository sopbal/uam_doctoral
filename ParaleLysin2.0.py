#!/usr/bin/env python3
# coding: utf-8

# base python utilities
from collections import defaultdict
from pathlib import Path
from System_utils import Parallel
from joblib import dump

# classes used for type hinting
from typing import Any, Callable, Collection, Dict, Tuple, Type, Union

# bae math, chart, table and random functions
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pandas import DataFrame
from random import shuffle
from scipy.stats import hmean

# Bayesian optimization workflow
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

# data pre processing and result assessment
from sklearn import metrics, feature_selection
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ML model architectures available for training
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

PathLike = Union[Collection[Path], Path, str]
Scaler = Union[StandardScaler, MinMaxScaler, RobustScaler]
Classifier = Union[XGBClassifier, RandomForestClassifier, MLPClassifier, SVC, Pipeline]


def add_file(in_file,
             def_dict: defaultdict) -> defaultdict:
    """
    :param in_file: takes a representation file or a label file and reads it. The input is a tab separated file
    :param def_dict: dictionary where the keys are protein ids, a the values the rest of the line in the file
    (features in a representation file or labels in a label file)
    :return: def_dict
    """
    in_file = Path(in_file)
    with in_file.open() as f:
        for line in f:
            if not line.startswith('#'):
                split_line = line.split()
                seqid = split_line.pop(0)
                def_dict[seqid].extend(split_line)

    return def_dict


def load_representation(input_files: PathLike,
                        labels: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vectors for each protein from one ore more files
    If multiple files are loaded then they are concatenated
    :param input_files: path or iterable of paths to files with tabular vectors
    :param labels: path to file with [0/1] vector labels
    :return: numpy array of shuffled vectors and shuffled labels for a classifier
    """
    if isinstance(input_files, (Path, str)):  
        input_files = [input_files]
    vector_dict = defaultdict(list)
    for in_file in input_files:
        vector_dict = add_file(in_file, vector_dict)

    label_dict = defaultdict(list)
    label_dict = add_file(labels, label_dict)

    labeled_vectors = []

    for protein, vector in vector_dict.items():
        label, = label_dict[protein]    
        labeled_vectors.append((vector, label))

    # shuffle the data to prevent clumping of positives, which might confuse the model
    shuffle(labeled_vectors)

    # inputs for the classifier must be numpy arrays, hence the switch of data format
    vectors = []
    labels = []
    while labeled_vectors:
        x, y = labeled_vectors.pop()
        y, = y
        vectors.append(x)
        labels.append(y)
    return np.array(vectors, dtype=float), np.array(labels, dtype=int)


def train(trainvectors: np.ndarray,
          trainlabels: np.ndarray,
          classifiertype: Callable,
          parameters: Dict[str, Any],
          scaler: Callable = StandardScaler) -> Pipeline:
    """
    :param trainvectors: a numpy array with numerical values for each feature
    :param trainlabels: a numpy array with numerical values for each category (in a 2-class classification event it would be
    0 or 1, denoting the two separate classes)
    :param classifiertype: denotes the classifier type from a list of classifiers, example: SVM
    :param parameters: denotes the type of parameters
    :param scaler: takes all values to one range (standard scaler puts all feature values in a -1 to 1 range)
    :return: classifier object
    """
    myclf = str(classifier)
    transform = feature_selection.SelectPercentile(feature_selection.f_classif)
    classifier = make_pipeline(transform(), scaler(), classifiertype(random_state=1964, **parameters))
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    for percentile in percentiles:
        classifier.set_params(anova__percentile=percentile)
        # Compute cross-validation score using 1 CPU
        this_scores = cross_val_score(classifier, trainvectors, trainlabels)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    feature_area = px.line(
        percentiles, error_x= score_means, error_y= np.array(score_stds)
        title=f' Performance of the {classifiertype}-Anova varying the percentile of features selected',
        labels=dict(x='Percentile', y='Prediction rate'),
        width=700, height=500
    )
    feature_area.write_html(path_stem.as_posix() + '.featureselection.html')

    classifier.fit(trainvectors, trainlabels)
    return classifier

def evaluate(classifier: Classifier,
             test_vectors: np.ndarray,
             test_labels: np.ndarray):
    """
    :param classifier: a trained classifier object obtained from the train function, from the sklearn package
    :param test_vectors: a numpy array of the vector representation of the test set
    :param test_labels: a numpy array of the test labels (0 or 1 in a binary classification)
    :return: metrics for assessment of the classifier
    """
    predictions = classifier.predict(test_vectors)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    sensitivity = metrics.recall_score(test_labels, predictions)
    probas_pred = classifier.predict_proba(test_vectors)[:, 1]
    auroc = metrics.roc_auc_score(test_labels, probas_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(test_labels, probas_pred)
    auprc = metrics.auc(recall, precision)
    f1_max = max(hmean([precision, recall]))

    return f1_max, accuracy, sensitivity, auroc, auprc


def draw_curves(classifier: Classifier,
                test_vectors: np.ndarray,
                test_labels: np.ndarray,
                path_stem: Path):
    probas_pred = classifier.predict_proba(test_vectors)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(test_labels, probas_pred)
    auroc = metrics.auc(fpr, tpr)
    roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC curve (AUC={auroc:.4f})',
        labels=dict(x='false positive rate', y='true positive rate'),
        width=700, height=500
    )
    roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    roc.update_yaxes(scaleanchor="x", scaleratio=1)
    roc.update_xaxes(constrain='domain')
    roc.write_html(path_stem.as_posix() + '.ROC.html')


    precision, recall, thresholds = metrics.precision_recall_curve(test_labels, probas_pred)
    auprc = metrics.auc(recall, precision)
    prc = px.area(
        x=recall, y=precision,
        title=f'P-R curve (AUC={auprc:.4f})',
        labels=dict(x='precision', y='recall'),
        width=700, height=500
    )
    prc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    prc.update_yaxes(scaleanchor="x", scaleratio=1)
    prc.update_xaxes(constrain='domain')
    prc.write_html(path_stem.as_posix() + '.PRC.html')

def bayes(datasets: Collection[Collection[Path]],
          classifiertype: Type[Classifier],
          out_directory: Path):
    """
    :param datasets: a set of directories with subsets of the same vector representation in each subfolder
    :param trainfile: a file with the vector representations for the training set for the classifier
    :param testfile: a file with the vector representations for the test set for the classifier
    :param trainlabels: a file with the labels of the samples used in the training set
    :param testlabels: a file with the labels of the samples used in the test set
    :param classifiertype: the type of classifier used in the classification process. example: MLPClassifier from sklearn
    :return: the value of the optimized set of parameters obtained in the bayes optimization process
    """

    continous = {SVC: {'C': (1e-4, 1e4), 'gamma': (1e-4, 1e4)},
                 MLPClassifier: {'alpha': (1e-3, 1e3)},
                 RandomForestClassifier: {},
                 XGBClassifier: {'max_depth': (1, 100),
                                 'n_estimators': (10, 1000),
                                 'gamma': (0, 1000),
                                 'eta': (0.01, 0.99)}}

    discrete = {SVC: {},
                MLPClassifier: {},
                RandomForestClassifier: {'max_depth': (3, 100),
                                         'n_estimators': (10, 1000)},
                XGBClassifier: {}}

    cathegorical = {SVC: {'kernel': ('linear', 'poly', 'rbf', 'sigmoid')},
                    MLPClassifier: {'solver': ('lbfgs', 'sgd', 'adam'),
                                    'activation': ('identity', 'logistic', 'tanh', 'relu'),
                                    'hidden_layer_sizes': (
                                        (100, 25, 5), (50, 100, 10), (200, 20), (300, 30), (100, 50), (250,), (150,),
                                        (50,),
                                        (20,))},
                    RandomForestClassifier: {},
                    XGBClassifier: {}}
    override = {SVC: {'probability': True},
                MLPClassifier: {'max_iter': int(1e6)},
                RandomForestClassifier: {},
                XGBClassifier: {}}

    def encode(continous_params: Dict[str, Tuple[float, float]],
               discrete_params: Dict[str, Tuple[int, int]],
               cathegorical_params: Dict[str, Collection]) -> Dict[str, Tuple[float, float]]:
        parameter_bounds = continous_params
        for param, bounds in discrete_params.items():
            parameter_bounds[param] = (bounds[0], bounds[1] + 1)  
        for param, variants in cathegorical_params.items():
            parameter_bounds[param] = (
                0, len(variants) - 1e-10)  
        return parameter_bounds

    def decode(selected_parameters,
               discrete_params: Dict[str, Tuple[int, int]],
               cathegorical_params: Dict[str, Collection],
               override_params: Dict[str, Collection]) -> Dict[str, Any]:
        """
        :param override_params:
        :param selected_parameters: set of selected parameters to optimize, different for each type of classifier
        :param discrete_params: subset of the selected parameters which are discrete values
        :param cathegorical_params: subset of the selected parameters which are categorical values, further changed to numerical
        :return: set of parameters which will be optimized in the bayes optimization process
        """
        try:
            back_translated_parameters = {}
            for param, value in selected_parameters.items():
                if param in discrete_params:
                    back_translated_parameters[param] = int(value)
                elif param in cathegorical_params:
                    back_translated_parameters[param] = tuple(cathegorical_params[param])[int(value)]
                else:
                    back_translated_parameters[param] = value
            back_translated_parameters.update(override_params)
            return back_translated_parameters
        except:
            raise ValueError(f'{selected_parameters}\n{discrete_params}\n{cathegorical_params}')

    all_results = []

    def procedure_wrapper(**kwargs):  
        """
        :param selected_parameters: set of selected parameters which will be optimized in the bayes optimization process
        :return: avg value of the f1 score (max f1 score) as a metric to assess the ability of the classifier to properly
        classify the samples
        """
        result_summary = {}
        parameters = decode(kwargs,
                            discrete[classifiertype],
                            cathegorical[classifiertype],
                            override[classifiertype])
        result_summary.update(parameters)

        jobs = Parallel(run_iteration, [(parameters, dataset) for dataset in datasets])

        used_metrics = ('f1', 'accuracy', 'sensitivity', 'auroc', 'auprc')
        metric_dict = defaultdict(list)
        for partial_result in jobs.result:
            partial_result = list(partial_result)
            classifier, test_vectors, test_labels = partial_result.pop()
            for metric_name, metric_value in zip(used_metrics, partial_result):
                metric_dict[metric_name].append(metric_value)
        for metric_name, metric_values in metric_dict.items():
            for stat_function in (np.mean, np.std):
                result_summary[f'{metric_name}_{stat_function.__name__}'] = stat_function(metric_values)

        f1_mean = result_summary['f1_mean']
        if not all_results or f1_mean > all_results[0]['f1_mean']:
            all_results.insert(0, result_summary)
            print(DataFrame.from_records([result_summary]))
            draw_curves(classifier, test_vectors, test_labels, out_directory.joinpath('Best_classifier'))
            model_path = out_directory.joinpath('Best_classifier.jpkl')
            with model_path.open('wb') as model_file:
                dump(classifier, model_file)
        else:
            all_results.append(result_summary)
        return f1_mean

    def run_iteration(input_data):
        parameters, dataset = input_data
        trainfile, trainlabels, testfile, testlabels = dataset
        train_vectors, train_labels = load_representation(trainfile, trainlabels)
        classifier = train(train_vectors, train_labels, classifiertype, parameters)
        test_vectors, test_labels = load_representation(testfile, testlabels)
        f1_max, accuracy, sensitivity, auroc, auprc = evaluate(classifier, test_vectors, test_labels)
        return f1_max, accuracy, sensitivity, auroc, auprc, (classifier, test_vectors, test_labels)

    pbounds = encode(continous[classifiertype],
                     discrete[classifiertype],
                     cathegorical[classifiertype])
    print(pbounds)
    optimizer = BayesianOptimization(f=procedure_wrapper,
                                     pbounds=pbounds,
                                     random_state=1342,
                                     verbose=2)
    logger = JSONLogger(path=out_directory.joinpath('log.json').as_posix())
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=100,
                       n_iter=200)

    formated_results = DataFrame.from_records(all_results)
    full_report = out_directory.joinpath('report.xlsx')
    formated_results.to_excel(full_report.as_posix())

    return optimizer.max


def read_datasets(master_dir: Path, representations: Collection[str], input_sets: Collection[int] = (1, 2, 3)):
    datasets = []
    for input_set in input_sets:
        train_vec_files = []
        test_vec_files = []
        for rep in representations:
            data_subir = master_dir.joinpath(f'{input_set}__{rep}')
            train_vec_files.append(data_subir.joinpath('trainset.txt'))
            test_vec_files.append(data_subir.joinpath('testset.txt'))
        train_labels = data_subir.joinpath('trainlabels.txt')
        test_labels = data_subir.joinpath('testlabels.txt')
        datasets.append((train_vec_files, train_labels, test_vec_files, test_labels))
    return datasets

data_dir = Path('...example_Directory_with_feature_representations')

out_dir = Path('...example_output_directory')

seq_representations = ('PAAC', 'CKSAAP', 'DDE', 'DPC', 'TPC')

architectures = (MLPClassifier, RandomForestClassifier, SVC, XGBClassifier)

for c in architectures:
    out_dir = Path(f'...exampledirectory/ParaleLysins_{c.__name__}.1.0')
    for rep in seq_representations:

        datasets = (rep, 'fizchem')

        data = read_datasets(data_dir, datasets)

        subdir = out_dir.joinpath('.'.join(datasets))
        subdir.mkdir(parents=True)

        print(bayes(data, MLPClassifier, subdir))