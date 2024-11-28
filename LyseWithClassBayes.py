#!/usr/bin/env python
# coding: utf-8

# base python utilities
from collections import defaultdict
from pathlib import Path

# classes used for type hinting
from typing import Any, Callable, Collection, Dict, Tuple, Type, Union

# bae math and random functions
import numpy as np
from random import shuffle
from scipy.stats import hmean

# Bayesian optimization workflow
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

# data pre processing and result assessment
from sklearn import metrics
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ML model architectures available for training
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

PathLike = Union[Collection[Path], Path, str]
Scaler = Union[StandardScaler, MinMaxScaler, RobustScaler]
Classifier = Union[XGBClassifier, RandomForestClassifier, MLPClassifier, SVC, Pipeline]


def add_file(in_file,
             def_dict: defaultdict) -> defaultdict:
    """
    :param in_file: input file, usually in txt format for  vectors and vector labels 
    :param def_dict: dictionary where the keys are the sequence ids and the values are a list of vector values from different representations 
    :return:dictionary
    """
    in_file = Path(in_file)
    with in_file.open() as f:
        for line in f:
            if not line.startswith('#'):
                line = line.split()
                seqid = line.pop(0)
                def_dict[seqid].extend(line)
    return def_dict


def load_representation(input_files: PathLike,
                        labels: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vectors for each protein from one ore more files
    If multiple files are loaded they are concatenated
    :param input_files: path or iterable of paths to files with tabular vectors
    :param labels: path to file with [0/1] vector labels
    :return: array of shuffled vectors (float) and labels (int)
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
    :param trainvectors: vectors (float) of the train sequence data, for every sequence each representation was calculated 
    :param trainlabels: labels (0 or 1) for positive and negative distinction of protein sequences 
    :param classifiertype: type of the classifier used - neural network, support vector machine, random forest, etc. from the sklearn package
    :param parameters: the type of classifier hyperparameters, specific for each classifier to be determined 
    :param scaler: type of scaler used in data preprocessing
    :return: trained classifier 
    """
    classifier = make_pipeline(scaler(), classifiertype(random_state=1964, **parameters))
    classifier.fit(trainvectors, trainlabels)
    return classifier


def evaluate(classifier: Classifier,
             test_vectors: np.ndarray,
             test_labels: np.ndarray):
    """
    :param classifier: trained classifier from the train function
    :param test_vectors: vectors (float) of the test sequence data, for every sequence each representation was calculated
    :param test_labels: labels (0 or 1) for positive and negative distinction of protein sequences 
    :return: classifier evaluation metrics - f1, accuracy, sensitivity and roc_auc_score
    """
    predictions = classifier.predict(test_vectors)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    sensitivity = metrics.recall_score(test_labels, predictions)
    probas_pred = classifier.predict_proba(test_vectors)[:, 1]
    roc_auc_score = metrics.roc_auc_score(test_labels, probas_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(test_labels, probas_pred)
    f1_max = max(hmean([precision, recall]))
    return f1_max, accuracy, sensitivity, roc_auc_score


def bayes(trainfile: Path,
          testfile: Path,
          trainlabels: Path,
          testlabels: Path,
          classifiertype: Type[Classifier],
          log_file: Path):
    """
    :param trainfile: file with sequence representations (vectors) for the training sequence data
    :param testfile: file with sequence representations (vectors) for the test sequence data 
    :param trainlabels: file with labels (0 or 1) for the training sequence data
    :param testlabels: file with labels (0 or 1) for the test sequence data 
    :param classifiertype: type of classifier used - neural network, svm, random forest, etc. 
    :return: optimal classifier, with optimal values of evaluation metrics 
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
                    MLPClassifier: {'solver': ('lbfgs', 'adam'),
                                    'activation': ('logistic', 'tanh', 'relu'),
                                    'hidden_layer_sizes': ((30, 300), (300, 30), (300,), (30, 300, 30), (100, 50, 25))},
                    RandomForestClassifier: {},
                    XGBClassifier: {}}
    override = {'max_iter': int(1e6)}

    def encode(continous_params: Dict[str, Tuple[float, float]],
               discrete_params: Dict[str, Tuple[int, int]],
               cathegorical_params: Dict[str, Collection]) -> Dict[str, Tuple[float, float]]:
        parameter_bounds = continous_params
        for param, bounds in discrete_params.items():
            parameter_bounds[param] = (bounds[0], bounds[1] + 1)  # assure that "border values" equal probabilities
        for param, variants in cathegorical_params.items():
            parameter_bounds[param] = (0, len(variants) - 1e-10)  # here the same is guaranteed by th python 0-based indexing
        return parameter_bounds


    def decode(selected_parameters,
               discrete_params: Dict[str, Tuple[int, int]],
               cathegorical_params: Dict[str, Collection]) -> Dict[str, Any]:
        """
        :param selected_parameters: selected parameters of the classifier, discrete, cathegorical or continous
        :param discrete_params: parameters that are discrete, here only for the Random Forest Classifier
        :param cathegorical_params: parameters that are cathegorical, here in the SVC or MLP classifiers
        :return: dictionary of selected parameters 
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
            back_translated_parameters.update(override)
            return back_translated_parameters
        except:
            raise ValueError(f'{selected_parameters}\n{discrete_params}\n{cathegorical_params}')

    def procedure_wrapper(**kwargs):  # model_maxn
        """
        :param selected_parameters: wrapper for the parameters selection 
        :return:maximal value of f1 (could be changed to a different value in theory, here best classifier selected based on max f1 value)
        """
        parameters = decode(kwargs,
                            discrete[classifiertype],
                            cathegorical[classifiertype])

        train_vectors, train_labels = load_representation(trainfile, trainlabels)
        classifier = train(train_vectors, train_labels, classifiertype, parameters)
        test_vectors, test_labels = load_representation(testfile, testlabels)
        f1_max, accuracy, sensitivity, roc_auc_score = evaluate(classifier, test_vectors, test_labels)
        print(f'Tested variant: {parameters} F1: {f1_max}')
        return f1_max



    pbounds = encode(continous[classifiertype],
                     discrete[classifiertype],
                     cathegorical[classifiertype])
    print(pbounds)
    optimizer = BayesianOptimization(f=procedure_wrapper,
                                     pbounds=pbounds,
                                     random_state=1342,
                                     verbose=2)
    logger = JSONLogger(path=log_file.as_posix())
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=100,
                       n_iter=200)

    return optimizer.max


trainfile = Path('...exampletrainfile.txt')
testfile = Path('...exampletestfile.txt')
trainlabelsfile = Path('...examplelabeltrainfile.txt')
testlabelsfile = Path('...examplelabeltestfile.txt')

log_file = Path('..examplelogfile.log')

print(bayes(trainfile, testfile, trainlabelsfile, testlabelsfile, MLPClassifier, log_file))
