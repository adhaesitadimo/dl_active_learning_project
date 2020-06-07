import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import itertools
import json
import multiprocessing
import os
import re
import sys
import time

import click
import flair
import numpy as np
import pandas as pd
import torch
from actleto import ActiveLearner, make_libact_strategy_ctor
from flair.data import Dictionary
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import ELMoEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from libact.query_strategies import RandomSampling, UncertaintySampling
from seqeval.metrics import f1_score
from tqdm import tqdm_notebook as tqdm

from src.active_learning_seq import RandomSamplingWithRetraining
from src.flair_libact import PositiveLessCertain, SequenceTaggerActiveStudent
from src.logger import initialize_logger
from src.vadim_ml.io import dump_file, load_file

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

LOGPATH = 'log'
os.makedirs(LOGPATH, exist_ok=True)
LOGGER = initialize_logger(os.path.join(LOGPATH, 'log_flair'), 'flair_crf_conll')

OUTPUT_FILE_PATH = 'experiments/2percent_ELMO_anneal'
SEED_RANDOM = 100


def strategies_to_try(tp):
    if tp == 'uncertainty':
        return lambda trn_ds, libact_model: UncertaintySampling(trn_ds, model=libact_model, method='lc')
    elif tp == 'random':
        return lambda trn_ds, libact_model: RandomSamplingWithRetraining(trn_ds, model=libact_model, method='lc')
    elif tp == 'positivelesscertain':
        return lambda tr_ds, libact_model: UncertaintySampling(tr_ds, model=PositiveLessCertain(libact_model), method='lc')
    else:
        raise ValueError('Wrong strategy')


def y_train2y_seed(y_train, n_seeds_per_class=10):
    y_vals = set(tag for row in y_train for tag in row)

    indexes = np.array(range(len(y_train)))
    known_indexes = []
    for label in y_vals:
        selected_indices = indexes[[label in y for y in y_train]]
        known_indexes += list(np.random.choice(selected_indices, size=n_seeds_per_class))

    known_indexes = list(set(known_indexes))
    tags_seed = [None for _ in range(len(y_train))]

    for i in known_indexes:
        tags_seed[i] = y_train[i]

    Y_seed = tags_seed

    return Y_seed


def emulate_active_learning(
    train_tags,
    active_learner,
    models_path,
    json_path,
    max_iterations=12,
    fit_model=False
):
    active_learner.start()
    LOGGER.info('Start emulating active learning.')

    statistics = []

    perf = active_learner.evaluate(fit_model=False)
    LOGGER.info(f'Performance on seed examples: {perf}')

    statistics.append(perf)

    for i in range(max_iterations):
        try:
            LOGGER.info(f'Active learning iteration: #{i}')

            samples_to_annotate = active_learner.choose_samples_for_annotation()

            annotations = np.array([train_tags[idx] for idx in samples_to_annotate])

            active_learner.make_iteration(samples_to_annotate, annotations)
            perf = active_learner.evaluate(fit_model=False)
            statistics.append(perf)
            dump_file(statistics, models_path, json_path)
            print(f'Dumped iteration {i}')
            LOGGER.info(f'Performance: {perf}')
        except (KeyboardInterrupt, EOFError):
            break
        except RuntimeError:
            print('Check GPU memory!')
            break

    return statistics


def load_task(data_folder):
    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}
    tag_dictionary = Dictionary()
    tag_dictionary.add_item('<START>')
    tag_dictionary.add_item('<STOP>')

    for part in ('train', 'test'):
        dataset = load_file(data_folder, f'{part}.txt')

        for sentence in dataset.split('\n\n'):
            X_sentence = []
            y_sentence = []

            for tagged_token in sentence.split('\n'):
                if not tagged_token:
                    continue
                token, _, _, tag = re.split(' ', tagged_token)
                if not token.startswith("-DOCSTART-"):
                    X_sentence.append(token)
                    y_sentence.append(tag)
                    tag_dictionary.add_item(tag)

            if X_sentence:
                X[part].append(X_sentence)
                y[part].append(y_sentence)

    return X['train'], X['test'], y['train'], y['test'], tag_dictionary


def get_embeddings(emb_name):
    emb_type, emb_name = emb_name.split('+')

    if emb_type == 'elmo':
        return lambda: ELMoEmbeddings(emb_name)  # pubmed
    elif emb_type == 'fasttext':
        return lambda: WordEmbeddings(emb_name)  # en
    else:
        raise ValueError('Wrong embedding type')


def run_experiment(
    data_path,
    models_path,
    ranking_strategy,
    n_al_iterations,
    emb_name,
    max_samples_number,
    n_seeds_random,
    percent=True,
    batch_size=64,
    batch_size_pred=64,
    max_epochs=20,
    learning_rate=0.1,
    repeat=1
):

    models_path = os.path.join(models_path, emb_name, ranking_strategy)
    os.makedirs(models_path, exist_ok=True)

    strat = strategies_to_try(ranking_strategy)
    emb = get_embeddings(emb_name)
    
    print(f'Opened statistics{repeat}.json')
    if os.path.exists(os.path.join(models_path, f'statistics{repeat}.json')):
        print(f'statistics{repeat}.json already exists. Next')
        return

    LOGGER.info(f'Strategy {ranking_strategy} is running')

    X_train, X_test, y_train, y_test, tag_dictionary = load_task(data_path)
    y_seed = y_train2y_seed(y_train, n_seeds_per_class=n_seeds_random)
    
    if percent:
        selector = [False for _ in range(len(y_seed))]
        for ind, answ in enumerate(y_seed):
            if answ is None:
                selector[ind] = False
            elif all(e is None for e in y_seed):
                selector[ind] = False
            else:
                selector[ind] = True
                
        y_nonempty = np.array(y_seed)[selector]
        max_samples_number = int((len(y_seed) - len(y_nonempty)) * 0.02)                   


    tagger = SequenceTagger(
        hidden_size=128,
        embeddings=emb(),
        tag_dictionary=tag_dictionary,
        tag_type='ner',
        use_crf=True
    )

    active_tagger = SequenceTaggerActiveStudent(
        tagger,
        base_path=models_path,
        reset_model_before_train=True,
        mini_batch_size=batch_size,
        eval_mini_batch_size=batch_size_pred,
        checkpoint=False,
        learning_rate=0.1,
        index_subset=False,
        save_all_models=False,
        save_final_model=False,
        anneal_with_restarts=True,
        max_epochs=max_epochs
    )

    active_learn_alg_ctor = make_libact_strategy_ctor(
        lambda tr_ds: strat(tr_ds, active_tagger), max_samples_number=max_samples_number)

    LOGGER.info('Active learning...')

    active_learner = ActiveLearner(
        active_learn_alg_ctor=active_learn_alg_ctor,
        y_dtype='str',
        X_full_dataset=X_train,
        y_full_dataset=y_seed,
        X_test_dataset=X_test,
        y_test_dataset=y_test,
        model_evaluate=active_tagger,
        eval_metrics=[f1_score],
        rnd_start_steps=0,
        rnd_start_samples=max_samples_number
    )

    statistics = emulate_active_learning(y_train, active_learner, models_path, f'statistics{repeat}.json', max_iterations=n_al_iterations)
    dump_file(statistics, models_path, f'statistics{repeat}.json')

    print(f'Experiment {repeat} ended')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join('data', 'conll2003'),
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['uncertainty', 'random', 'positivelesscertain'],
        default='uncertainty',
    )
    
    parser.add_argument(
        '--percent',
        type=bool,
        default=True,
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
    )

    parser.add_argument(
        '--batch_size_pred',
        type=int,
        default=512,
    )

    parser.add_argument(
        '--n_samples_per_al_iter',
        type=int,
        default=100
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=20
    )

    parser.add_argument(
        '--n_experiments',
        type=int,
        default=5
    )

    parser.add_argument(
        '--n_al_iterations',
        type=int,
        default=50
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--embedding',
        type=str,
        choices=['elmo+small', 'elmo+medium', 'fasttext+en'],
        default='elmo+medium'
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        default=5
    )

    arguments = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    flair.device = device

    multiprocessing.set_start_method('spawn')

    processes = []

    n_processes = arguments.n_processes or 1

    starttime = time.time()
    i = 0
    while i < arguments.n_experiments:
        p = multiprocessing.Process(
            target=run_experiment,
            args=(
                arguments.data_path,
                OUTPUT_FILE_PATH,
                arguments.strategy,
                arguments.n_al_iterations,
                arguments.embedding,
                arguments.n_samples_per_al_iter,
                SEED_RANDOM,
                arguments.percent,
                arguments.batch_size,
                arguments.batch_size_pred,
                arguments.max_epochs,
                arguments.learning_rate,
                i
            )
        )
        processes.append(p)
        print(processes)
        p.start()
        i += 1
        if not (i % n_processes) and n_processes > 0:
            for process in processes:
                process.join()
            processes = []

    for process in processes:
        process.join()
    processes = []

    print(f'Overall time for {arguments.n_experiments} iterations took {time.time() - starttime} seconds')
    