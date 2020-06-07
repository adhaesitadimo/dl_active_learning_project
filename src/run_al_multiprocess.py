import flair
from flair.embeddings import WordEmbeddings, ELMoEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from flair.data import Dictionary

import time
import hydra
import pandas as pd
import numpy as np
import itertools
import multiprocessing
import os
import torch
from seqeval.metrics import f1_score
from pathlib import Path
import nltk

from al4ner.libact_flair import LibActFlair, PositiveLessCertain, LibActFlairBayes
from al4ner.libact_crf import LibActCrf
from al4ner.libact_nn import LibActNN

from libact.query_strategies import UncertaintySampling, RandomSampling
from active_learning_seq import RandomSamplingWithRetraining
from actleto import ActiveLearner, make_libact_strategy_ctor

from vadim_ml.memoize import memoize
from vadim_ml.io import load_file, dump_file

from flair.datasets import ColumnCorpus, ColumnDataset
from torch import nn

from al4ner.mc_dropout import convert_to_mc_dropout, DropoutMC
from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom, ModelTrainerBert
from bert_sequence_tagger.bert_utils import get_parameters_without_decay
from bert_sequence_tagger.metrics import f1_entity_level

from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch.optim.lr_scheduler import ReduceLROnPlateau


#MIN_LEARNING_RATE = LEARNING_RATE / (2**4)

BERT_LEARNING_RATE = 5e-5
VALIDATION_RATIO = 0.25
ANNEAL_FACTOR = 0.5
PATIENCE = 2
MAX_TO_ANNEAL = 3


def load_task(data_folder, task, tag_column, preprocess):
    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}
    tag_dictionary = Dictionary()
    tag_dictionary.add_item('<START>')
    tag_dictionary.add_item('<STOP>')

    for part in ('train', 'test'):
        #dataset = load_file(data_folder, task, f'{part}.txt')

        file_path = Path(f'{data_folder}/{task}/{part}.txt')
        print('Loading: ', file_path)

        corpus = ColumnDataset(
            path_to_column_file=file_path,
            column_name_map={0: 'text', tag_column: 'ner'},
            tag_to_bioes=None,
            encoding='utf8',
            comment_symbol=None,
            in_memory=True,
            document_separator_token=None,
        )

        for sent in corpus:
            tokens = [w.text for w in sent]
            if preprocess:
                X[part].append(list(zip(tokens, [nltk.pos_tag([tok])[0][1] for tok in tokens])))
            else:
                X[part].append(tokens)

            labels = [w.get_tag('ner').value for w in sent]
            y[part].append(labels)

            for tag in labels:
                tag_dictionary.add_item(tag)

    print('Train size:', len(X['train']))
    print('Test size:', len(X['test']))

    return X['train'], X['test'], y['train'], y['test'], tag_dictionary


def strategies_to_try(tp):
    if tp == 'uncertainty':
        return lambda trn_ds, libact_model: UncertaintySampling(trn_ds, model=libact_model, method='lc')
    elif tp == 'random':
        return lambda trn_ds, libact_model: RandomSamplingWithRetraining(trn_ds, model=libact_model, method='lc')
    elif tp == 'positivelesscertain':
        return lambda tr_ds, libact_model: UncertaintySampling(tr_ds, model=PositiveLessCertain(libact_model), method='lc')
    else:
        raise ValueError('Wrong strategy')


def y_train2y_seed(y_train, n_seeds_per_class=100):
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


def y_train2y_seed_percent(y_train, percent=0.02):
    y_vals = set(tag for row in y_train for tag in row)

    indexes = np.array(range(len(y_train)))
    known_indexes = []
    selected_indices = []
    
    #for label in y_vals:
    #    selected_indices += indexes[[label in y for y in y_train]]
        
    print(len(indexes))
    print(int(len(y_train) * 0.02))
        
    known_indexes += list(np.random.choice(indexes, size=int(len(y_train) * 0.02), replace=False))

    known_indexes = list(set(known_indexes))

    tags_seed = [None for _ in range(len(y_train))]

    for i in known_indexes:
        tags_seed[i] = y_train[i]

    Y_seed = tags_seed

    return Y_seed



def emulate_active_learning(train_tags, active_learner,
                            max_iterations=12,
                            fit_model=False):
    print('Start emulating active learning.')
    active_learner.start()
    print('Evaluation:', active_learner.evaluate(fit_model=False))

    statistics = []
    for i in range(max_iterations):
        try:
            print('Active learning iteration: #{}'.format(i))
            samples_to_annotate = active_learner.choose_samples_for_annotation()
            annotations = np.array([train_tags[idx] for idx in samples_to_annotate])
            active_learner.make_iteration(samples_to_annotate, annotations)
            perf = active_learner.evaluate(fit_model=fit_model)
            statistics.append(perf)
            print('Performance:', perf)
        except (KeyboardInterrupt, EOFError):
            print('Stopping early. Saving partial results')
            break
        except RuntimeError:
            print('Check (V)RAM! Saving partial results')
            break

    return statistics


def get_embeddings(emb_name):
    emb_type, emb_name = emb_name.split('+')

    if emb_type == 'elmo':
        return lambda: ELMoEmbeddings(emb_name)  # pubmed
    elif emb_type == 'fasttext':
        return lambda: WordEmbeddings(emb_name)  # en
    elif emb_type == 'flair':
        return lambda: FlairEmbeddings(emb_name) # news-forward-fast

    else:
        raise ValueError('Wrong embedding type')


def create_libact_adaptor_bert(tag2index, index2tag, adaptor_type, *args, **kwargs):
    def model_ctor():
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                       cache_dir=kwargs['config'].model.cache_dir,
                                                       do_lower_case=False)
        # BIO_BERT
        model = BertForTokenClassificationCustom.from_pretrained('bert-base-cased',
                                                                 cache_dir=kwargs['config'].model.cache_dir,
                                                                 num_labels=len(tag2index)).cuda()

        if config.model.bayes:
            if config.model.bayes_type == 'mcd_last_layer':
                # last dropout layer name in BertForTokenClassification
                model.dropout = DropoutMC(p=model.dropout.p, activate=True)
            elif config.model.bayes_type == 'mcd_all_layers':
                convert_to_mc_dropout(model)

        seq_tagger = SequenceTaggerBert(model, bert_tokenizer, idx2tag=index2tag, tag2idx=tag2index)
        return seq_tagger

    def trainer_ctor(seq_tagger, corpus_len, train_dataloader, val_dataloader):
        optimizer = AdamW(get_parameters_without_decay(seq_tagger._bert_model),
                          lr=BERT_LEARNING_RATE, betas=(0.9, 0.999),
                          eps=1e-6, weight_decay=0.01, correct_bias=True)

        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=ANNEAL_FACTOR, patience=PATIENCE)

        trainer = ModelTrainerBert(seq_tagger,
                                   optimizer,
                                   lr_scheduler,
                                   train_dataloader,
                                   val_dataloader,
                                   update_scheduler='ee',
                                   keep_best_model=True,
                                   restore_bm_on_lr_change=True,
                                   max_grad_norm=1.,
                                   validation_metrics=[f1_entity_level],
                                   decision_metric=lambda metrics: metrics[0],
                                   smallest_lr=config.model.smallest_lr)
        # BERT_LEARNING_RATE / (MAX_TO_ANNEAL**(1./ANNEAL_FACTOR) + 0.1)

        return trainer

    return adaptor_type(*args,
                        model_ctor=model_ctor,
                        trainer_ctor=trainer_ctor,
                        batch_size=kwargs['config'].model.bs,
                        bs_pred=kwargs['config'].model.ebs,
                        train_from_scratch=True,
                        retrain_epochs=kwargs['config'].model.n_epochs,
                        valid_ratio=VALIDATION_RATIO,
                        string_input=False)


def run_experiment(config, repeat=1):
    print('Active learning strategy:', config.al.strat_name)

    print('Loading task...', config.data.task)
    preprocess = (config.model.model_type == 'crf')
    print(config.data.data_folder)
    X_train, X_test, y_train, y_test, tag_dictionary = load_task(config.data.data_folder, 
                                                                 config.data.task, 
                                                                 config.data.tag_column,
                                                                 preprocess)
    print('Done.')

    strat = strategies_to_try(config.al.strat_name)

    model_name = config.model.model_type


    if config.al.percent:
        print('FULL:', len(y_train))
        y_seed = y_train2y_seed_percent(y_train)
        selector = [False for _ in range(len(y_seed))]
        for ind, answ in enumerate(y_seed):
            if answ is None:
                selector[ind] = False
            elif all(e is None for e in y_seed):
                selector[ind] = False
            else:
                selector[ind] = True

        y_nonempty = np.array(y_seed)[selector]
        print('2PERCENT:', len(y_nonempty))
        max_samples_number = int(len(y_seed) * 0.02)

    else:
        y_seed = y_train2y_seed(y_train)
        max_samples_number = config.al.max_samples_number

    print('MAX_SAMPLES:', max_samples_number)

    if 'flair' in config.model.model_type:
        print(config.model.model_type)

        bayes_type = config.model.bayes_type if config.model.bayes else 'no_bayes'
        models_path = os.path.join(config.exp_path, f'{model_name}_{config.model.emb_name}_{bayes_type}/{config.al.strat_name}')
        print(models_path)
        os.makedirs(models_path, exist_ok=True)

        if os.path.exists(os.path.join(models_path, f'statistics{repeat}.json')):
            print(f'statistics{repeat}.json already exists. Next')
            return

        print('Embeddings', config.model.emb_name)
        emb = get_embeddings(config.model.emb_name)

        tagger = SequenceTagger(hidden_size=config.model.hidden_size,
                                embeddings=emb(),
                                tag_dictionary=tag_dictionary,
                                tag_type=config.data.task,
                                use_crf=True)
        print(config.model.bayes)
        if config.model.bayes:
            convert_to_mc_dropout(tagger, (nn.Dropout, flair.nn.WordDropout, flair.nn.LockedDropout), option='flair')
            active_tagger = LibActFlairBayes(tagger,
                                        base_path=models_path,
                                        reset_model_before_train=True,
                                        mini_batch_size=config.model.bs,
                                        eval_mini_batch_size=config.model.ebs,
                                        checkpoint=False,
                                        learning_rate=config.model.lr,
                                        index_subset=False,
                                        save_all_models=False,
                                        max_epochs=config.model.n_epochs,
                                        min_learning_rate=config.model.min_lr)


        else:
            active_tagger = LibActFlair(tagger,
                                        base_path=models_path,
                                        reset_model_before_train=True,
                                        mini_batch_size=config.model.bs,
                                        eval_mini_batch_size=config.model.ebs,
                                        checkpoint=False,
                                        learning_rate=config.model.lr,
                                        index_subset=False,
                                        save_all_models=False,
                                        max_epochs=config.model.n_epochs,
                                        min_learning_rate=config.model.min_lr)
        fit_model = False

    elif config.model.model_type == 'crf':
        models_path = os.path.join(config.exp_path, model_name)
        print(models_path)
        if os.path.exists(os.path.join(models_path, f'statistics{repeat}.json')):
            print(f'statistics{repeat}.json already exists. Next')
            return

        active_tagger = LibActCrf(algorithm="lbfgs",
                                  c1=0.1,
                                  c2=0.1,
                                  max_iterations=100,
                                  all_possible_transitions=True)
        fit_model = True

    elif config.model.model_type == 'bert':

        bayes_type = config.model.bayes_type if config.model.bayes else 'no_bayes'
        models_path = os.path.join(config.exp_path, f'{model_name}_{bayes_type}')

        if os.path.exists(os.path.join(models_path, f'statistics{repeat}.json')):
            print(f'statistics{repeat}.json already exists. Next')
            return

        index2tag = ['[PAD]'] + tag_dictionary.get_items()
        tag2index = {e: i for i, e in enumerate(index2tag)}
        active_tagger = create_libact_adaptor_bert(tag2index, index2tag, LibActNN, config=config)
        fit_model = False

    active_learn_alg_ctor = make_libact_strategy_ctor(lambda tr_ds: strat(
        tr_ds, active_tagger), max_samples_number=max_samples_number)

    active_learner = ActiveLearner(active_learn_alg_ctor=active_learn_alg_ctor,
                                   y_dtype='str',
                                   X_full_dataset=X_train,
                                   y_full_dataset=y_seed,
                                   X_test_dataset=X_test,
                                   y_test_dataset=y_test,
                                   model_evaluate=active_tagger,
                                   eval_metrics=[f1_score],
                                   rnd_start_steps=0)
    print(models_path)
    statistics = emulate_active_learning(y_train, active_learner,
                                         max_iterations=config.al.n_iterations,
                                         fit_model=fit_model)
    dump_file(statistics, models_path, f'statistics{repeat}.json')


@hydra.main(config_path=os.environ['HYDRA_CONFIG_PATH'])
def main(config):
    auto_generated_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    
    multiprocessing.set_start_method('spawn')

    processes = []
    n_processes = config.n_processes 

    starttime = time.time()
    i = 0
    while i < config.n_repeats:
        p = multiprocessing.Process(
            target=run_experiment,
            args=(
                config,
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

    print(f'Overall time for {config.n_repeats} repeats took {time.time() - starttime} seconds')


if __name__ == "__main__":
    main()
