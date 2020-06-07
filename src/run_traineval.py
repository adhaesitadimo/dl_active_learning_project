import flair
from flair.embeddings import WordEmbeddings, ELMoEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
#from flair.trainers import ModelTrainer
from al4ner.trainer_flair import ModelTrainer as ModelTrainerFlair
from flair.training_utils import EvaluationMetric
from flair.data import Dictionary
from flair.datasets import ColumnCorpus    

from pathlib import Path
import itertools
import os
import torch
import gc
import hydra

from al4ner.exp_utils import tokenize_row, entity_level_f1, train_dev_test_distr

from vadim_ml.io import dump_file

from pytorch_transformers import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule

from bert_sequence_tagger.bert_utils import create_loader_from_flair_corpus, get_parameters_without_decay
from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus, get_model_parameters
from bert_sequence_tagger.bert_utils import get_model_parameters, prepare_flair_corpus
from bert_sequence_tagger.model_trainer_bert import ModelTrainerBert
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level
from bert_sequence_tagger.bert_for_token_classification_custom import BertForTokenClassificationCustom
from bert_sequence_tagger import SequenceTaggerBert

import torch
torch.manual_seed(0)
from torch.optim.lr_scheduler import ReduceLROnPlateau


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_corpus(data_folder, tag_column, attr, downsample_perc):
    corpus = ColumnCorpus(Path(data_folder) / attr, 
                        {0: 'text', tag_column: 'ner'},
                        train_file='train.txt',
                        test_file='test.txt',
                        dev_file='dev.txt') 

    if downsample_perc > 0:
        print('Downsampling: ', downsample_perc)
        corpus = corpus.downsample(percentage=downsample_perc, only_downsample_train=True)
        
    return corpus

    
def train_eval_bert(corpus, res_dir, model_name, cfg_model):
    idx2tag, tag2idx = make_bert_tag_dict_from_flair_corpus(corpus)
    model = BertForTokenClassificationCustom.from_pretrained(model_name, 
                                                             cache_dir=cfg_model.cache_dir, 
                                                             num_labels=len(tag2idx))
    
    tokenizer =  BertTokenizer.from_pretrained(cfg_model.tokenizer, 
                                               cache_dir=cfg_model.cache_dir, 
                                               do_lower_case=('uncasaed' in cfg_model.tokenizer))

    w_decay = 0.01
    model = model.cuda()

    seq_tagger = SequenceTaggerBert(bert_model=model, bpe_tokenizer=tokenizer, 
                                    idx2tag=idx2tag, tag2idx=tag2idx, max_len=cfg_model.max_len)
    
    train_dataset = prepare_flair_corpus(corpus.train)
    val_dataset = prepare_flair_corpus(corpus.dev)
    test_dataset = prepare_flair_corpus(corpus.test)
 
    optimizer = AdamW(get_model_parameters(model), 
                      lr=cfg_model.lr, betas=(0.9, 0.999), 
                      eps=1e-6, weight_decay=w_decay, correct_bias=True) 
    
    if cfg_model.sched == 'warmup':
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0.1, 
                                            t_total=(len(corpus.train) / cfg_model.bs)*cfg_model.n_epochs)
        trainer = ModelTrainerBert(model=seq_tagger, 
                                   optimizer=optimizer, 
                                   lr_scheduler=lr_scheduler,
                                   train_dataset=train_dataset, 
                                   val_dataset=val_dataset,
                                   update_scheduler='es',
                                   validation_metrics=[f1_entity_level],
                                   batch_size=cfg_model.bs)
    elif cfg_model.sched == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg_model.patience, mode='min')
        trainer = ModelTrainerBert(model=seq_tagger, 
                                   optimizer=optimizer, 
                                   lr_scheduler=lr_scheduler,
                                   train_dataset=train_dataset, 
                                   val_dataset=val_dataset,
                                   keep_best_model=True,
                                   restore_bm_on_lr_change=True,
                                   update_scheduler='ee',
                                   validation_metrics=[f1_entity_level],
                                   decision_metric=lambda metrics: -metrics[1],
                                   batch_size=cfg_model.bs)
        
    trainer.train(epochs=cfg_model.n_epochs)
    
    _, __, metrics = seq_tagger.predict(test_dataset, evaluate=True, 
                                        metrics=[f1_entity_level, f1_token_level])
    print('Test performance:', metrics)
    
    return seq_tagger, metrics
    

def create_flair_embeddings(emb_name):
    emb_type, emb_subname = emb_name.split('+')
    if emb_type == 'elmo':
        return ELMoEmbeddings(emb_subname)
    elif emb_type == 'fasttext':
        return WordEmbeddings(emb_subname)
    elif emb_type == 'custom_elmo':
        return ELMoEmbeddings(options_file=Path(emb_subname) / 'options.json', weight_file=Path(emb_subname) / 'model.hdf5')
    #'../../../data/BioWordVec_PubMed_MIMICIII_d200.kv'

    
def train_eval_flair(corpus, res_dir, model_name, cfg_model):
    embeddings = create_embeddings(cfg_model.emb_name)

    tagger = SequenceTagger(hidden_size=cfg_model.hidden_size,
                           embeddings=embeddings,
                           tag_dictionary=corpus.make_tag_dictionary('ner'),
                           tag_type='ner',
                           use_crf=True)

    trainer = ModelTrainerFlair(tagger, corpus)
    trainer.train(base_path=folder,
                  learning_rate=cfg_model.lr,
                  mini_batch_size=cfg_model.bs,
                  mini_batch_chunk_size=cfg_model.ebs,
                  max_epochs=cfg_model.n_epochs,
                  checkpoint=False)
    
    entity_level_f1(res_dir)

    return tagger
    
    
def train_eval_tagger(corpus, res_dir, cfg_model):
    model_type = cfg_model.model_type.split('+')
    
    if model_type[0] == 'flair':
        return train_eval_flair(corpus, res_dir, model_type[1], cfg_model)
    
    elif model_type[0] == 'bert':
        return train_eval_bert(corpus, res_dir, model_type[1], cfg_model)
    
        
def run_task(task):
    auto_generated_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    
    result_path = auto_generated_dir
    pathology = task.data.task
    data_folder = task.data.data_folder
    
    corpus = load_corpus(data_folder, task.data.tag_column, 
                         pathology, task.data.downsample_perc)
    
    for i in range(task.n_repeats):
        print(f'Repeating: #{i}')
        
        model_folder = f'{result_path}/{i}'
        
        tagger, scores = train_eval_tagger(corpus, model_folder, cfg_model=task.model)
        
        dump_file(scores, model_folder, 'score.json')
        
        print('GC collect')
        gc.collect()
        torch.cuda.empty_cache() 


@hydra.main(config_path=os.environ['HYDRA_CONFIG_PATH'])
def main(configs):
    run_task(configs)

    
if __name__ == "__main__":
    main()
