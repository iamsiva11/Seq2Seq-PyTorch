import sys


from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAttention, Seq2SeqFastAttention
from evaluate import evaluate_model
import math
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)

args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
#load_dir = config['data']['load_dir']


# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print 'Reading data ...'

src, trg, srcf3, srcf5 = read_nmt_data(
    src=config['data']['src'],
    config=config,
    src_f3 = config['data']['f3_src'],
    src_f5 = config['data']['f5_src'],
    trg=config['data']['trg']
)

src_test, trg_test, srcf3_test, srcf5_test  = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    src_f3 = config['data']['test_f3_src'],
    src_f5 = config['data']['test_f5_src'],
    trg=config['data']['test_trg']
)


batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
#src_vocab_size = len(src['word2id'])
src_vocab_size = len(src['word2id']) + len(srcf3['word2id']) + len(srcf5['word2id'])

trg_vocab_size = len(trg['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()


model = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=config['model']['bidirectional'],
        pad_token_src=src['word2id']['<pad>'],
        pad_token_trg=trg['word2id']['<pad>'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.,
    ).cuda()


# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")


# Load the Model

load_dir = "/home/siva/seq2seq-pytorch/features-seq2seq/data/MODELS-NEW/"
#model_file_name = "model_translation__src_fr__trg_en__attention_attention__dim_1000__emb_dim_500__optimizer_adam__n_layers_src_2__n_layers_trg_1__bidir_True__epoch_100.model"
model_file_name = "model_translation__src_fr__trg_en__attention_attention__dim_1000__emb_dim_500__optimizer_adam__n_layers_src_2__n_layers_trg_1__bidir_True__epoch_300.model"
#model_file_name = "model_translation__src_fr__trg_en__attention_attention__dim_1000__emb_dim_500__optimizer_adam__n_layers_src_2__n_layers_trg_1__bidir_True__epoch_200.model"

model.load_state_dict(torch.load(
 open(os.path.join(
            load_dir,
            model_file_name), 'rb'
        )

	))


# Evaluate Model

print 'Evaluating model..\n'

bleu = evaluate_model(
        model, src, src_test, trg,
        trg_test, 
	srcf3, srcf3_test, srcf5 ,srcf5_test, config, verbose=False,
        metric='bleu',
        )

print bleu