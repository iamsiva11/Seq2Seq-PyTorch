#!/u/subramas/miniconda2/bin/python
"""Data processing utilities."""
import sys

sys.path.append('/u/subramas/Research/nmt-pytorch/')

from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2Seq, Seq2SeqAttention
from evaluate import evaluate_accuracy
import math
import numpy as np
import logging
import argparse

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

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

src, trg = read_nmt_data(
    src=config['data']['src'],
    trg=config['data']['trg']
)

src_test, trg_test = read_nmt_data(
    src=config['data']['valid_src'],
    trg=config['data']['valid_trg']
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)

model = Seq2Seq(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    nlayers=config['model']['n_layers_src'],
    dropout=0.,
    peek_dim=0
).cuda()

'''
model = Seq2SeqAttention(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    ctx_hidden_dim=config['model']['dim'],
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    nlayers=config['model']['n_layers_src'],
    dropout=0.,
    peek_dim=0
).cuda()
'''


def clip_gradient(model, clip):
    """Compute a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

# import ipdb
# ipdb.set_trace()
optimizer = optim.Adam(model.parameters(), lr=4e-4)

for i in xrange(1000):
    losses = []
    for j in xrange(0, len(src['data']), batch_size):

        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src['data'], src['word2id'], j, batch_size, max_length, add_start=True, add_end=True
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg['data'], trg['word2id'], j, batch_size, max_length, add_start=True, add_end=True
        )

        if input_lines_src.size()[0] != batch_size:
            break

        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()
        loss = loss_criterion(decoder_logit.view(-1, trg_vocab_size), output_lines_trg.view(-1))
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (i, j, np.mean(losses)))
            losses = []
        '''
        if j % 10000 == 0:
            word_probs = model.decode(decoder_logit).data.cpu().numpy().argmax(axis=-1)
            output_lines_trg = output_lines_trg.data.cpu().numpy()
            for sentence_pred, sentence_real in zip(word_probs[:5], output_lines_trg[:5]):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                index = sentence_real.index('</s>')
                sentence_real = sentence_real[:index]
                sentence_pred = sentence_pred[:index]

                logging.info('---------------------------------------------------')
                logging.info(' '.join(sentence_pred))
                logging.info(' '.join(sentence_real))
                logging.info('---------------------------------------------------')
        '''
    accuracy = evaluate_accuracy(model, src, src_test, trg, trg_test, config, verbose=False)
    logging.info('Epoch : %d Accuracy : %.5f ' % (i, accuracy))