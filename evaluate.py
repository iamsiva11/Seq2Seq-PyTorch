"""Evaluation utils."""
import sys

sys.path.append('/u/subramas/Research/nmt-pytorch')

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import get_minibatch
from collections import Counter
import math
import numpy as np
import subprocess
import sys


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", 'multi-bleu.perl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()


def compute_accuracy(preds, ground_truths):
    """Compute prediction accuracy."""
    equal = 0.
    for pred, gold in zip(preds, ground_truths):
        pred = ' '.join(pred)
        gold = ' '.join(gold)
        if pred == gold:
            equal += 1

    return (equal / len(preds)) * 100


def evaluate_model(
    model, src, src_test, trg,
    trg_test, config, src_valid=None, trg_valid=None,
    verbose=True, metric='accuracy'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in xrange(0, len(src_test['data']), config['data']['batch_size']):

        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = get_minibatch(
            trg_test['data'], trg['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        input_lines_trg = Variable(torch.LongTensor(
            [
                [trg['word2id']['<s>']]
                for i in xrange(input_lines_src.size(0))
            ]
        )).cuda()

        for i in xrange(config['data']['max_src_length']):

            if config['model']['seq2seq'] == 'vanilla_crf':
                mask = Variable(torch.ones(
                    input_lines_trg.size(1), input_lines_trg.size(0)
                )).cuda()
                decoder_logit_rnn, decoder_logit_crf = model(
                    input_lines_src, input_lines_trg, trg_mask=mask
                )
                if config['model']['decode'] == 'greedy':
                    word_probs = model.decode(decoder_logit_crf)
                else:
                    decoder_argmax = model.viterbi_decode(decoder_logit_crf)
            else:
                decoder_logit = model(input_lines_src, input_lines_trg)
                word_probs = model.decode(decoder_logit)
            if config['model']['decode'] == 'greedy':
                decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )
        if (
            config['model']['seq2seq'] == 'vanilla_crf' and
            config['model']['decode'] == 'viterbi'
        ):
            input_lines_trg = model.viterbi_decode(decoder_logit_crf)

        if not config['model']['decode'] == 'viterbi':
            input_lines_trg = input_lines_trg.data.cpu().numpy()

        input_lines_trg = [
            [trg['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
        output_lines_trg_gold = [
            [trg['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        for sentence_pred, sentence_real, sentence_real_src in zip(
            input_lines_trg,
            output_lines_trg_gold,
            output_lines_src
        ):
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
            else:
                index = len(sentence_pred)
            preds.append(['<s>'] + sentence_pred[:index + 1])

            if verbose:
                print ' '.join(['<s>'] + sentence_pred[:index + 1])

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print ' '.join(['<s>'] + sentence_real[:index + 1])
            if verbose:
                print '--------------------------------------'
            ground_truths.append(['<s>'] + sentence_real[:index + 1])

            if '</s>' in sentence_real_src:
                index = sentence_real_src.index('</s>')
            else:
                index = len(sentence_real_src)

    if metric == 'accuracy':
        return compute_accuracy(preds, ground_truths)
    else:
        return get_bleu(preds, ground_truths)