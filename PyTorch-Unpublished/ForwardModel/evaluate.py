"""Evaluation utils."""
import sys

#sys.path.append('/mnt/sdb1/Dropbox (GaTech)/Chicken/SciFiEventExperiments/SciFi-GenEvent-GenEvent-Split')

import torch
import torch.nn.functional as F
#from torch.autograd import Variable
#from data_utils import get_minibatch, get_autoencode_minibatch
from data_utils import read_bucket_data, get_story_batch, init_state, process_batch
from collections import Counter, defaultdict
import math
import numpy as np
import subprocess
import sys, statistics


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(0, len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(0, len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
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


def decode_minibatch(
    config,
    model,
    input_lines_src,
    input_lines_trg,
    output_lines_trg_gold
):
    """Decode a minibatch."""
    for i in range(0, config['data']['max_trg_length']):

        decoder_logit = model(input_lines_src, input_lines_trg)
        word_probs = model.decode(decoder_logit)
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        next_preds = Variable(
            torch.from_numpy(decoder_argmax[:, -1])
        ).cuda()

        input_lines_trg = torch.cat(
            (input_lines_trg, next_preds.unsqueeze(1)),
            1
        )

    return input_lines_trg


def model_perplexity_prebatched(model, src_test, word2ind, ind2word, config, unigram_probs, end_token = "%%%%%%%%%%", add_start=True, add_end=True, not_hidden=False):

    h = None
    c = None
    p = 0
    first = True
   
    for batch_key in list(src_test.keys()):
        batch = src_test[batch_key]
        input_lines_src, output_lines_src, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2ind, add_start=True, add_end=True)

        if batch["restart"] == True or not_hidden:
            h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
        
        decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)

        predictions = model.decode(decoder_logit).data.cpu().numpy()

        h.detach_()
        c.detach_()

        top_words = predictions.argmax(axis=-1)        
        #top_words = top_words.transpose()
        #word_probs = predictions.max(axis=-1)
        word_probs = list(predictions)#[-1]
        #print("WORD PROBS", word_probs)
        #print("INPUT TRG",input_lines_trg)
        #print("TOP WORDS", top_words)

        for i, event in enumerate(batch['trg']): #for each event in the validation set
            #if add_end:
            prob_event = list(word_probs[i])
            for j, word in enumerate(event):                
                prob_word = list(prob_event[j])
                p+=math.log(prob_word[word2ind[word]],2)*unigram_probs[word]
            #p = p/len(sentence_pred)

            first = True
            if first: #only print the predictions for the first batch (just for a sample)
                top_preds = [ind2word[x] for x in top_words[i][:-1]]
                print('Predicted : %s ' % (' '.join(top_preds)))
                print('Real : %s ' % (' '.join(event)))
                print('===============================================')
                first = False
    try:
        #print(len(list(src_test.keys())))
        p = p/len(list(src_test.keys()))
        #print(2**-x)
        #print((2**-p)/len(list(src_test.keys())))
        final = 2 ** -p
    except ValueError:
        final = 150

    return final




def getPerplexityBLEU_forBatch(predictions, target, word2ind, ind2word, unigram_probs, p):
    top_words = predictions.argmax(axis=-1)
    word_probs = list(predictions)#[-1]

    BLEUs = []
    preds = []
    reals = []

    for i, event in enumerate(target): #for each event in the validation set
        #print(event)
        prob_event = list(word_probs[i])
        for j, word in enumerate(event):            
            prob_word = list(prob_event[j])
            #print(prob_word[word])
            p+=math.log(prob_word[word],2)*unigram_probs[ind2word[word]]

        word_event = [ind2word[x] for x in event][1:]
        top_preds = [ind2word[x] for x in top_words[i][1:]]
        print("Sampled event", word_event)
        print("Prediction",top_preds)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #top_preds = [x for x in top_words[i][:-1]]
        bleu = get_bleu(top_preds, word_event)
        BLEUs.append(bleu)

        preds.append(top_preds)
        reals.append(event)
    return preds, reals, p, BLEUs


def getPerplexityBLEU_prebatched(model, src_test, word2ind, ind2word, config, unigram_probs, end_token = "%%%%%%%%%%", add_start=True, add_end=True):

    h = None
    c = None
    p = 0
    first = True
    BLEUs = []
    preds = []
    reals = []
   
    for batch_key in list(src_test.keys()):
        batch = src_test[batch_key]
        input_lines_src, output_lines_src, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2ind, add_start=True, add_end=True)

        if batch["restart"] == True:
            h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
        
        decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)

        predictions = model.decode(decoder_logit).data.cpu().numpy()

        h.detach_()
        c.detach_()

        top_words = predictions.argmax(axis=-1)        
        #top_words = top_words.transpose()
        #word_probs = predictions.max(axis=-1)
        word_probs = list(predictions)#[-1]
        #print("WORD PROBS", word_probs)
        #print("INPUT TRG",input_lines_trg)
        #print("TOP WORDS", top_words)

        

        for i, event in enumerate(batch['trg']): #for each event in the validation set
            #if add_end:
            
            for j, word in enumerate(event):
                prob_event = list(word_probs[i])
                prob_word = list(prob_event[j])
                p+=math.log(prob_word[word2ind[word]],2)*unigram_probs[word]

            top_preds = [ind2word[x] for x in top_words[i][:-1]]
            bleu = get_bleu(top_preds, event)
            BLEUs.append(bleu)

            preds.append(top_preds)
            reals.append(event)

            """
            if first: #only print the predictions for the first batch (just for a sample)
                print('Predicted : %s ' % (' '.join(top_preds)))
                print('Real : %s ' % (' '.join(event)))
                print('===============================================')
                first = False
            """
    #print(len(list(src_test.keys())))
    p = p/len(list(src_test.keys()))
    #print(2**-x)
    #print((2**-p)/len(list(src_test.keys())))

    return preds, reals, 2 ** -p, BLEUs



def getPerplexityBLEU_prebatched_orig(model, src_test, word2ind, ind2word, config, unigram_probs, end_token = "%%%%%%%%%%", add_start=True, add_end=True):

    h = None
    c = None
    p = 0
    first = True
    BLEUs = []
    preds = []
    reals = []
   
    for batch_key in list(src_test.keys()):
        batch = src_test[batch_key]
        input_lines_src, output_lines_src, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2ind, add_start=True, add_end=True)

        h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
        
        decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)

        predictions = model.decode(decoder_logit).data.cpu().numpy()

        h.detach_()
        c.detach_()

        top_words = predictions.argmax(axis=-1)        
        #top_words = top_words.transpose()
        #word_probs = predictions.max(axis=-1)
        word_probs = list(predictions)#[-1]
        #print("WORD PROBS", word_probs)
        #print("INPUT TRG",input_lines_trg)
        #print("TOP WORDS", top_words)


        for i, event in enumerate(batch['trg']): #for each event in the validation set
            for j, word in enumerate(event):
                prob_event = list(word_probs[i])
                prob_word = list(prob_event[j])
                p+=math.log(prob_word[word2ind[word]],2)*unigram_probs[word]

            top_preds = [ind2word[x] for x in top_words[i][:-1]]
            bleu = get_bleu(top_preds, event)
            BLEUs.append(bleu)

            preds.append(top_preds)
            reals.append(event)

    p = p/len(list(src_test.keys()))

    return preds, reals, 2 ** -p, BLEUs



def evaluate_model(
    model, src, src_test, trg,
    trg_test, config, src_valid=None, trg_valid=None,
    verbose=True, metric='bleu'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in range(0, len(src_test['data']), config['data']['batch_size']):

        # Get source minibatch
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        # Get target minibatch
        input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
            get_minibatch(
                trg_test['data'], trg['word2id'], j,
                config['data']['batch_size'], config['data']['max_trg_length'],
                add_start=True, add_end=True
            )
        )

        # Initialize target with <s> for every sentence
        input_lines_trg = Variable(torch.LongTensor(
            [
                [trg['word2id']['<s>']]
                for i in range(0, input_lines_src.size(0))
            ]
        )).cuda()

        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_trg = decode_minibatch(
            config, model, input_lines_src,
            input_lines_trg, output_lines_trg_gold
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_trg = input_lines_trg.data.cpu().numpy()
        input_lines_trg = [
            [trg['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        # Do the same for gold sentences
        output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
        output_lines_trg_gold = [
            [trg['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        # Process outputs
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
                print(' '.join(['<s>'] + sentence_pred[:index + 1]))

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print(' '.join(['<s>'] + sentence_real[:index + 1]))
            if verbose:
                print('--------------------------------------')
            ground_truths.append(['<s>'] + sentence_real[:index + 1])

    return get_bleu(preds, ground_truths)


def evaluate_autoencode_model(
    model, src, src_test,
    config, src_valid=None,
    verbose=True, metric='bleu'
):
    """Evaluate model."""
    preds = []
    ground_truths = []
    for j in range(0, len(src_test['data']), config['data']['batch_size']):

        print('Decoding batch : %d out of %d ' % (j, len(src_test['data'])))
        input_lines_src, lens_src, mask_src = get_autoencode_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )

        input_lines_trg = Variable(torch.LongTensor(
            [
                [src['word2id']['<s>']]
                for i in range(0, input_lines_src.size(0))
            ]
        )).cuda()

        for i in range(config['data']['max_src_length']):

            decoder_logit = model(input_lines_src, input_lines_trg)
            word_probs = model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        input_lines_trg = input_lines_trg.data.cpu().numpy()

        input_lines_trg = [
            [src['id2word'][x] for x in line]
            for line in input_lines_trg
        ]

        output_lines_trg_gold = input_lines_src.data.cpu().numpy()
        output_lines_trg_gold = [
            [src['id2word'][x] for x in line]
            for line in output_lines_trg_gold
        ]

        for sentence_pred, sentence_real in zip(
            input_lines_trg,
            output_lines_trg_gold,
        ):
            if '</s>' in sentence_pred:
                index = sentence_pred.index('</s>')
            else:
                index = len(sentence_pred)
            preds.append(sentence_pred[:index + 1])

            if verbose:
                print(' '.join(sentence_pred[:index + 1]))

            if '</s>' in sentence_real:
                index = sentence_real.index('</s>')
            else:
                index = len(sentence_real)
            if verbose:
                print(' '.join(sentence_real[:index + 1]))
            if verbose:
                print('--------------------------------------')
            ground_truths.append(sentence_real[:index + 1])

    return get_bleu(preds, ground_truths)
