"""Main script to run things"""
import sys

#sys.path.append('/mnt/sdb1/Dropbox (GaTech)/Chicken/SciFiEventExperiments/SciFi-GenEvent-GenEvent-Split')

from data_utils import *
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
from evaluate import model_perplexity_prebatched
import math
import numpy as np
import logging
import argparse
import os
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
load_dir = config['data']['load_dir']
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


logging.info('Reading data ...')

word2id, id2word = read_vocab(config['data']['vocab_file'])

src = read_bucket_data(word2id, id2word,
	src=config['data']['src'],
	config=config
)

val = read_bucket_data(word2id, id2word,
	src=config['data']['val_src'],
	config=config)

bucket_delim = "%%%%%%%%%%%"


max_length = config['data']['max_length']
src_vocab_size = len(word2id)


logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['model']['model_name']))
logging.info('Hidden : %s ' % (config['model']['hidden']))
logging.info('Source Word Embedding Dim  : %s' %(config['model']['dim_word_src']))
logging.info('RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (config['model']['n_layers_trg']))
logging.info('Source RNN Bidirectional  : %s' %(config['model']['bidirectional']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lr']))
logging.info('Found %d words in vocabulary ' % (src_vocab_size))

weight_mask = torch.ones(src_vocab_size).cuda()
weight_mask[word2id['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
np.set_printoptions(precision=8)

unigram_probs = read_unigram(config['data']['unigram_probs'])

model = Seq2SeqAttentionSharedEmbeddingHidden(
	emb_dim=config['model']['dim_word_src'],
	vocab_size=src_vocab_size,
	hidden_dim=config['model']['dim'],
	pad_token=word2id['<pad>'],
	bidirectional=config['model']['bidirectional'],
	nlayers=config['model']['n_layers_src'],
	nlayers_trg=config['model']['n_layers_trg'],
	dropout=config['model']['dropout']
  ).cuda()

if load_dir:
	model.load_state_dict(torch.load(
		open(load_dir)
	))

if config['training']['optimizer'] == 'adam':
	lr = config['training']['lr']
	optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
	optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
	lr = config['training']['lr']
	optimizer = optim.SGD(model.parameters(), lr=lr)
else:
	raise NotImplementedError("Learning method not recommend for task")

if config['data']['bigram'] == True:
	logging.info("bigram events")
	batches = getAllBigramBatches(src['data'],bucket_delim, 20, 5, config['model']['hidden'])
	val_batches = getAllBigramBatches(val['data'], bucket_delim, 20, 5)
else:
	batches = getAllBatches(src['data'],bucket_delim, 20, 5, config['model']['hidden'])
	val_batches = getAllBatches(val['data'], bucket_delim, 20, 5)
logging.info("Pre-loaded all data")

min_perp = sys.maxsize


for i in range(0,config['training']['epochs']):
	logging.info("Epoch: %0d" % i)

	losses = []
	h = None
	c = None
	X = []

	#for each batch
	batch_keys = list(batches.keys())
	batch_keys.sort()
	for batch_num in batch_keys:
		batch = batches[batch_num]
		input_lines_src, _, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2id, add_start=True, add_end=True)


		if batch["restart"] == True or config['model']['hidden'] == False:
			h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
				#restart hidden state; input_lines_src only used for size here

		decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)

		optimizer.zero_grad()

		loss = loss_criterion(
			decoder_logit.contiguous().view(-1, src_vocab_size),
			output_lines_trg.view(-1)
		)
		h.detach_()
		c.detach_()
		losses.append(loss.data.item())
		loss.backward()
		optimizer.step()



		if (
			config['management']['print_samples'] and
			batch_num % config['management']['print_samples'] == 0
		):

			word_probs = model.decode(
				decoder_logit
			).data.cpu().numpy().argmax(axis=-1)

			output_lines_trg = output_lines_trg.data.cpu().numpy()
			for sentence_pred, sentence_real in zip(
				word_probs[:5], output_lines_trg[:5]
			):
				sentence_pred = [id2word[x] for x in sentence_pred]
				sentence_real = [id2word[x] for x in sentence_real]

				if '</s>' in sentence_real:
					index = sentence_real.index('</s>')
					sentence_real = sentence_real[:index]
					sentence_pred = sentence_pred[:index]

				print('Predicted : %s ' % (' '.join(sentence_pred)))
				print('-----------------------------------------------')
				print('Real : %s ' % (' '.join(sentence_real)))
				print('===============================================')


	print('Epoch : %d  Loss : %.8f' % (i, np.mean(losses)))
	losses = []

	hidden = True if config['model']['hidden'] == True else False
	perp = model_perplexity_prebatched(model, val_batches, word2id, id2word, config, unigram_probs,not_hidden=not(hidden))
	logging.info('Perplexity on Validation: %f' % perp)
"""
	if perp < min_perp:
		logging.info('Saving model ...')		
		min_perp = perp
		torch.save(
			model.state_dict(),
			open(os.path.join(
				save_dir,
				experiment_name + '__epoch_%d' % (i) + '.model'), 'wb'
			)
		)


torch.save(
	model.state_dict(),
	open(os.path.join(
		save_dir,
		experiment_name + '__epoch_final%d' % (config['training']['epochs']) + '.model'), 'wb'
	)
)
"""