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
model_name = config['data']['preload_weights']


word2id, id2word = read_vocab(config['data']['vocab_file'])

src = read_bucket_data(word2id, id2word,
	src=config['data']['test_src'],
	config=config
)


bucket_delim = "%%%%%%%%%%%"


max_length = config['data']['max_length']
src_vocab_size = len(word2id)

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


if save_dir:
	model.load_state_dict(torch.load(
		open(os.path.join(save_dir,model_name), 'rb')
	))

if config['data']['bigram'] == True:
	print("bigram events")
	batches = getAllBigramBatches(src['data'],bucket_delim, 20, 5, config['model']['hidden'])
else:
	batches = getAllBatches(src['data'],bucket_delim, 20, 5, config['model']['hidden'])



batch_keys = list(batches.keys())
batch_keys.sort()
h = None
c = None
for batch_num in batch_keys:
	batch = batches[batch_num]
	input_lines_src, _, _, output_lines_trg = process_batch(batch['src'], batch['trg'], word2id, add_start=True, add_end=True)
	#print(input_lines_src)

	input_lines_trg = torch.LongTensor(
		[
			[word2id['<s>']]
			for i in range(0, input_lines_src.data.size(0))
		]
	).cuda()



	if batch["restart"] == True or config['model']['hidden'] == False:
		h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
			#restart hidden state; input_lines_src only used for size here

	for i in range(0,input_lines_src.data.size(1)):
		#print(input_lines_trg)

		decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)
		h.detach_()
		c.detach_()

		word_probs = model.decode(decoder_logit)
		decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)


		#print(decoder_argmax)


		next_preds = Variable(
	        torch.from_numpy(decoder_argmax[:, -1])
		).cuda()

		#print(next_preds)


		
		input_lines_trg = torch.cat(
	        (input_lines_trg, next_preds.unsqueeze(1)),
	        1
		)

			
	input_lines_trg = input_lines_trg.data.cpu().numpy()
	#print(input_lines_trg)



	output_lines_trg = output_lines_trg.data.cpu().numpy()
	for sentence_pred, sentence_real in zip(
		input_lines_trg[:5], output_lines_trg[:5]
	):
		sentence_pred = [id2word[x] for x in sentence_pred]
		#print(sentence_pred)
		sentence_real = [id2word[x] for x in sentence_real]

		if '</s>' in sentence_real:
			index = sentence_real.index('</s>')
			sentence_real = sentence_real[:index]
			sentence_pred = sentence_pred[1:]

		print('Predicted : %s ' % (' '.join(sentence_pred)))
		print('-----------------------------------------------')
		print('Real : %s ' % (' '.join(sentence_real)))
		print('===============================================')



