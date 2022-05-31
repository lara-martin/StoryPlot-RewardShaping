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
	#print(batch['src'])
	#print(batch['trg'])
	input_lines_src, _ = process_batch_pipeline(batch['src'], batch['trg'], word2id, add_start=True, add_end=True)

	if batch["restart"] == True or config['model']['hidden'] == False:
		h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
			#restart hidden state; input_lines_src only used for size here

	
	options = []


	input_lines_trg = torch.LongTensor(
		[
			[word2id['<s>']]
			for i in range(0, input_lines_src.data.size(0))
		]
	).cuda()


	#print("INPUT_LINES_TRG",input_lines_trg)
	size = 0
	#for i in range(input_lines_src.data.size(1)):
	while size < config["data"]["max_length"]:
		decoder_logit, _, _ = model(input_lines_src, input_lines_trg, h, c)

		h.detach_()
		c.detach_()

		word_probs = model.decode(
			decoder_logit
		)#.data.cpu().numpy()#.argmax(axis=-1) # batch size x event x vocab size

		word_max = word_probs.max(-1)
		#print("WORD_MAX",word_max.indices)

		#input_lines_trg = word_max.indices
		size = input_lines_trg.size(-1)

		input_lines_trg = torch.cat(
			 (input_lines_trg, word_max.indices),#.unsqueeze(1)),
			 1
		)
		

	options.append(word_max.indices.data.cpu().numpy().tolist())
	#for sentence_pred in options[:5]:
	for option_n in options: #for each of the n
		for sentence_pred in option_n: #for each item in batch
			print(sentence_pred)
			sentence_pred = [id2word[x] for x in sentence_pred[:5]]

			if '</s>' in sentence_pred:
				index = sentence_pred.index('</s>')
				sentence_pred = sentence_pred[:index]

			print('Predicted : %s ' % (' '.join(sentence_pred)))
			print('-----------------------------------------------')

	quit()
