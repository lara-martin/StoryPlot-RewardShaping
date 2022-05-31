"Decode Seq2Seq model with beam search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
from data_utils import read_config, read_data_pipeline, read_vocab, read_bucket_data, init_state, getAllBatches, process_batch, pipeline_srcOnly
import argparse
import os
import numpy
import json

class GreedyDecoder(object):
	"""Beam Search decoder."""

	def __init__(
		self,
		config,
		top_n = 50
	):
		"""Initialize model."""
		self.config = read_config(config)
		self.model_weights = os.path.join(self.config['data']['save_dir'], self.config['data']['preload_weights'])
		self.top_n = top_n
		self.w2i, self.i2w = read_vocab(self.config['data']['vocab_file'])
		self.src_vocab_size = len(self.w2i)
		self._load_model()
		

	def _load_model(self):
		self.model = Seq2SeqAttentionSharedEmbeddingHidden(
			emb_dim=self.config['model']['dim_word_src'],
			vocab_size=self.src_vocab_size,
			hidden_dim=self.config['model']['dim'],
			pad_token=self.w2i['<pad>'],
			bidirectional=self.config['model']['bidirectional'],
			nlayers=self.config['model']['n_layers_src'],
			nlayers_trg=self.config['model']['n_layers_trg'],
			dropout=self.config['model']['dropout']
		  ).cuda()

		self.model.load_state_dict(torch.load(
			open(self.model_weights,'rb')
		))

	def decode_oneToMany(
		self,
		input_lines_src,
		input_lines_trg,
		h,
		c
	):
		"""Decode a single item and produce top n."""
		options = []
		decoder_logit, h, c = self.model(input_lines_src, input_lines_trg, h, c)
		h.detach_()
		c.detach_()
		word_probs = self.model.decode(decoder_logit)
		#for n in range(1,self.top_n+1):
				
		#for i in range(self.config['data']['max_length']):
		output = word_probs.data.cpu().numpy().argmax(axis=-1)
		"""
		if len(input_lines_trg.data.cpu().numpy()[0]) == 1:
			#changed to get top n indices
			decoder_argmax = word_probs.data.cpu().numpy().argsort(axis=-1)  #argmax(axis=-1)
			next_preds = Variable(
				 torch.from_numpy(decoder_argmax[:,-1, -n])
			).cuda()
		else:
			#if it's not a new event, take the top prob			
			decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
			next_preds = Variable(
			torch.from_numpy(decoder_argmax[:, -1])
			).cuda()

		input_lines_trg = torch.cat(
			 (input_lines_trg, next_preds.unsqueeze(1)),
			 1
		)
		"""
		options.append(output[:5])
		return options

	def translate(self, input_lines_src, input_lines_trg, h, c):
		"""Evaluate model."""
		preds = []

		# Decode a minibatch greedily add beam search decoding
		#print("DATA",input_lines_src.data)
		word_probs = self.decode_oneToMany(input_lines_src, input_lines_trg, h, c)
		output_list = []
		for trg in word_probs:
			# Copy minibatch outputs to cpu and convert ids to words
			trg = [
					[self.i2w[int(x)] for x in line]
					for line in trg
			]
			# Process outputs
			for sentence_pred in trg:
				#print("SENT PRED",sentence_pred)
				if '</s>' in sentence_pred:
					index = sentence_pred.index('</s>')
				else:
					index = len(sentence_pred)
				#preds.append(' '.join(['<s>'] + sentence_pred[:index + 1]))
				preds.append(' '.join(sentence_pred))

		#print("PREDS",preds)
		for p in preds:
			output_list.append(p.replace('</s>', '').replace('<s>', '').strip())
		#quit()
		return output_list, h, c

	def pipeline_predict(self,data, h, c, start=False):
		#method for actual pipeline, continues hidden state, if necessary
		input_lines_src, input_lines_trg = pipeline_srcOnly(data, self.w2i)
		if start:
			h,c = init_state(input_lines_src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
		output, h, c = self.translate(input_lines_src, input_lines_trg, h, c)
		return " ".join(output), h, c


