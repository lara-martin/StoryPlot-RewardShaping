"Decode Seq2Seq model with beam search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
from data_utils import *
import argparse
import os
import numpy
import json
from evaluate_forward import getPerplexityBLEU_prebatched, getPerplexityBLEU_prebatched_orig

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

		self.bucket_delim = "%%%%%%%%%%%%%"
		self.src_vocab_size = len(self.w2i)
		self._load_model()

		
		weight_mask = torch.ones(self.src_vocab_size).cuda()
		weight_mask[self.w2i['<pad>']] = 0
		self.loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
		self.test_batches = {}
		

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


	def forward_decode(self, input_lines_src, input_lines_trg, h, c):
		output = []
		decoder_logit, h, c = self.model(input_lines_src, input_lines_trg, h, c)
		word_probs = self.model.decode(
				decoder_logit
			).data.cpu().numpy().argmax(axis=-1)


		for sentence_pred in word_probs[:5]:

			sentence_pred = [self.i2w[x] for x in sentence_pred]

			if '</s>' in sentence_pred:
				index = sentence_pred.index('</s>')
				sentence_pred = sentence_pred[:index]
			output.append(sentence_pred)


		return output, h, c



	def load_test_data():
		data = read_bucket_data(self.w2i, self.i2w,
			src=self.config['data']['test_src'],
			config=self.config
		)		
		self.test_batches = getAllBatches(data['data'], bucket_delim, 20, 5)
		del(data)



	def story_helper_prebatch(self):
		self.load_test_data()
		rets = []
		trgs = []
		bleu = []
		h = None
		c = None
		start = True
		for batch_num in list(self.test_batches.keys()):
			batch = self.test_batches[batch_num]
			src, trg = process_batch_pipeline(batch['src'], batch['trg'], self.w2i, add_start=True, add_end=True)

			if batch["restart"] == True or self.config['model']['hidden'] == False:
				h,c = init_state(src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])

			outputs, h, c = self.translate(src, h, c)
			rets.append(outputs)
			translated_trg = [self.i2w[w] for w in trg[0][1:-1]]
			bleu.append(get_bleu(outputs,translated_trg))
			trgs.append(translated_trg)

		return rets, trgs, bleu

	def story_helper(self, data):
		t = 0
		rets = []
		trgs = []
		bleu = []
		trg = ""
		h = None
		c = None
		start = True
		while("NoneType" not in str(type(trg))): #reached the end of the story
			src, trg, t = read_data_pipeline(data, t, self.w2i)
			if start or self.config['model']['hidden'] == False:
				h,c = init_state(src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
				start = False
			if "NoneType" not in str(type(src)):
				outputs, h, c = self.translate(src, h, c)
				rets.append(outputs)
				if trg != None:
					bleu.append(get_bleu(src,trg))
					trgs.append(trg)
		translated_trg = []
		for trg in trgs:
			#print(trg[0])
			translated_trg.append([self.i2w[w] for w in trg[0][1:-1]])

		return rets, translated_trg, bleu

	def story_eval_prebatch(self, data=None):
		#data should be a single story string, events delimited by "|||"
		story_rets = []
		bleus = []
		gold = []
		if data == None:
			#use test file
			rets, trgs, bleu = self.story_helper_prebatch()
			story_rets.append(rets)
			bleus.append(bleu)
			gold.append(trgs)
		else:
			rets, trgs, bleu = self.story_helper(line.strip())
			story_rets.append(rets)
			bleus.append(bleu)
			gold.append(trgs)
		return story_rets, gold, bleus

	def story_eval(self, data=None):
		#data should be a single story string, events delimited by "|||"
		story_rets = []
		bleus = []
		gold = []
		if data == None:
			#use test file
			for line in self.data['data']:
				if self.bucket_delim in line: continue
				rets, trgs, bleu = self.story_helper(line.strip())
				story_rets.append(rets)
				bleus.append(bleu)
				gold.append(trgs)
		else:
			rets, trgs, bleu = self.story_helper(line.strip())
			story_rets.append(rets)
			bleus.append(bleu)
			gold.append(trgs)
		return story_rets, gold, bleus

	def perplexity(self):
		return model_perplexity_prebatched(self.model, self.test_batches, self.w2i, self.loss_criterion)

	def pipeline_predict(self,data):
		#data should be a single event
		input_lines_src, input_lines_trg = process_batch_srcOnly(data, self.w2i)
		h,c = init_state(input_lines_src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
		output, h, c = self.forward_decode(input_lines_src, input_lines_trg, h, c)
		return output

	def test_set_eval(self):
		test = read_bucket_data(self.w2i, self.i2w,
			src=self.config['data']['test_src'],
			config=self.config)
		batches = getAllBatches(test['data'],self.bucket_delim, 20, 5)
		unigram_probs = read_unigram(self.config['data']['unigram_probs'])

		return getPerplexityBLEU_prebatched(self.model, batches, self.w2i, self.i2w, self.config, unigram_probs, self.bucket_delim, add_start=True, add_end=True)

	def test_set_eval_orig(self):
		test = read_bucket_data(self.w2i, self.i2w,
			src=self.config['data']['test_src'],
			config=self.config)
		batches = getAllBatches(test['data'],self.bucket_delim, 20, 5)
		unigram_probs = read_unigram(self.config['data']['unigram_probs'])

		return getPerplexityBLEU_prebatched_orig(self.model, batches, self.w2i, self.i2w, self.config, unigram_probs, self.bucket_delim, add_start=True, add_end=True)

	def test_set_eval_orig_bigram(self):
		test = read_bucket_data(self.w2i, self.i2w,
			src=self.config['data']['test_src'],
			config=self.config)
		batches = getAllBigramBatches(test['data'],self.bucket_delim, 20, 5)
		unigram_probs = read_unigram(self.config['data']['unigram_probs'])

		return getPerplexityBLEU_prebatched_orig(self.model, batches, self.w2i, self.i2w, self.config, unigram_probs, self.bucket_delim, add_start=True, add_end=True)

