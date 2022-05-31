from data_utils import *
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
import numpy as np
import logging
import argparse
import os
import torch


class Decoder:
	def __init__(self, config_path, top_n=1):
		self.config = read_config(config_path)
		experiment_name = hyperparam_string(self.config)
		save_dir = self.config['data']['save_dir']
		model_name = self.config['data']['preload_weights']

		self.w2i, self.i2w = read_vocab(self.config['data']['vocab_file'])
		self.max_length = self.config['data']['max_length']
		src_vocab_size = len(self.w2i)
		self.top_n = top_n


		self.model = Seq2SeqAttentionSharedEmbeddingHidden(
			emb_dim=self.config['model']['dim_word_src'],
			vocab_size=src_vocab_size,
			hidden_dim=self.config['model']['dim'],
			pad_token=self.w2i['<pad>'],
			bidirectional=self.config['model']['bidirectional'],
			nlayers=self.config['model']['n_layers_src'],
			nlayers_trg=self.config['model']['n_layers_trg'],
			dropout=self.config['model']['dropout']
		  ).cuda()


		if save_dir:
			self.model.load_state_dict(torch.load(
				open(os.path.join(save_dir,model_name), 'rb')
			))


	def pipeline_predict(self,data, h, c, start=False):
		input_lines_src, _ = pipeline_srcOnly(data, self.w2i)
		#input_lines_src, _ = process_batch_pipeline(batch['src'], batch['trg'], self.w2i, add_start=True, add_end=True)

		if start == True or self.config['model']['hidden'] == False:
			h, c = init_state(input_lines_src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
				#restart hidden state; input_lines_src only used for size here

		
		options = []


		input_lines_trg = torch.LongTensor(
			[
				[self.w2i['<s>']]
				for i in range(0, input_lines_src.data.size(0))
			]
		).cuda()


		#print("INPUT_LINES_TRG",input_lines_trg)
		size = 0
		decoder_init_state, ctx, h, c, c_t = self.model.decode_encoder(input_lines_src, h, c)
		for i in range(1,self.top_n+1):
			while size < self.max_length:
				#decoder_logit, _, _ = self.model(input_lines_src, input_lines_trg, h, c)
				decoder_logit = self.model.decode_decoder(input_lines_trg, decoder_init_state, c_t, ctx)
				h.detach_()
				c.detach_()

				word_probs = self.model.decode(
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

		returning = []
		for option_n in options: #for each of the n
			for sentence_pred in option_n: #for each item in batch
				#print(sentence_pred)
				sentence_pred = [self.i2w[x] for x in sentence_pred[:5]]

				if '</s>' in sentence_pred:
					index = sentence_pred.index('</s>')
					sentence_pred = sentence_pred[:index]

				returning.append(sentence_pred)
				#print('Predicted : %s ' % (' '.join(sentence_pred)))
				#print('-----------------------------------------------')
		return returning, h, c


