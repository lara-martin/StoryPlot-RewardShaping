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
from evaluate import get_bleu, model_perplexity_prebatched

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
		self.goal = self.config['training']['goal_verb']
		

	def _load_model(self):
		#if self.config['model']['seq2seq'] == 'hidden':
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

	def load_test_data():
		data = read_bucket_data(self.w2i, self.i2w,
			src=self.config['data']['test_src'],
			config=self.config
		)		
		self.test_batches = getAllBatches(data['data'], bucket_delim, 20, 5)
		del(data)








	def decode_oneToMany(
		self, input_lines_src,
		h,
		c
	):
		#print("INPUT SRC",input_lines_src)
		#print("INPUT SRC Size",input_lines_src.size())
		"""Decode a single item and produce top n."""

		decoder_init_state, ctx, _, _, c_t = self.model.decode_encoder(input_lines_src, h, c)
		h.detach_()
		c.detach_()

		input_lines_trg = torch.LongTensor(
			[
				[self.w2i['<s>']]
				for i in range(0, input_lines_src.size(0))
			]
		).cuda()



		options = []
		for n in range(1,self.top_n+1): #for each of the top n


			for i in range(input_lines_src.size(1)):
				#print("TRG INPUT",input_lines_trg)
				decoder_logit = self.model.decode_decoder(input_lines_trg, decoder_init_state, c_t, ctx)
				word_probs = self.model.decode(decoder_logit)
				#print("WORD PROB SIZE",word_probs.size())
				
				if len(input_lines_trg.data.cpu().numpy()[0]) == 2:
					#changed to get top n indices for verb
					decoder_argmax = word_probs.data.cpu().numpy().argsort(axis=-1)  #topk
					next_preds = torch.from_numpy(decoder_argmax[:,-1, -n]).cuda()
				else:				
					#if it's not a new event, take the top prob			
					decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
					next_preds = torch.from_numpy(decoder_argmax[:,-1]).cuda()
				#print([self.i2w[w] for w in next_preds.cpu().numpy()])
				input_lines_trg = torch.cat((input_lines_trg, next_preds.unsqueeze(1)),1)
				#print("INPUT LINES TRG",input_lines_trg)
			options.append(input_lines_trg)
			input_lines_trg = torch.LongTensor(
				[
					[self.w2i['<s>']]
					for i in range(0, input_lines_src.size(0))
				]
			).cuda()
				#input_lines_trg = torch.from_numpy(decoder_argmax[:, -1]).cuda()
				#event_option += [self.i2w[w] for w in input_lines_trg.cpu().numpy()]
				#print("EVENT OPTION",event_option)
			
			#options.append(event_option)
		return options
























	"""
	def decode_oneToMany(
		self,
		input_lines_src,
		h,
		c
	):


		# Initialize target with <s> for every sentence
		input_lines_trg = torch.LongTensor(
				[
					[self.w2i['<s>']]
					for i in range(0, input_lines_src.size(0))
				]
			).cuda()

		options = []




		###########
		decoder_logit, h, c = self.model(input_lines_src, input_lines_trg, h, c)
		predictions = self.model.decode(decoder_logit).data.cpu().numpy()
		h.detach_()
		c.detach_()
		top_words = predictions.argmax(axis=-1)        
		word_probs = list(predictions)#[-1]

		for i in range(input_lines_src.size(1)):
			for j in range(input_lines_src.size(0)):
				prob_event = list(word_probs[i])
				prob_word = list(prob_event[j])
			top_preds = [self.i2w[x] for x in top_words[i][:-1]]
			print(top_preds)
			print('Predicted : %s ' % (' '.join(top_preds)))

		quit()
		#############


		for n in range(1,self.top_n+1):
			print(n)			
			for i in range(input_lines_src.size(1)-1): #for the length of the event
				decoder_logit, h, c = self.model(input_lines_src, input_lines_trg, h, c)
				word_probs = self.model.decode(decoder_logit)
				h.detach_()
				c.detach_()
				print(i)
				if len(output_lines_trg.data.cpu().numpy()[0]) == 2: #find the top n verbs
					#changed to get top n indices
					decoder_argmax = word_probs.data.cpu().numpy().argsort(axis=-1)
					print("DECODER ARGSRT",decoder_argmax)
					next_preds = Variable(
						 torch.from_numpy(decoder_argmax[:,i, -n])
					).cuda()

				else:
					#if we're not at the verb, take the top prob			
					decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
					print("DECODER ARGMAX",decoder_argmax)
					print([self.i2w[int(x)] for x in list(decoder_argmax[0])])
					next_preds = Variable(
						torch.from_numpy(decoder_argmax[:, i])
					).cuda()
					print("NEXT PREDS",next_preds)

				print("OUTPUT LINES" , output_lines_trg)

				output_lines_trg = torch.cat(
					 (output_lines_trg, next_preds.unsqueeze(1)),
					 1
				)
			options.append(output_lines_trg)
			print(options)

			#restart
			output_lines_trg = torch.LongTensor(
					[
						[self.w2i['<s>']]
						for i in range(0, output_lines_trg.size(0))
					]
				).cuda()
		return options, h, c
	"""
	def translate(self, input_lines_src, input_lines_trg, h, c):
		"""Evaluate model."""
		preds = []
		with torch.no_grad():
			# Decode a minibatch greedily add beam search decoding
			output_lines_trg = self.decode_oneToMany(input_lines_src, h, c)
			output_list = []
			for trg in output_lines_trg:
				# Copy minibatch outputs to cpu and convert ids to words
				trg = trg.data.cpu().numpy()
				print(trg)
				print([self.i2w[int(x)] for x in trg[0]])
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
					preds.append(' '.join(sentence_pred))

		for p in preds:
			output_list.append(p.replace('</s>', '').replace('<s>', '').strip())
		return output_list, h, c

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

			if batch["restart"] == True or self.config['hidden'] == False:
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
			if start or self.config['hidden'] == False:
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


	"""
	def pipeline_predict(self,data):
		#data should be a single event
		src, _, _ = read_data_pipeline(data, 0, self.w2i)
		h,c = init_state(src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
		output, h, c = self.translate(src, h, c)
		return " ".join(output)
	"""
	def pipeline_predict(self,data, h, c, start=False):
		#method for actual pipeline, continues hidden state, if necessary
		src_input_lines, src_output_lines = pipeline_srcOnly(data, self.w2i)
		if start:
			h,c = init_state(src_input_lines, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['dim'])
		output, h, c = self.translate(src_input_lines, src_output_lines, h, c)
		return " ".join(output), h, c



