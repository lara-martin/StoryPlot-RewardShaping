import torch
import torch.nn as nn



def decode_oneToMany(
	model,
	input_lines_src,
	input_lines_trg,
	h,
	c,
	config,
	w2i,
	top_n
):
	"""Decode a single item and produce top n."""
	options = []
	for n in range(1,top_n+1):
		for i in range(config['data']['max_trg_length']):
			decoder_logit, h, c = model(input_lines_src, input_lines_trg, h, c)
			word_probs = model.decode(decoder_logit)
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
		options.append(input_lines_trg)
		with torch.no_grad():
			input_lines_trg = torch.LongTensor(
					[
						[w2i['<s>']]
						for i in range(0, input_lines_src.size(0))
					]
				).cuda()
	return options, h, c

def translate(model, src_test, h, c, config, w2i, i2w, top_n=1):
	"""Evaluate model."""
	preds = []
	#for j in range(0, len(src_test)):
	with torch.no_grad():
		# Initialize target with <s> for every sentence
		input_lines_trg = torch.LongTensor(
			[
				[w2i['<s>']]
				for i in range(0, src_test.size(0))
			]
		).cuda()


		# Decode a minibatch greedily add beam search decoding
		input_lines_trg, h, c = decode_oneToMany(model, src_test, input_lines_trg, h, c, config, w2i, top_n)
		output_list = []
		for trg in input_lines_trg:
			# Copy minibatch outputs to cpu and convert ids to words
			trg = trg.data.cpu().numpy()
			trg = [
					[i2w[int(x)] for x in line]
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