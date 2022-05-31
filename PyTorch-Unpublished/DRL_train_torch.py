import math
from statistics import mean
import os
from collections import defaultdict
import sys
import pickle
import numpy as np
from random import randint
import csv
from data_utils import *
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
from vocab_for_sampling import load_obj, statistics, statistics_horizontal
from compute_statistics import word_distance, word_clustering
from evaluate import model_perplexity_prebatched, getPerplexityBLEU_forBatch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import logging


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

src_data = read_bucket_data(word2id, id2word,
  src=config['data']['src'],
  config=config
)

val_data = read_bucket_data(word2id, id2word,
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
logging.info('Goal verb is : %s' % (config['training']['goal_verb']))

weight_mask = torch.ones(src_vocab_size).cuda()
weight_mask[word2id['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask, reduction='none').cuda() #reduction is changed to none to integrate reward
np.set_printoptions(precision=8)


goal_word = config['training']['goal_verb'].split("-")[0]
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
	open(os.path.join(load_dir,config['data']['preload_weights']),'rb')
  ))
else:
  raise "No loading directory given." 


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
  batches = getAllBigramBatches(src_data['data'],bucket_delim, 20, 5, config['model']['hidden'])
  val_batches = getAllBigramBatches(val_data['data'], bucket_delim, 20, 5)
else:
  batches = getAllBatches(src_data['data'],bucket_delim, 20, 5, config['model']['hidden'])
  val_batches = getAllBatches(val_data['data'], bucket_delim, 20, 5)
logging.info("Pre-loaded all data")


VerbNet = ['absorb-39.8','accept-77.1','accompany-51.7','acquiesce-95.1','act-114','addict-96','adjust-26.9','admire-31.2','admit-64.3','adopt-93','advise-37.9','allow-64.1','amalgamate-22.2','amuse-31.1','animal_sounds-38','appeal-31.4','appear-48.1.1','appoint-29.1','assessment-34.1','assuming_position-50','attend-107.4','avoid-52','banish-10.2','base-97.1','battle-36.4','become-109.1','beg-58.2','begin-55.1','being_dressed-41.3.3','bend-45.2','benefit-72.2','berry-13.7','bill-54.5','birth-28.2','body_internal_motion-49.1','body_internal_states-40.6','body_motion-49.2','braid-41.2.2','break-45.1','break_down-45.8','breathe-40.1.2','bring-11.3','build-26.1','bulge-47.5.3','bully-59.5','bump-18.4','butter-9.9','calibratable_cos-45.6.1','calve-28.1','captain-29.8','care-88.1','caring-75.2','carry-11.4','carve-21.2','caused_calibratable_cos-45.6.2','change_bodily_state-40.8.4','characterize-29.2','chase-51.6','cheat-10.6.1','chew-39.2','chit_chat-37.6','classify-29.10','clear-10.3','cling-22.5','cognize-85','coil-9.6','coloring-24','compel-59.1','complain-37.8','complete-55.2','comprehend-87.2','comprise-107.2','concealment-16','conduct-111.1','confess-37.10','confine-92','confront-98','conjecture-29.5','consider-29.9','conspire-71','consume-66','contain-15.4','contiguous_location-47.8','continue-55.3','contribute-13.2','convert-26.6.2','cooking-45.3','cooperate-73.1','cope-83','correlate-86.1','correspond-36.1.1','cost-54.2','crane-40.3.2','create-26.4','curtsey-40.3.3','cut-21.1','debone-10.8','declare-29.4','dedicate-79','deduce-97.2','defend-72.3','deprive-10.6.2','destroy-44','devour-39.4','die-42.4','differ-23.4','dine-39.5','disappearance-48.2','disassemble-23.3','discover-84','disfunction-105.2.2','distinguish-23.5','dress-41.1.1','dressing_well-41.3.2','drive-11.5','dub-29.3','earn-54.6','eat-39.1','empathize-88.2','employment-95.3','encounter-30.5','enforce-63','engender-27.1','ensure-99','entity_specific_cos-45.5','entity_specific_modes_being-47.2','equip-13.4.2','escape-51.1','establish-55.5','estimate-34.2','exceed-90','exchange-13.6.1','exclude-107.3','exhale-40.1.3','exist-47.1','feeding-39.7','ferret-35.6','fill-9.8','fire-10.10','fit-54.3','flinch-40.5','floss-41.2.1','focus-87.1','forbid-64.4','free-10.6.3','free-80','fulfilling-13.4.1','function-105.2.1','funnel-9.3','future_having-13.3','get-13.5.1','give-13.1','gobble-39.3','gorge-39.6','groom-41.1.2','grow-26.2.1','harmonize-22.6','help-72.1','herd-47.5.2','hiccup-40.1.1','hire-13.5.3','hit-18.1','hold-15.1','hunt-35.1','hurt-40.8.3','illustrate-25.3','image_impression-25.1','indicate-78','initiate_communication-37.4.2','inquire-37.1.2','instr_communication-37.4.1','intend-61.2','interact-36.6','interrogate-37.1.3','invest-13.5.4','investigate-35.4','involve-107.1','judgment-33.1','keep-15.2','knead-26.5','learn-14','leave-51.2','lecture-37.11','let-64.2','light_emission-43.1','limit-76','linger-53.1','lodge-46','long-32.2','lure-59.3','manner_speaking-37.3','marry-36.2','marvel-31.3','masquerade-29.6','matter-91','meander-47.7','meet-36.3','mine-10.9','mix-22.1','modes_of_being_with_motion-47.3','multiply-108','murder-42.1','neglect-75.1','nonvehicle-51.4.2','nonverbal_expression-40.2','obtain-13.5.2','occur-48.3','order-58.3','orphan-29.7','other_cos-45.4','overstate-37.12','own-100.1','pain-40.8.1','patent-101','pay-68','peer-30.3','pelt-17.2','performance-26.7','pit-10.7','pocket-9.10','poison-42.2','poke-19','pour-9.5','preparing-26.3','price-54.4','promise-37.13','promote-102','pronounce-29.3.1','prosecute-33.2','push-12','put-9.1','put_direction-9.4','put_spatial-9.2','reach-51.8','rear-26.2.2','reciprocate-112','reflexive_appearance-48.1.2','refrain-69','register-54.1','rehearse-26.8','reject-77.2','relate-86.2','rely-70','remedy-45.7','remove-10.1','render-29.90','representation-110.1','require-103','resign-10.11','respond-113','result-27.2','risk-94','rob-10.6.4','roll-51.3.1','rummage-35.5','run-51.3.2','rush-53.2','satisfy-55.7','say-37.7','scribble-25.2','search-35.2','see-30.1','seem-109','send-11.1','separate-23.1','settle-36.1.2','shake-22.3','sight-30.2','simple_dressing-41.3.1','slide-11.2','smell_emission-43.3','snooze-40.4','sound_emission-43.2','sound_existence-47.4','spank-18.3','spatial_configuration-47.6','spend_time-104','split-23.2','spray-9.7','stalk-35.3','steal-10.5','stimulate-59.4','stimulus_subject-30.4','stop-55.4','subjugate-42.3','subordinate-95.2.1','substance_emission-43.4','substitute-13.6.2','succeed-74','suffocate-40.7','supervision-95.2.2','support-15.3','suspect-81','sustain-55.6','swarm-47.5.1','swat-18.2','talk-37.5','tape-22.4','tell-37.2','terminus-47.9','throw-17.1','tingle-40.8.2','touch-20','transcribe-25.4','transfer_mesg-37.1.1','trick-59.2','trifle-105.3','try-61.1','turn-26.6.1','urge-58.1','use-105.1','vehicle-51.4.1','vehicle_path-51.4.3','vn_class-3.dtd','vn_schema-3.xsd','void-106','volunteer-95.4','waltz-51.5','want-32.1','weather-57','weekend-56','wink-40.3.1','wipe_instr-10.4.2','wipe_manner-10.4.1','wish-62','withdraw-82','work-73.2']


def cluster_number(verb,clusters):
	for c in clusters.keys():
		if verb in clusters[c]:
	  		#print("Found "+str(verb)+" in cluster "+str(c))
	  		return c 
	#print("Couldn't find "+str(verb))
	return -1


def resample(model, decoder_logit, input_lines_src, input_lines_trg, id2word, word2id, clusters):
	#Resample
	#Set 'outputs' variable equal to the decoder input
	size = list(input_lines_src.size())
	outputs = np.zeros(shape = (size[0],size[1]), dtype = np.int)
	predictions = list(model.decode(decoder_logit).data.cpu().numpy())

	#print(input_lines_src)
	for i in range(0,size[0]): #for each event in the batch
		src = list(input_lines_src[i].cpu().numpy())
		trg = list(input_lines_trg[i].cpu().numpy())
		#print("SRC",src)
		#print("TRG",trg)
		if len(set(trg[1:])) == 1: continue #if they're all <pad> skip
		#print(outputs)

		outputs[i][0] = trg[0] #should be "<s>"
		for j in range(1,size[1]):
			outputs[i][j] = trg[j]
		#outputs[i][j+1] = word2id['</s>']
		#print("SOURCE"," ".join([id2word[x] for x in src]))
		#print("TARGET"," ".join([id2word[x] for x in trg]))
		

		#Resample the verb based on the current distribution of the DRL model.
		c_num = cluster_number(id2word[src[2]],clusters)
		new_c_num = cluster_number(id2word[trg[2]],clusters)
		
		if c_num==config['training']['len_clusters']: #Last cluster is goal
		  	#print("CASE1",id2word[outputs[i][2]])
		  	outputs[i][2] = word2id[config['training']['goal_verb']]
		elif c_num==-1:
			#print("CASE2",id2word[outputs[i][2]])
			if len(clusters[0]) == 1:
				outputs[i][2] = word2id[clusters[0][0]]
			else:
				outputs[i][2] = word2id[clusters[0][randint(0,(len(clusters[0])-1))]]
		else:
			if (new_c_num - c_num) !=1: 
				string = []
				cluster_prob = defaultdict(int)

				if len(clusters[c_num+1])==0:
					c_num+=1

				event_probs = list(predictions[i])
				word_probs = list(event_probs[1]) #position of the verb probs
				for cn in range(len(clusters[c_num+1])): #for each verb in the next cluster
					cluster_word = clusters[c_num+1][cn]			
					cluster_prob[cluster_word] = float(word_probs[word2id[cluster_word]])
				
				max_c_prob = max(list(cluster_prob.values()))
				min_c_prob = min(list(cluster_prob.values()))
				sum_c_prob = np.sum(list(cluster_prob.values()))
				keys = []
				values = []
				for word in cluster_prob.keys(): #normalizing
					cluster_prob[word] = float(cluster_prob[word])/float(sum_c_prob)
					keys.append(word)
					values.append(cluster_prob[word])
				next_verb = [""]
				#while next_verb[0] not in VerbNet:
				next_verb = np.random.choice(keys,1, p=values)
				del(keys)
				del(values)
				#print("NEXT SELECTED VERB",next_verb[0])
				outputs[i][2] = word2id[next_verb[0]]
	#print("RESAMPLED", " ".join([id2word[x] for x in outputs[i]]))
	#print("OUTPUTS",outputs)
	return outputs


def reward(input_lines_src, input_lines_trg, outputs, id2word, word_dict_statistics, config, clusters, word_dist):	
	#Calculate reward
	batch_reward = []
	size = list(input_lines_src.size())

	#Check if any of the sampled tokens are in the wrong position. If so, set reward to 0
	for i in range(0,size[0]): #event
		train = True
		src = list(input_lines_src[i].cpu().numpy())
		trg = list(input_lines_trg[i].cpu().numpy())
		#print("SRC_r",src)
		#print("TRG_r",trg)
		reward = 1

		if len(set(trg[1:])) == 1: continue #if they're all <pad> skip
		if len(set(src[1:])) == 1: continue
		#Check if any of the sampled tokens are in the wrong position. If so, set reward to 0
		outputs[i][0] = trg[0] #should be "<s>"
		for p in range(1,size[1]): #parameter
			#print(id2word[outputs[i][p]])
			if id2word[outputs[i][p]] not in word_dict_statistics[p]:
				#there's a mistake somewhere in the event;
				#make the output the trg and don't train on it
				#print("NOT FOUND", id2word[outputs[i][p]])
				reward = 0
				for j in range(1,size[1]): 
					outputs[i][j] = trg[j]
				#outputs[i][5] = word2id['</s>']
				train = False
				break

		#Assign appropriate reward if clustering condition is satisfied.
		if train:
			#print("TRAIN")
			if id2word[outputs[i][2]] == config['training']['goal_verb']:
				#print(id2word[outputs[i][2]],"is goal")
				old_c_num = cluster_number(id2word[src[2]],clusters)
				if (old_c_num) == (len(clusters)-1):
				  	reward = max_reward #math.ceil(max_reward)
				else:
				  	reward = 0
			else:
				#print("NOT GOAL",id2word[outputs[i][2]])
				old_c_num = cluster_number(id2word[src[2]],clusters)
				new_c_num = cluster_number(id2word[outputs[i][2]],clusters)
				if (new_c_num - old_c_num) == 1:
				  	reward = word_dist[id2word[outputs[i][2]]]
				else:
				  	reward = 0

		#penalize for having positions 2 & 3 the same
		if outputs[i][2]==outputs[i][3] and reward!=-1:
			reward = reward/100

		#print(reward)
		batch_reward.append(reward)
	return batch_reward

def getDecoderLogitFromBatch(batch, h, c, id2word, word2id, config, clusters, model, add_start=True, add_end=True):
	input_lines_src, output_lines_src, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2id, add_start=True, add_end=True)

	if batch["restart"] == True or config['model']['hidden'] == False:
		h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
		#restart hidden state; input_lines_src only used for size here

	#forward run to get trg probabilities for reward
	decoder_logit, _, _ = model(input_lines_src, input_lines_trg, h, c)

	#resample and rerun
	outputs = resample(model, decoder_logit, input_lines_src, input_lines_trg, id2word, word2id, clusters)
	new_trg = torch.LongTensor(outputs).cuda()
	decoder_logit_policy, h, c = model(input_lines_src, new_trg, h, c)
	h.detach_()
	c.detach_()

	return decoder_logit_policy, h, c, outputs


min_perp = sys.maxsize
###################################################
#Calculating word clustering
word_dict_statistics = {}
word_dict_statistics = load_obj('words_for_sampling')

#posStats,individualPosStats,secondPositionStats,thirdPositionStats,positionDict = statistics(word_dict_statistics,config['data']['combined_src'])
#horizonPosStats, secondHorizonPosStats = statistics_horizontal(word_dict_statistics, config['data']['combined_src'])
	#words that occur before goal in a story

word_dist = word_distance(config['data']['combined_src'],word_dict_statistics[1],config['training']['goal_verb'])
clusters = word_clustering(word_dist,config['training']['len_clusters'],config['training']['goal_verb'])

sum = 0
num = 0

max_reward = 0
min_reward = sys.maxsize
for cluster in clusters:
	for word in clusters[cluster]:
		sum += word_dist[word]
		if word_dist[word]<min_reward:
		  	min_reward = word_dist[word]
		num += 1
	sum = 0
	num = 0
for key in word_dist.keys():
  word_dist[key] += -min_reward
  word_dist[key] = word_dist[key]/2
for cluster in clusters:
	for word in clusters[cluster]:
		sum += word_dist[word]
		if word_dist[word]>max_reward:
		  	max_reward = word_dist[word]
		num += 1
	# print (float(sum)/num)
	sum = 0
	num = 0

sorted_word_dist = [(word_dist[k],k) for k in sorted(word_dist, key=word_dist.get)]
#sorted(word_dist, key=lambda k,v: (v,k))
new_sorted_word = defaultdict(int)
for wd in sorted_word_dist:
  new_sorted_word[wd[0]] = wd[1]


w = csv.writer(open("output.csv", "w"))
for key, val in sorted_word_dist:
  w.writerow([key, val])


clusters = word_clustering(word_dist,config['training']['len_clusters'],config['training']['goal_verb'])
for word in word_dist.keys():
 	word_dist[word] = (word_dist[word] - min_reward)/(max_reward - min_reward)


max_reward = 0
min_reward = sys.maxsize
for cluster in clusters:
	for word in clusters[cluster]:
		sum += word_dist[word]
		if word_dist[word]<min_reward:
		  	min_reward = word_dist[word]
		num += 1
	sum = 0
	num = 0
for cluster in clusters:
	for word in clusters[cluster]:
		sum += word_dist[word]
		if word_dist[word]>max_reward:
			max_reward = word_dist[word]
		num += 1
	# print (float(sum)/num)
	sum = 0
	num = 0

word_dist[config['training']['goal_verb']] = 0.97
#print(word_dist)
####################################################

#training loop
for i in range(0,config['training']['epochs']):
	epoch_reward = 0
	for param_group in optimizer.param_groups:
		logging.info("Learning rate is: %0.8f" % param_group['lr'])

	losses = []
	h = None
	c = None

	#for each batch
	batch_keys = list(batches.keys())
	batch_keys.sort()
	for batch_num in batch_keys:
		batch = batches[batch_num]

		input_lines_src, output_lines_src, input_lines_trg, output_lines_trg = process_batch(batch['src'], batch['trg'], word2id, add_start=True, add_end=True)

		if batch["restart"] == True or config['model']['hidden'] ==False:
			h, c = init_state(input_lines_src, config['model']['n_layers_src'], config['model']['bidirectional'], config['model']['dim'])
			#restart hidden state; input_lines_src only used for size here

		#forward run to get trg probabilities for reward
		decoder_logit, _, _ = model(input_lines_src, input_lines_trg, h, c)

		#resample the events' verbs
		outputs = resample(model, decoder_logit, input_lines_src, input_lines_trg, id2word, word2id, clusters)

	  	#calculate reward for each event
		batch_reward = reward(input_lines_src, input_lines_trg, outputs, id2word, word_dict_statistics, config, clusters, word_dist)
		reward_tensor = torch.tensor(batch_reward).cuda()

	  	#####################################

		#Run through model again to backprop loss
		####Help from https://www.datahubbs.com/reinforce-with-pytorch/  and   https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

		#forward run
		new_trg = torch.LongTensor(outputs).cuda()
		#Run new target data through
		decoder_logit_policy, h, c = model(input_lines_src, new_trg, h, c)
		#Calculate loss
		uncondensed_loss = loss_criterion(decoder_logit_policy.contiguous().view(-1, src_vocab_size), new_trg.view(-1))

		h.detach_()
		c.detach_()

		#print("REWARDS",reward_tensor)
		optimizer.zero_grad()

		policy_loss = []
		for log_prob, R in zip(uncondensed_loss, reward_tensor):
			#print("PROB",log_prob)
			policy_loss.append(log_prob * R)
		policy_loss = torch.stack(policy_loss).sum()

		#Finally run it backward
		policy_loss.backward()
		losses.append(policy_loss.data.item())
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

				logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
				logging.info('-----------------------------------------------')
				logging.info('Real : %s ' % (' '.join(sentence_real)))
				logging.info('===============================================')






	#Batch Complete
	print('Epoch : %d  Loss : %.8f' % (i, np.mean(losses)))
	losses=[]
	
	"""
	#############VALIDATION############
	#print("VALIDATION")
	val_keys = list(val_batches.keys())
	val_keys.sort()
	p = 0
	BLEUs = []
	h_val = None
	c_val = None

	for batch_key in val_keys:
		val_batch = val_batches[batch_key]
		val_decoder_logit, h_val, c_val, new_trg = getDecoderLogitFromBatch(batch, h, c, id2word, word2id, config, clusters, model)
		predictions = model.decode(val_decoder_logit).data.cpu().numpy()
		_, _, p, b = getPerplexityBLEU_forBatch(predictions, new_trg, word2id, id2word, unigram_probs, p)
		BLEUs+=b

	perp = 2 ** -(p/len(val_keys))
	logging.info('Perplexity on Validation: %f' % perp)
	logging.info('Average BLEU on Validation: %f' % mean(BLEUs))
	#####################################
	"""
	#if perp < min_perp:
	logging.info('Saving model ...')    
	#min_perp = perp
	torch.save(
	  model.state_dict(),
	  open(os.path.join(
		save_dir,
		experiment_name + '__%s__epoch_%d' % (config['training']['goal_verb'], i) + '.model'), 'wb'
	  )
	)
"""
torch.save(
  model.state_dict(),
  open(os.path.join(
	save_dir,
	experiment_name + '__%s__epoch_final%d' % (config['training']['goal_verb'],config['training']['epochs']-1) + '.model'), 'wb'
  )
)
"""
