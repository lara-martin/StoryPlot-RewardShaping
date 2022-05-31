# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
python translate_forward.py --train_file events_genre68_train_v1.txt --test_file events_genre68_test_v1.txt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from collections import defaultdict
import random
import sys
import time
import read_data_genre68_full as read_data
import seq2seq_model_policy_v1_genre68_full_admire as seq2seq_model
import seq2seq_model_fb_genre68_full as seq2seq_model_fb
from vocab_for_sampling_genre68_full import load_obj, statistics, statistics_horizontal
from compute_statistics import reward_word_statistics2
from compute_statistics import reward_word_statistics
from compute_statistics import word_distance
from compute_statistics import word_clustering
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils_genre68_full as data_utils
from time import gmtime, strftime
import random
import csv
from random import randint

run_num = 5
max_e = 200
save_at = 40
# max_e = 10
# save_at = 10


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("cmu_vocab_size", 3548, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
# tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/data", "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "/checkpoints", "Training directory.")
tf.app.flags.DEFINE_string("train_dir", "./checkpoints", "Training directory.")
tf.app.flags.DEFINE_string("train_file", "final_train_genre68_full.txt", "Training file.")
tf.app.flags.DEFINE_string("test_file", None, "Testing file.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("max_epochs", max_e, "Maximum number of epochs.")
tf.app.flags.DEFINE_integer("beam_size",5,"beam size")

FLAGS = tf.app.flags.FLAGS

# FLAGS.max_epochs=2000

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(4,5),(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(4,5)]


#def read_data(source_path, target_path, max_size=None):
"""Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
"""

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def create_model(session, forward_only, filename = "start_fresh",scope = ""):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  # with tf.variable_scope("forward"):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.cmu_vocab_size,
      FLAGS.cmu_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype,
      scope=scope)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir,filename)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def create_model_fb(session, forward_only, filename = "start_fresh",scope = ""):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  # with tf.variable_scope("forward"):
  model = seq2seq_model_fb.Seq2SeqModel(
      FLAGS.cmu_vocab_size,
      FLAGS.cmu_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype,
      scope=scope)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir,filename)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def get_previous_lines(pos,train_set):
  secondPrev = None
  thirdPrev = None
  if train_set[0][pos-1][0][0] != data_utils.START_ID and train_set[0][pos-1][0][0]!=data_utils.END_ID:
    secondPrev = train_set[0][pos-1][0]
  if train_set[0][pos-2][0][0] != data_utils.START_ID and train_set[0][pos-2][0][0]!=data_utils.END_ID:
    thirdPrev = train_set[0][pos-2][0]
  return secondPrev,thirdPrev

def get_check_successor_lines(pos,train_set, inv_vocab, num_lines=5,target_verb="admire-31.2"):
  for i in range(num_lines):
    train_set_size = np.array(train_set).shape
    # print("The shape is:",train_set_size)
    if train_set_size[1]<=pos+i+1:
      return False
    next = inv_vocab[train_set[0][pos+i+1][0][1]]
    if next == target_verb:
      return True
    if next == data_utils.START_ID or next == data_utils.END_ID:
      return False
  return False

def print_events(outputs):
  ou = ''
  for w in xrange(FLAGS.batch_size):
    ou+=(inv_vocab[outputs[0][w]]+ ',')
    ou+=(inv_vocab[outputs[1][w]] + ',')
    ou+=(inv_vocab[outputs[2][w]] + ',')
    ou+=(inv_vocab[outputs[3][w]])
    ou+=('\n')
  print (ou)

def cluster_number(verb,clusters):
  for c in clusters.keys():
    if verb in clusters[c]:
      
      return c 
  return -1

def normalize_dict(d):
  factor=1.0/sum(d.itervalues())
  for k in d:
    d[k] = d[k]*factor
  return d

def train():
  
  with tf.Session() as sess, tf.Session() as sess_inner:
    # Create model.
    word_number = 1
    reward_words = {1: 'admire-31.2'}
    target_verb = 'admire-31.2'
    number_of_words = 1

    samples = tf.placeholder(tf.float32, shape=())
    placeholder = tf.placeholder(tf.float32,shape = (1,FLAGS.cmu_vocab_size))

    placeholder_2 = tf.placeholder(tf.float32,shape = (1,FLAGS.cmu_vocab_size-2))
    # update_op = tf.assign(samples,tf.multinomial(tf.reshape(placeholder,[FLAGS.batch_size,FLAGS.cmu_vocab_size]),1))
    samples = tf.multinomial(placeholder,1)

    samples_2 = tf.multinomial(placeholder_2,1)

    placeholder_3 = tf.placeholder(tf.float32,shape = (5,FLAGS.batch_size,FLAGS.cmu_vocab_size))
    predictions = tf.nn.softmax(placeholder_3)

    sess.run(tf.global_variables_initializer())
    sess_inner.run(tf.global_variables_initializer())

    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    bucket_id = 0
    # model = create_model(sess, False, "checkpoint_preload_weights_250")
    model = create_model(sess, False,"checkpoint_forward_genre68_full_run_1")
    # with tf.Graph().as_default():	
    # 	pretrained_model = create_model(sess, False,"checkpoint_forward_genre68_full_run_1")

    #*******************CHANGE THIS BACK*************************
    # model = create_model(sess, False,"checkpoint")
    pretrained_model = model

    # model = create_model(sess, False)
    vocab_file = open('Final_genre68_full.vocab', 'rb')
    vocab = pickle.load(vocab_file)
    vocab_file.close()
    # train_set = read_data.read_with_buckets_policy(FLAGS.train_file, vocab,_buckets)
    train_set = read_data.read_with_buckets(FLAGS.train_file, vocab,_buckets)


    params = tf.trainable_variables()
    #gradients = tf.gradients(self.losses[bucket_id],params)
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    

    flag = True
    epochs_flag = False
    inv_vocab = {k: v for k, v in enumerate(vocab)}    
    
    # vocab.index()
    # exit(0)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    #train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
    #                       for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    counter_iter=0
    pos = 0
    epochs = 0
    reward = None
    count = 0
    rollout_num = 4
    average_value = None
    own_prob = 0.9
    decrease_prob = 0.85
    old_reward = None
    div = 5
    #sess.run(tf.initialize_all_variables())   
    inv_vocab = {k: v for k, v in enumerate(vocab)}
    fl = False
    r = None

    #tf.get_default_graph().finalize()
    #print ("finalized graph") 
    log_file = open("Run-Logs/log_"+str(run_num), "wb")
    word_dict_statistics = {}
    # word_dict_statistics = load_obj('words_for_sampling')
    word_dict_statistics = load_obj('words_for_sampling_parsed_editted')

    # bigram_word_dict = {}
    # # bigram_word_dict = load_obj('words_for_sampling_bigrams')
    # bigram_word_dict = load_obj('words_for_sampling_bigrams_parsed_editted')

    word_dict = {}
    # word_dict = load_obj('words_for_sampling_model')
    word_dict = load_obj('words_for_sampling_parsed_editted')

    # posStats,individualPosStats,secondPositionStats,thirdPositionStats,positionDict = statistics(word_dict,'Events_combined_reduced.txt')
    posStats,individualPosStats,secondPositionStats,thirdPositionStats,positionDict = statistics(word_dict_statistics,'final_combined_genre68_full.txt')
    horizonPosStats, secondHorizonPosStats = statistics_horizontal(word_dict_statistics, 'final_combined_genre68_full.txt')
    #words that occur before marry in a story. 
    
    word_dist = word_distance('final_combined_genre68_full.txt',word_dict[1],'admire-31.2')
    clusters = word_clustering(word_dist,30)
    # print(clusters)
    # exit(0)

    sum = 0
    num = 0
    max_reward = 0
    min_reward = 1000000
    for cluster in clusters:
      for word in clusters[cluster]:
        sum += word_dist[word]
        if word_dist[word]<min_reward:
          min_reward = word_dist[word]
        num += 1
      sum = 0
      num = 0
    gamma = 0.5
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
    sorted_word_dist = sorted(word_dist.iteritems(), key=lambda (k,v): (v,k))
    new_sorted_word = defaultdict(int)
    for wd in sorted_word_dist:
      new_sorted_word[wd[0]] = wd[1]
    w = csv.writer(open("output.csv", "w"))
    for key, val in sorted_word_dist:
      w.writerow([key, val])
    # exit(0)
    clusters = word_clustering(word_dist,30)
    # print (word_dist)
    # print (clusters)  
    # exit(0)
    # exit(0)
    encoder_counter = 0
    raw_reward = 10000
    for word in word_dist.keys():
      word_dist[word] = (word_dist[word] - min_reward)/(max_reward - min_reward)
    max_reward = 0
    min_reward = 1000000
    for cluster in clusters:
      for word in clusters[cluster]:
        sum += word_dist[word]
        if word_dist[word]<min_reward:
          min_reward = word_dist[word]
        num += 1
      sum = 0
      num = 0
    gamma = 0.5
    for cluster in clusters:
      for word in clusters[cluster]:
        sum += word_dist[word]
        if word_dist[word]>max_reward:
          max_reward = word_dist[word]
        num += 1
      # print (float(sum)/num)
      sum = 0
      num = 0
    # print(clusters)
    # exit(0)
    while epochs < FLAGS.max_epochs:
      
      bucket_id = 0
      # Get a batch and make a step.
      start_time = time.time()
      #print (train_set[0][0])
      encoder_inputs, decoder_inputs, target_weights, pos, epoch_flag = model.get_batch_skip(
          train_set, bucket_id, pos, vocab)

      if epoch_flag:
        epochs += 1
        own_prob = own_prob * decrease_prob
        print("The number of epochs is:{}".format(epochs))
        print ("The step_loss is: ",float(step_loss))
        print("Actual Reward: ", final_reward)
        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print ("Current Time: "+cur_time)
        print ("Current Learning rate: ", sess.run(model.learning_rate))
        if epochs%save_at == 0:
          # Save checkpoint and zero timer and loss.
          # checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
          checkpoint_path = os.path.join("./checkpoints","policy_model_full_genre68_admire_"+str(run_num)+"_epoch{}.ckpt".format(epochs))
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          print ("MODEL SAVED")
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]) and epochs%50==0:
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        
      reward = None
      if epochs==0 and fl==False:
        init_new_vars_op = tf.initialize_variables([model.global_step,model.learning_rate])
        sess.run(init_new_vars_op)
        fl = True

      _, step_loss, selected_actions = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True, False)


      # print(np.asarray(selected_actions).shape)
      # exit(0)
      # predictions = tf.nn.softmax(np.asarray(selected_actions))
      # tf.Print(predictions,[predictions],message="This is a: ")
      # print(np.sum(np.asarray(new_p[3,1,:])))

      # exit(0)

      _, pretrained_step_loss, pretrained_selected_actions = pretrained_model.step(sess, encoder_inputs, decoder_inputs,
                             target_weights, bucket_id, True, False)
    
      language_model = sess.run(predictions,{placeholder_3 : np.asarray(pretrained_selected_actions)})
      # for i in xrange(FLAGS.batch_size):
      #   write_str=",".join([inv_vocab[o[i]] for o in encoder_inputs])
      #   log_file.write("Encoder Inputs: " + str(write_str))
      #   log_file.write("\n")
      log_file.write("\n")
      reward = 1
      reward1 = 1
      keep_prob = 1

      
      outputs = np.zeros(shape = (5,FLAGS.batch_size), dtype = np.int)
      chained_output = np.zeros(shape = (2,5), dtype = np.int)
      encoder_inputs1 = encoder_inputs
      decoder_inputs1 = decoder_inputs
      if epochs>2:
        if epochs%30==1:
          if epochs_flag == False:
            div = div + 1
            epochs_flag = True
            # print (div)
        else:
          epochs_flag = False

      pos_incrementer = 0
      ### NEW CHANGES START ###
      ## We will create the outputs/targets for the entire batch here

      for i in xrange(FLAGS.batch_size):
      #   write_str=",".join([inv_vocab[o[i]] for o in encoder_inputs])
      #   log_file.write("Encoder Inputs: " + str(write_str))
      #   log_file.write("\n")


        # Sample from the prev_verbs list
        if False:
          # log_file.write("Sampling from the distribution with verb replacement")
          # log_file.write("\n")
          # for j in xrange(5):
          #   if j==4:
          #     outputs[j][i] = data_utils.EOS_ID
          #   else:
          #     k = np.array(selected_actions[j][i][:]).reshape(1,FLAGS.cmu_vocab_size)
          #     outputs[j][i] = sess.run(samples,{placeholder : k})[0][0]
          #     if j==2 and outputs[j][i]==2:
          #       k2 = np.array(k[0][2:]).reshape(1, FLAGS.cmu_vocab_size-2)
          #       outputs[j][i] = sess.run(samples_2,{placeholder_2: k2})[0][0] + 2
          #     if j==3 and outputs[j][i]==1:
          #       k2 = np.array(k[0][2:]).reshape(1, FLAGS.cmu_vocab_size-2)
          #       outputs[j][i] = sess.run(samples_2,{placeholder_2: k2})[0][0] + 2

          # c_num = cluster_number(inv_vocab[encoder_inputs[2][i]],clusters)
          # if c_num==1:
          #   outputs[1][i] = int(vocab.index(target_verb))
          # elif c_num==0:
          #   outputs[1][i] = int(vocab.index(clusters[len(clusters)][randint(0,len(clusters[len(clusters)])-1)]))
          # else:
          #   outputs[1][i] = int(vocab.index(clusters[c_num-1][randint(0,len(clusters[c_num-1])-1)]))

          reward1 = 0            

          if encoder_counter%6==1:
            # log_file.write("Argmax with encoder replacement")
            # log_file.write("\n")
            for j in xrange(4):
              outputs[j][i] = decoder_inputs[j+1][i]
            outputs[4][i] = data_utils.EOS_ID
            # for m in xrange(5):
            #   if m==4:
            #     outputs[m][i] = data_utils.EOS_ID
            #   else:
            #     outputs[m][i] = np.argmax(pretrained_selected_actions[m][0][:])
            #   if m!=4:
            #     if inv_vocab[outputs[m][i]] not in word_dict[m]:
            #       outputs[m][i] = int(vocab.index(word_dict[m][randint(0,len(word_dict[m])-1)]))
            cluster_num = randint(0,len(clusters)-1)
            if cluster_num == len(clusters)-1:
              outputs[1][i] = int(vocab.index(target_verb))
              # outputs[1][i] = int(vocab.index(clusters[len(clusters)-1][randint(0,len(clusters[len(clusters)-1])-1)]))
              encoder_inputs[2][i] = int(vocab.index(clusters[cluster_num][randint(0,len(clusters[cluster_num])-1)]))
            else:
              if len(clusters[cluster_num]) == 1:
                encoder_inputs[2][i] = int(vocab.index(clusters[cluster_num][0]))
              else:
                encoder_inputs[2][i] = int(vocab.index(clusters[cluster_num][randint(0,len(clusters[cluster_num])-1)]))
              if len(clusters[cluster_num+1]) == 1:
                outputs[1][i] = int(vocab.index(clusters[cluster_num+1][0]))
              else:
                outputs[1][i] = int(vocab.index(clusters[cluster_num+1][randint(0,len(clusters[cluster_num+1])-1)]))
            encoder_counter+=1
          else:
            # log_file.write("Argmax from the pretrained model with replacement")
            # log_file.write("\n")
            for j in xrange(4):
              outputs[j][i] = decoder_inputs[j+1][i]
            outputs[4][i] = data_utils.EOS_ID
            # for m in xrange(5):
            #     if m==4:
            #       outputs[m][i] = data_utils.EOS_ID
            #     else:
            #       outputs[m][i] = np.argmax(pretrained_selected_actions[m][0][:])
            #     if m!=4:
            #       if inv_vocab[outputs[m][i]] not in word_dict[m]:
            #         outputs[m][i] = int(vocab.index(word_dict[m][randint(0,len(word_dict[m])-1)]))
            c_num = cluster_number(inv_vocab[encoder_inputs[2][i]],clusters)
            string = []
            # for j in xrange(5):
            #     string.append(inv_vocab[outputs[j][i]])
            # print (string)
            # print(inv_vocab[encoder_inputs[2][i]])
            # print(c_num)            
            if c_num==(len(clusters)-1):
              outputs[1][i] = int(vocab.index(target_verb))
              # outputs[1][i] = int(vocab.index(clusters[len(clusters)-1][randint(0,len(clusters[len(clusters)-1])-1)]))
            elif c_num==-1:
              if len(clusters[0]) == 1:
                outputs[1][i] = int(vocab.index(clusters[0][0]))
              else:  
                outputs[1][i] = int(vocab.index(clusters[0][randint(0,(len(clusters[0])-1))]))
            else:
                outputs[1][i] = int(vocab.index(clusters[c_num+1][randint(0,len(clusters[c_num+1])-1)]))
                # print(inv_vocab[outputs[1][i]])
            # ll = []
            # for m in xrange(5):
            #   ll.append(inv_vocab[outputs[m][0]])
            # print(ll)
            encoder_counter+=1
          # print (inv_vocab[outputs[1][i]])
          # print (inv_vocab[encoder_inputs[2][i]])
        else:
          # log_file.write("Sampling from the distribution of next words")
          # log_file.write("\n")
          for j in xrange(4):
            outputs[j][i] = decoder_inputs[j+1][i]
          outputs[4][i] = data_utils.EOS_ID
          c_num = cluster_number(inv_vocab[encoder_inputs[2][i]],clusters)
          new_c_num = cluster_number(inv_vocab[decoder_inputs[2][i]],clusters)
          if c_num==(len(clusters)-1):
            outputs[1][i] = int(vocab.index(target_verb))
            # outputs[1][i] = int(vocab.index(clusters[len(clusters)-1][randint(0,len(clusters[len(clusters)-1])-1)]))
          elif c_num==-1:
            if len(clusters[0]) == 1:
              outputs[1][i] = int(vocab.index(clusters[0][0]))
            else:  
              outputs[1][i] = int(vocab.index(clusters[0][randint(0,(len(clusters[0])-1))]))
          else:
            if (new_c_num - c_num) !=1: 
              string = []
              cluster_prob = defaultdict(int)
              # for p in range(len(language_model[1,i,:])):
              #   language_model[1,i,p] = language_model[1,i,p] + 0.1
              # sum_prob = np.sum(language_model[1,i,:])
              # for p in range(len(language_model[1,i,:])):
              #   language_model[1,i,p] = language_model[1,i,p]/sum_prob
              for cn in range(len(clusters[c_num+1])):
                cluster_prob[clusters[c_num+1][cn]] = float(language_model[1,i,:][vocab.index(clusters[c_num+1][cn])])
              max_c_prob = max(cluster_prob.values())
              min_c_prob = min(cluster_prob.values())
              sum_c_prob = np.sum(cluster_prob.values())
              # print(cluster_prob)
              # exit(0)
              # print(sum_c_prob)
              # print(float(sum_c_prob))
              for word in cluster_prob.keys():
                cluster_prob[word] = float(cluster_prob[word])/(float(sum_c_prob))
              # print(cluster_prob)
              # print(cluster_prob.values())
              # print(np.sum(np.asarray(cluster_prob.values())))
              next_verb = np.random.choice(len(cluster_prob.keys()),1, p=cluster_prob.values())
              # print(inv_vocab[encoder_inputs[2][i]])
              # print(cluster_prob.keys()[next_verb[0]])
              # print("----------------------------------")
              outputs[1][i] = int(vocab.index(cluster_prob.keys()[next_verb[0]]))
              # strs = []
              # print(cluster_prob)
              # for l in range(5):
              #   strs.append(inv_vocab[outputs[l][i]])
                # strs.append(",")
              # print(strs)
              encoder_counter+=1
            # else:
              # print("else")
              # print(inv_vocab[encoder_inputs[2][i]])
              # print(inv_vocab[outputs[1][i]])
              # print("---------------------------------")

      # sum = 0
      horizontalSum = 0
      secondPrev, thirdPrev = get_previous_lines(pos,train_set)
      final_reward = 0
      break_flag = False
      train_flag = True
      for i in range(FLAGS.batch_size):
        # sum = 0
        for p in xrange(4):
          if outputs[p][i] != int(vocab.index('<start>')) and outputs[p][i] != int(vocab.index('<end>')) and encoder_inputs[3-p][i] != int(vocab.index('<start>')) and encoder_inputs[3-p][i] != int(vocab.index('<end>')):
            if inv_vocab[outputs[p][i]] not in word_dict[p]:
              log_file.write("Invalid token positioning")
              log_file.write("\n")
              log_file.write(str(i))
              log_file.write("\n")
              # sum = 1
              reward = 1
              train_flag = False
              for j in xrange(4):
                outputs[j][i] = decoder_inputs[j+1][i]
              outputs[4][i] = data_utils.EOS_ID
              break_flag = True
              break
            # sum += posStats[p][positionDict[p][inv_vocab[outputs[p][i]]]][positionDict[p][inv_vocab[encoder_inputs[3-p][i]]]]*individualPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]]
            # if secondPrev is not None:
            #   sum += 0.67*secondPositionStats[p][positionDict[p][inv_vocab[outputs[p][i]]]][positionDict[p][inv_vocab[secondPrev[p]]]]*individualPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]]
            # if thirdPrev is not None:
            #   sum += 0.33*thirdPositionStats[p][positionDict[p][inv_vocab[outputs[p][i]]]][positionDict[p][inv_vocab[thirdPrev[p]]]]*individualPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]]
            # if p!=0:
            #   horizontalSum += horizonPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]][positionDict[p-1][inv_vocab[outputs[p-1][i]]]]*individualPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]]
            # if p==2:
            #   horizontalSum += 0.5*secondHorizonPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]][positionDict[p-2][inv_vocab[outputs[p-2][i]]]]*individualPosStats[p][positionDict[p][inv_vocab[outputs[p][i]]]]
          else:
            log_file.write("Found start, shouldn't happen")
            log_file.write("\n")
            # sum = 1
            reward = 1
            train_flag = False
            for j in xrange(4):
              outputs[j][i] = decoder_inputs[j+1][i]
            outputs[4][i] = data_utils.EOS_ID
            break_flag = True
            break


        # if break_flag==False:
        #   if inv_vocab[outputs[1][i]] == target_verb:
        #     reward += math.ceil(max_reward)
        #     # reward = math.ceil(max_reward) * (rollout_num+1)
        #     # print(reward)
        #   else:
        #     new_encoder_inputs = list(np.zeros(np.array(encoder_inputs[:][i]).shape))
        #     new_decoder_inputs = list(np.zeros(np.array(decoder_inputs[:][i]).shape))
        #     reward += math.ceil(max_reward) - word_dist[inv_vocab[outputs[1][i]]]
        #     for j in range(rollout_num):
        #       outputs1 = np.zeros(shape = (5,FLAGS.batch_size), dtype = np.int)
        #       for l in xrange(4):
        #         new_encoder_inputs[3-l][i] = outputs[l][i]
        #       for l in xrange(4):
        #         new_decoder_inputs = decoder_inputs
        #       _, step_loss1, selected_actions1 = model.step(sess, new_encoder_inputs, new_decoder_inputs,
        #                              target_weights, bucket_id, True, False)
        #       for m in xrange(5):
        #         if m==4:
        #           outputs1[m][i] = data_utils.EOS_ID
        #         else:
        #           outputs1[m][i] = np.argmax(selected_actions1[m][0][:])
        #           # k = np.array(selected_actions1[m][0][:]).reshape(1,FLAGS.cmu_vocab_size)
        #           # outputs1[m][0] = sess.run(samples,{placeholder : k})[0][0]
        #           # if j==2 and outputs1[m][0]==2:
        #           #   k2 = np.array(k[0][2:]).reshape(1, FLAGS.cmu_vocab_size-2)
        #           #   outputs1[m][0] = sess.run(samples_2,{placeholder_2: k2})[0][0] + 2
        #           # if j==3 and outputs1[m][0]==1:
        #           #   k2 = np.array(k[0][2:]).reshape(1, FLAGS.cmu_vocab_size-2)
        #           #   outputs1[m][0] = sess.run(samples_2,{placeholder_2: k2})[0][0] + 2

        #       # for m in xrange(5):
        #       #   print (inv_vocab[outputs1[m][0]])
        #       # exit(0)
        #       if inv_vocab[outputs1[1][0]] != target_verb:
        #         reward += math.pow(gamma,j+1)*(math.ceil(max_reward) - word_dist[inv_vocab[outputs1[1][0]]])
        #       else:
        #         reward += math.pow(gamma,j+1)*(math.ceil(max_reward))*(rollout_num - (j))
        #         break
        #     reward /= math.ceil(max_reward) * (rollout_num+1)


        if break_flag==False:
          if inv_vocab[outputs[1][i]] == target_verb:
            old_c_num = cluster_number(inv_vocab[encoder_inputs[2][i]],clusters)
            # new_c_num = cluster_number(inv_vocab[outputs[1][i]],clusters)
            # print (old_c_num)
            # print (inv_vocab[encoder_inputs[2][i]])
            if (old_c_num) == (len(clusters)-1):
              reward = math.ceil(max_reward)
              # reward = 1
            else:
              reward = 0
              print("inside")
              # for j in xrange(4):
              #   outputs[j][i] = decoder_inputs[j+1][i]
              # outputs[4][i] = data_utils.EOS_ID
              # train_flag = False
            # print(reward)
          else:
            old_c_num = cluster_number(inv_vocab[encoder_inputs[2][i]],clusters)
            new_c_num = cluster_number(inv_vocab[outputs[1][i]],clusters)
            # print (new_c_num)
            # print (old_c_num)
            if (new_c_num - old_c_num) == 1:
              reward = word_dist[inv_vocab[outputs[1][i]]]
              # reward = math.ceil(max_reward)
              # reward = 1
            else:
              reward = 0
              print("inside")
              # for j in xrange(4):
              #   outputs[j][i] = decoder_inputs[j+1][i]
              # outputs[4][i] = data_utils.EOS_ID
              # train_flag = False
        # print(reward)

        # if sum == 0 and horizontalSum == 0:
        #   log_file.write("Vertical and horizontal sum are zero")
        #   log_file.write("\n")
        #   reward = 0

        # if reward>-2:
        #   # horizontalSum *= 2
        #   # sum = (horizontalSum*sum)/(horizontalSum+sum)
        #   # reward = reward*sum
        #   log_file.write("Normal reward calculation")
        #   log_file.write("\n")
        if outputs[2][i]==outputs[3][i] and reward!=-1 and outputs[2][i]!=int(vocab.index('<start>')) and outputs[2][i]!=int(vocab.index('<end>')): 
          # log_file.write("last two tokens are the same")
          # log_file.write("\n")
          reward = reward/100     
        # print(reward)
           
        # print(clusters)
        # exit(0)
        final_reward += reward
      final_reward /= FLAGS.batch_size
      # print(final_reward)
      reward_flag = False
      # if final_reward > -2:
      lll = []
      # final_reward = 0
      # for m in xrange(5):
      #   if m==4:
      #     outputs[m][i] = data_utils.EOS_ID
      #   else:
      #     outputs[m][i] = np.argmax(selected_actions[m][0][:])
      # for m in xrange(5):
      #   lll.append(inv_vocab[outputs[m][0]])
      # print (lll)
      # lll = []
      # for m in xrange(4):
      #   lll.append(inv_vocab[decoder_inputs[m+1][0]])
      # print("decoder")
      # print(lll)
      # for i in xrange(FLAGS.batch_size):
      #   write_out=",".join([inv_vocab[o[i]] for o in outputs])
      #   log_file.write("Sampled Output: "+str(write_out))
      #   log_file.write("\n")
      # log_file.write("The actual Reward is: "+str(final_reward)+"\n\n")
      outputs_model = np.zeros(shape = (5,FLAGS.batch_size), dtype = np.int)
      for i in range(FLAGS.batch_size):
        for j in xrange(5):
          if j==4:
            outputs_model[j][i] = data_utils.EOS_ID
          else:
            k = np.array(selected_actions[j][i][:]).reshape(1,FLAGS.cmu_vocab_size)
            outputs_model[j][i] = sess.run(samples,{placeholder : k})[0][0]
      # log_file.write("Distribution of the model")
      # log_file.write("\n")
      # for i in range(FLAGS.batch_size):
      #   write_out=",".join([inv_vocab[o[i]] for o in outputs_model])
      #   log_file.write("Sampled Output: "+str(write_out))
      #   log_file.write("\n")
      reward_flag = True
      new_train_set = read_data.read_policy_data_set(encoder_inputs, outputs, _buckets, FLAGS.batch_size, vocab)
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(new_train_set, bucket_id, 0)

      # else:
      #   for i in xrange(FLAGS.batch_size):
      #     write_out=",".join([inv_vocab[o[i]] for o in decoder_inputs])
      #     log_file.write("Decoder Output: "+str(write_out))
      #     log_file.write("\n")
      #   for i in xrange(FLAGS.batch_size):
      #     write_out=",".join([inv_vocab[o[i]] for o in outputs])
      #     log_file.write("Rejected Sampled Output: "+str(write_out))
      #     log_file.write("\n")
      #   log_file.write("The actual Reward is: "+str(final_reward)+"\n\n")
      # final_reward = 1
      if train_flag:
        _, step_loss,selected_actions,_ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False, reward_flag = reward_flag, vocab=vocab, inv_vocab=inv_vocab,
                                     session_inner=sess_inner, reward = final_reward)
      # log_file.write("Loss: " + str(step_loss))
      # log_file.write("\n")
      # log_file.write("------------------------------------------------------")
      # log_file.write("\n")
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # pos = (pos + FLAGS.batch_size) % len(train_set[bucket_id])
      
      if epochs%50==0: 
        if flag==False:
          word_number = word_number%number_of_words + 1
          flag = True
      else:
        flag = False   
      
        
      #figure out how to convert decode function for the learnt model and not the forward model alone. 
      #decode()
    output = []
    
def decode():
  with tf.Session() as sess, tf.variable_scope("forward"):
    # Create model and load parameters.
    model = create_model(sess, True, "checkpoint_forward_new_2_250", "forward")
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.en" % FLAGS.cmu_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.fr" % FLAGS.cmu_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = 0
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()

