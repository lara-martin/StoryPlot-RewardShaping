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
import random
import sys
import time
# import read_data_genre68_full as read_data
# import seq2seq_model_fb as seqI opened this to make that change
# import seq2seq_model_policy_v49_reduced_meet as seq2seq_model_meet
# import seq2seq_model_policy_v1_genre68_full_admire as seq2seq_model_admire
# import seq2seq_model_policy_v49_reduced_marry as seq2seq_model_marry

# import translate_policy_decode_reduced_meet as decode_meet
# import translate_policy_decode_reduced_marry as decode_marry
# import translate_policy_decode_reduced_admire_new as decode_admire

import pickle

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#use 
# import data_utils_genre68_full as data_utils


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("cmu_vocab_size", 3548, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
# tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
# tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/data", "Data directory")
#tf.app.flags.DEFINE_string("train_dir", "/home/twister/Desktop/GALE_RL/Storytelling_Run_MN/storytelling_dl/Neural-Net-Experiments/Spring2017-Attempt/NewEvents-PolicyGradient/Reduced/checkpoints/", "Training directory.")
tf.app.flags.DEFINE_string("train_dir","../Full_Genre_68/checkpoints/", "Training directory.")
tf.app.flags.DEFINE_string("train_file", "editted_Parsed_events_train_reduced.txt", "Training file.")
tf.app.flags.DEFINE_string("test_file", None, "Testing file.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", True,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("max_epochs", 1000, "Maximum number of epochs.")

FLAGS = tf.app.flags.FLAGS

# FLAGS.max_epochs=2000

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(4,5)]

# Call this function with appropriate checkpoint filename to create the model and store the model in a variable
def create_model(session, forward_only, checkpoint):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  print("before")
  model = seq2seq_model_admire.Seq2SeqModel(
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
      dtype=dtype)
  print("inside iourgabg")
  # ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir,"checkpoint_forward_new_2_250")
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir, checkpoint)
  print("CKPT is:",ckpt)
  print(ckpt.model_checkpoint_path)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    print("Finished loading saved model")
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

# Instead of using the function below - Use the model agnostic one that follows after this!!!
# This function will calculate how many times admire was reached and how many lines it took to get there
# Input parameters - the checkpoint filename for the forward model and DRL model.
# And the file which contains all the events that will be fed in (eg: test event file)
# Highly suggest passing in a name for the log file (the path)
def get_admire_stats(forward_model_ckpt, drl_ckpt, input_event_file, vocab_file='Final_genre68_full.vocab', log_file_name='genre_68_eval_stats_logs'):
  cfg = tf.ConfigProto()
  cfg.gpu_options.allow_growth = True
  with tf.Session(config = cfg) as sess:
    # Create model and load parameters.
    log_file = open(log_file_name, "wb")
    samples = tf.placeholder(tf.float32, shape=())
    placeholder = tf.placeholder(tf.float32,shape = (1,FLAGS.cmu_vocab_size))
    samples = tf.multinomial(placeholder,1)
    sess.run(tf.global_variables_initializer())
    print("\nLoading forward model (pretrained)\n")
    model_forward = create_model(sess, True, forward_model_ckpt)
    print("\nLoading DRL model\n")
    model_DRL = create_model(sess, True, drl_ckpt)
    model.batch_size = 1  # We decode one sentence at a time.

    # Vocab Path and Creating the dictionary
    vocab_file = open(vocab_file, 'rb')
    vocab = pickle.load(vocab_file)
    vocab_file.close()

    inv_vocab = {k: v for k, v in enumerate(vocab)}

    # to_admire_lines will only get a value added when the story actually got to admire
    # So for the drl model the avg num of lines it took to get to admire would be drl_to_admire_lines/drl_got_admire_count
    # Similar thing for the forward model
    num_lines = 0; fwd_got_admire_count = 0; fwd_to_admire_lines = 0; drl_got_admire_count = 0; drl_to_admire_lines = 0;
    # max_count is the number of lines you want to keep as the upper limit before calling it a day :P
    max_count = 10
    
    # Read one event at a time and process it to generate stories
    with open(input_event_file, "rb") as in_file:
      for event in in_file:
        write_line = "\nFor event:\n"+event+"\n"
        # First do the calculation for the forward model - pass the correct model into the function
        got_admire, count_val = decode_event(sess, event, model_forward, max_count, vocab, inv_vocab, samples, placeholder)
        if got_admire:
          fwd_got_admire_count+=1
          fwd_to_admire_lines+=count_val
        write_line += "Forward model:\n"+str(got_admire)+"\n"+str(count_val)+"\n"

        # Now, do the same for the DRL model
        got_admire, count_val = decode_event(sess, event, model_DRL, max_count, vocab, inv_vocab, samples, placeholder)
        if got_admire:
          drl_got_admire_count+=1
          drl_to_admire_lines+=count_val

        write_line += "DRL model:\n"+str(got_admire)+"\n"+str(count_val)+"\n"
        log_file.write(write_line)

        num_lines+=1

    sess.close()
    log_file.close()
# This function will calculate how many times admire was reached and how many lines it took to get there
# Input parameters - the checkpoint filename for the current model that you want to evaluate.
# And the file which contains all the events that will be fed in (eg: test event file)
# Highly suggest passing in a name for the log file (the path)
def get_model_common_admire_stats(model_ckpt, input_event_file, vocab_file='Final_genre68_full.vocab', log_file_name='genre_68_eval_stats_logs'):
  cfg = tf.ConfigProto()
  cfg.gpu_options.allow_growth = True
  with tf.Session(config = cfg) as sess:
    # Create model and load parameters.
    log_file = open(log_file_name, "wb")
    samples = tf.placeholder(tf.float32, shape=())
    placeholder = tf.placeholder(tf.float32,shape = (1,FLAGS.cmu_vocab_size))
    samples = tf.multinomial(placeholder,1)
    sess.run(tf.global_variables_initializer())
    print("\nLoading the model\n")
    model = create_model(sess, True, model_ckpt)
    model.batch_size = 1  # We decode one sentence at a time.

    # Vocab Path and Creating the dictionary
    vocab_file = open(vocab_file, 'rb')
    vocab = pickle.load(vocab_file)
    vocab_file.close()

    inv_vocab = {k: v for k, v in enumerate(vocab)}

    # to_admire_lines will only get a value added when the story actually got to admire
    # So for the drl model the avg num of lines it took to get to admire would be drl_to_admire_lines/drl_got_admire_count
    # Similar thing for the forward model
    num_lines = 0; got_admire_count = 0; to_admire_lines = 0;
    # max_count is the number of lines you want to keep as the upper limit before calling it a day :P
    max_count = 10
    
    # Read one event at a time and process it to generate stories
    with open(input_event_file, "rb") as in_file:
      for event in in_file:
        write_line = "\nFor event:\n"+event+"\n"
        # print(event)
        # print(event=='<end_of_story>')
        # print(event in '<end_of_story>')
        # print('<end' in event)

        if '<end' in event or '<start' in event:
          continue
        # First do the calculation for the forward model - pass the correct model into the function
        else:
          # print(sess, event, model, max_count, samples, placeholder)
          # print(decode_event)
          got_admire, count_val = decode_event(sess, event, model, max_count, vocab, inv_vocab, samples, placeholder)
          if got_admire:
            got_admire_count+=1
            to_admire_lines+=count_val
          write_line += "For current model:\n"+str(got_admire)+"\n"+str(count_val)+"\n"
          log_file.write(write_line)
          num_lines+=1

    sess.close()

    # Write the results before returning the values
    write_line = "For the model - "+str(model_ckpt)+". The results are:\n"
    write_line += "Got count:"+str(got_admire_count)+"\nTo admire lines:"+str(to_admire_lines)+"\nTotal lines:"+str(num_lines)+"\n"
    write_line += "Average count to admire: " + str(float(to_admire_lines)/got_admire_count)
    write_line += "\nAverage percentage of getting to admire: " + str(float(got_admire_count)/num_lines)
    log_file.write(write_line)
    log_file.close()
    #Percentage of lines that get to admire. (got_admire )

    print ("Completed processing the ckpt - "+model_ckpt)
    print ("Got count:"+str(got_admire_count)+"\nTo admire lines:"+str(to_admire_lines)+"\nTotal lines:"+str(num_lines)+"\n")
  return got_admire_count, to_admire_lines, num_lines

def decode_event(sess,cur_event,model,cur_max,vocab,inv_vocab,samples,placeholder):
  """Create translation model and initialize or load parameters in session."""
  lines_generated = 0
  story_flag = False
  cur_event=cur_event.strip('\n')
  token_ids = [int(vocab.index(word)) for word in cur_event.split(',')]
  while lines_generated <= cur_max:
    if token_ids[1]==int(vocab.index("admire-31.2")):
      story_flag = True  
      break
    bucket_id=0
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [data_utils.GO_ID,data_utils.GO_ID,data_utils.GO_ID,data_utils.GO_ID,data_utils.GO_ID])]}, bucket_id, 0)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True, False)
    outputs = np.zeros(shape = (4,FLAGS.batch_size), dtype = np.int)
    for i in xrange(FLAGS.batch_size):
      for j in xrange(4):
        k = np.array(output_logits[j][i][:]).reshape(FLAGS.batch_size,FLAGS.cmu_vocab_size)
        outputs[j][i] = sess.run(samples,{placeholder : k})[0][0]
        # outputs[j][i] = np.argmax(output_logits[j][i][:])
    for i in xrange(FLAGS.batch_size):
      for j in xrange(4):
        token_ids[j] = outputs[j][i]
    lines_generated+=1
  return story_flag,lines_generated

def calculate_corpus_stats_setting_1(filename):
  lines_to_admire=0; num_stories=0; stories_got_admire=0; stories_got_admire_non_avg = 0; admire_flag=False
  with open(filename,"rb") as f:
    lines_counter=0; # This will keep resetting
    for line in f:
      if line.strip()=="<start_of_story>":
        # print("Going in here (start) -"+line.strip())
        lines_counter=0
        admire_flag=False
      elif line.strip()=="<end_of_story>":
        # print("Going in here (end) -"+line.strip())
        lines_counter=0
        num_stories+=1
      elif "admire-31.2" in line:
        if not admire_flag:
          stories_got_admire_non_avg+=1
        admire_flag=True
        stories_got_admire+=1
        lines_to_admire+=lines_counter
        lines_counter+=1
      else:
        lines_counter+=1
  print(lines_to_admire, stories_got_admire, stories_got_admire_non_avg, num_stories)

def calculate_corpus_stats_setting_2(filename):
  lines_to_admire=0; num_stories=0; stories_got_admire=0; stories_got_admire_non_avg = 0; admire_flag=False
  with open(filename,"rb") as f:
    lines_counter=0; # This will keep resetting
    for line in f:
      if line.strip()=="<start_of_story>":
        # print("Going in here (start) -"+line.strip())
        lines_counter=0
        admire_flag=False
      elif line.strip()=="<end_of_story>":
        # print("Going in here (end) -"+line.strip())
        lines_counter=0
        num_stories+=1
      elif "admire-31.2" in line:
        if not admire_flag:
          stories_got_admire_non_avg+=1
        admire_flag=True
        stories_got_admire+=1
        lines_to_admire+=lines_counter
        lines_counter=0
      else:
        lines_counter+=1
  print(lines_to_admire, stories_got_admire, stories_got_admire_non_avg, num_stories)

# In this setting consider each line as an input line (so num_stories is equal to total num of lines in the corpus)
# If admire doesn't occur in max_lines then move on to the next line in the corpus
def calculate_corpus_stats_setting_3(filename, max_lines):
  num_stories=0; lines_to_admire=0; got_admire_count=0;
  with open(filename,"rb") as f:
    # first need to copy the whole file in a list - to keep looking forward
    full_file=[]
    for line in f:
      full_file.append(line.strip())

  for i, line in enumerate(full_file):
    admire_flag=False
    for j in xrange(i+1, i+max_lines+1):
      if j>=len(full_file):
        break
      if "admire-31.2" in full_file[j]:
        admire_flag=True
        break
    if admire_flag:
      # print (i,j)
      lines_to_admire+=j-i
      got_admire_count+=1
  num_stories=i+1
  return num_stories, got_admire_count, lines_to_admire

def main(_):
  # print("Inside")
  # decode()
  # Need to call the function get_model_common_admire_stats() with the right parameters - Check the definition of the function
  # Make two calls to the function - 1. With forward_ckpt and 2. With DRL ckpt
  # get_model_common_admire_stats("checkpoint_full_genre68","final_test_genre68_full.txt")
  # get_model_common_admire_stats("checkpoint_forward_genre68_full_run_1","final_test_genre68_full.txt",log_file_name='Run-Logs/genre_68_eval_Forward_run_1_stats_logs')
  # calculate_corpus_stats("one_story.txt")
  # calculate_corpus_stats_setting_1("final_combined_genre68_full.txt")
  print (calculate_corpus_stats_setting_3("final_combined_genre68_full.txt",30))

if __name__ == "__main__":
  # Look at the comments in the main above this main :P - Implement those things
  # decode()
  tf.app.run()

