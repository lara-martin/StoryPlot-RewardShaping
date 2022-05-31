def train():
  #Setting Up Training
  with tf.Session() as sess, tf.Session() as sess_inner:
	# Create model.
	target_verb = FLAGS.goal

	placeholder_3 = tf.placeholder(tf.float32,shape = (6,FLAGS.batch_size,FLAGS.cmu_vocab_size)) #########This was already 5
	predictions = tf.nn.softmax(placeholder_3)

	sess.run(tf.global_variables_initializer())
	sess_inner.run(tf.global_variables_initializer())

	print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
	bucket_id = 0
	
	#Load seq2seq model
	model = create_model(sess, False,"checkpoint_forward_SCIFI")
	pretrained_model = model

	#Setup vocab
	vocab_file = open(FLAGS.vocab_file, 'rb')
	vocab = pickle.load(vocab_file)
	vocab_file.close()

	inv_vocab = {k: v for k, v in enumerate(vocab)}

	#Read in data
	val_set = read_data.read_with_buckets(FLAGS.val_file, vocab,_buckets)
	train_set = read_data.read_with_buckets(FLAGS.train_file, vocab,_buckets)

	#params = tf.trainable_variables()

	loss = 0.0
	pos = 0
	epochs = 0
	
	#####################################

	#Calculating word clustering
	word_dict_statistics = {}
	word_dict_statistics = load_obj('words_for_sampling')

	word_dict = {}
	word_dict = load_obj('words_for_sampling')

	print("word_dict", word_dict)

	posStats,individualPosStats,secondPositionStats,thirdPositionStats,positionDict = statistics(word_dict_statistics,FLAGS.combined_file)
	horizonPosStats, secondHorizonPosStats = statistics_horizontal(word_dict_statistics, FLAGS.combined_file)
	#words that occur before goal in a story
	
	word_dist = word_distance(FLAGS.combined_file,word_dict[1],str(FLAGS.goal))
	clusters = word_clustering(word_dist,FLAGS.len_clusters,str(FLAGS.goal))

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


	clusters = word_clustering(word_dist,FLAGS.len_clusters,FLAGS.goal)
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
	for cluster in clusters:
	  for word in clusters[cluster]:
		sum += word_dist[word]
		if word_dist[word]>max_reward:
		  max_reward = word_dist[word]
		num += 1
	  # print (float(sum)/num)
	  sum = 0
	  num = 0

	word_dist[FLAGS.goal] = 0.97
	print(word_dist)

	#####################################

	#Training
	epoch_reward = 0
	#epoch_forward_loss = 0
	epoch_backward_loss = 0
	batch_count = 0
	while epochs < FLAGS.max_epochs:
	  bucket_id = 0
	  # Get a batch and make a step.
	  encoder_inputs, decoder_inputs, target_weights, pos, epoch_flag = model.get_batch_skip(train_set, bucket_id, pos, vocab)

	  if epoch_flag:
		v_loss, v_reward = checkValidation(sess, sess_inner, model, val_set, vocab, inv_vocab, word_dict, clusters, word_dist, max_reward)
		print("The validation loss is: ", v_loss)
		print("Validation Reward: ", v_reward)
		epochs += 1
		print("The number of epochs is:{}".format(epochs))
		#print ("The step_loss is: ",float(step_loss))
		#print("The forward loss is: ", float(epoch_forward_loss/batch_count))
		print("The training loss is: ", float(epoch_backward_loss/batch_count))
		print("Actual Reward: ", epoch_reward / batch_count)
		epoch_reward = 0
		batch_count = 0
		#epoch_forward_loss = 0
		epoch_backward_loss = 0
		cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		print ("Current Time: "+cur_time)
		print ("Current Learning rate: ", sess.run(model.learning_rate))

		if epochs%save_at == 0:
		  # Save checkpoint and zero timer and loss.
		  checkpoint_path = os.path.join("./checkpoints","policy_model_full_SCIFI_"+goal_word+"_"+str(run_num)+"_epoch{}.ckpt".format(epochs))
		  model.saver.save(sess, checkpoint_path, global_step=model.global_step)
		  print ("MODEL SAVED")

		sess.run(model.learning_rate_decay_op)
		#previous_losses.append(step_loss)
	  
	  #Initialize variables
	  if epochs==0: #and fl==False:
		init_new_vars_op = tf.initialize_variables([model.global_step,model.learning_rate])
		sess.run(init_new_vars_op)
		#fl = True

	  #####################################

	  #Forward run of the DRL model
	  _, step_loss, selected_actions = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, False)
	  #epoch_forward_loss+= step_loss
	  #print("encoder_inputs", encoder_inputs)
	  #print("decoder_inputs", decoder_inputs)
 
	  #Convert log probabilities to softmax probabilities 
	  language_model = sess.run(predictions,{placeholder_3 : np.asarray(selected_actions)})
	  
	  #####################################

	  #Resample
	  #Set 'outputs' variable equal to the decoder input
	  outputs = np.zeros(shape = (6,FLAGS.batch_size), dtype = np.int) ####### was 5

	  for i in range(0,FLAGS.batch_size):
		for j in range(0,5): ####### was 4
		  outputs[j][i] = decoder_inputs[j+1][i]
		outputs[5][i] = data_utils.EOS_ID ####### was 4
		#print(str(encoder_inputs[0][i])+ " "+str(encoder_inputs[1][i])+ " "+str(encoder_inputs[2][i]) + " "+str(encoder_inputs[3][i]) +" "+ str(encoder_inputs[4][i]))
		#print(inv_vocab[encoder_inputs[0][i]]+ " "+inv_vocab[encoder_inputs[1][i]] + " "+inv_vocab[encoder_inputs[2][i]] + " "+inv_vocab[encoder_inputs[3][i]] +" "+ inv_vocab[encoder_inputs[4][i]])

		#Resample the verb based on the current distribution of the DRL model.
		c_num = cluster_number(inv_vocab[encoder_inputs[3][i]],clusters) ###### was 2
		new_c_num = cluster_number(inv_vocab[decoder_inputs[2][i]],clusters)
		if c_num==FLAGS.len_clusters: #Last cluster is goal
		  outputs[1][i] = int(vocab.index(target_verb))
		elif c_num==-1:
		  if len(clusters[0]) == 1:
			outputs[1][i] = int(vocab.index(clusters[0][0]))
		  else:  
			outputs[1][i] = int(vocab.index(clusters[0][randint(0,(len(clusters[0])-1))]))
		else:
		  if (new_c_num - c_num) !=1: 
			string = []
			cluster_prob = defaultdict(int)

			if len(clusters[c_num+1])==0:
				c_num+=1

			for cn in range(len(clusters[c_num+1])):
			  cluster_prob[clusters[c_num+1][cn]] = float(language_model[1,i,:][vocab.index(clusters[c_num+1][cn])])
			
			max_c_prob = max(cluster_prob.values())
			min_c_prob = min(cluster_prob.values())
			sum_c_prob = np.sum(cluster_prob.values())
			for word in cluster_prob.keys():
			  cluster_prob[word] = float(cluster_prob[word])/(float(sum_c_prob))
			next_verb = np.random.choice(len(cluster_prob.keys()),1, p=cluster_prob.values())
			outputs[1][i] = int(vocab.index(cluster_prob.keys()[next_verb[0]]))

	  #####################################

	  #Calculate reward
	  reward = 1
	  final_reward = 0
	  train_flag = True
	  #Check if any of the sampled tokens are in the wrong position. If so, set reward to 0
	  for i in range(FLAGS.batch_size):
		for p in range(0,5): ####### was 4
		  if outputs[p][i] != data_utils.START_ID and outputs[p][i] != data_utils.END_ID and encoder_inputs[3-p][i] != data_utils.START_ID and encoder_inputs[3-p][i] != data_utils.END_ID:
			if inv_vocab[outputs[p][i]] not in word_dict[p]:
			  #print("Invalid token postioning "+inv_vocab[outputs[p][i]]+" at line "+str(i)+", position "+str(p))
			  reward = 0
			  train_flag = False
			  for j in range(0,5): ####### was 4
				outputs[j][i] = decoder_inputs[j+1][i]
			  outputs[5][i] = data_utils.EOS_ID ####### was 4
			  break
			
		  else:
			print("Found start, shouldn't happen")
			reward = 0
			train_flag = False
			for j in range(0,5): ####### was 4
			  outputs[j][i] = decoder_inputs[j+1][i]
			outputs[5][i] = data_utils.EOS_ID ####### was 4
			break


		#Assign appropriate reward if clustering condition is satisfied. 
		if train_flag:
		  if inv_vocab[outputs[1][i]] == target_verb:
			old_c_num = cluster_number(inv_vocab[encoder_inputs[3][i]],clusters) ###### was 2
			if (old_c_num) == (len(clusters)-1):
			  reward = math.ceil(max_reward)
			else:
			  reward = 0
			  
		  else:
			old_c_num = cluster_number(inv_vocab[encoder_inputs[3][i]],clusters) ###### was 2
			new_c_num = cluster_number(inv_vocab[outputs[1][i]],clusters)
			if (new_c_num - old_c_num) == 1:
			  reward = word_dist[inv_vocab[outputs[1][i]]]
			else:
			  reward = 0
			  
		#penalize for having positions 2 & 3 the same
		if outputs[2][i]==outputs[3][i] and reward!=-1 and outputs[2][i]!=data_utils.START_ID and outputs[2][i]!=data_utils.END_ID:
		  reward = reward/100     
		
		final_reward += reward
	  final_reward /= FLAGS.batch_size
	  epoch_reward += final_reward
	  batch_count +=1
	  if final_reward == 0:
		final_reward = 0.00001

	  #####################################

	  #Get new training set and train
	  new_train_set = read_data.read_policy_data_set(encoder_inputs, outputs, _buckets, FLAGS.batch_size, vocab)
	  encoder_inputs, decoder_inputs, target_weights = model.get_batch(new_train_set, bucket_id, 0)
	  if train_flag:
		_, step_loss,selected_actions,_ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False, reward_flag = True, vocab=vocab, inv_vocab=inv_vocab, session_inner=sess_inner, reward = final_reward)
	  #loss += step_loss / FLAGS.batch_size
	  epoch_backward_loss += step_loss

	output = []
	