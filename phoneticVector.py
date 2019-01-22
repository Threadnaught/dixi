import dixi as dx
import tensorflow as tf
import numpy as np
import datetime
import os
#model vars:
batch_size = 50
epoch_count = 2000
nce_sample_count = 8
#load synth and start session:
dx.loadWords("synth")
sess = tf.InteractiveSession()
#audio input and definition:
audio_in = tf.placeholder(tf.float32, [batch_size, 60000])
audio_lengths_in = tf.placeholder(tf.int32, [batch_size])
#which word does the clip correspond do:
index_in = tf.placeholder(tf.int32, [batch_size])
index_reshape = tf.reshape(index_in, [batch_size, 1])
#add dixi frontend, which ffts, then stretches each clip to 800 width
audio_stretched = dx.addFrontend(audio_in, audio_lengths_in, addNoise=False)
#pass through conv and max pool and flatten:
audio_conv, _ = dx.addConvPoolLayers(audio_stretched)
conv_flat = tf.reshape(audio_conv, [batch_size, -1])
#pass through dense:
dense, _ = dx.addDense(conv_flat, [int(conv_flat.shape[1]), 100, dx.phonetic_vector_size])
#nce vars:
nce_weights = tf.Variable(tf.random_uniform([len(dx.train_words), int(dense.shape[1])]))
nce_biases = tf.Variable(tf.zeros([len(dx.train_words)]))
#apply noise contrastive estimation:
nce_loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, index_reshape, dense, nce_sample_count, len(dx.train_words)))#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(index_in, len(dx.train_words)), logits=classified))
train_step = tf.train.AdamOptimizer().minimize(nce_loss)
#calculate which word is guessed for accuracy:
guess = tf.argmax(tf.matmul(dense, tf.transpose(nce_weights)) + nce_biases, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(guess, tf.int32), index_in), tf.float32))
#init all the vars:
tf.global_variables_initializer().run()
#for eta estimation:
train_start_time = datetime.datetime.now()
#train:
for i in range(epoch_count):
	for _ in range(100):
		batch_indicies = dx.getBatch(batch_size)
		batch_length, batch_audio = dx.getAudio(batch_indicies)
		train_step.run(feed_dict={audio_in:batch_audio, audio_lengths_in:batch_length, index_in:batch_indicies})
	eval_indicies = dx.getBatch(batch_size, False)
	eval_length, eval_audio = dx.getAudio(eval_indicies, False)
	delta_per_epoch = (datetime.datetime.now() - train_start_time) / (i+1)
	time_remaining = delta_per_epoch * (epoch_count - i)
	print "\t" + str(i+1) + "/" + str(epoch_count) + " accuracy: " + str(accuracy.eval(feed_dict={audio_in:eval_audio, audio_lengths_in:eval_length, index_in:eval_indicies})) + " loss: " + str(nce_loss.eval(feed_dict={audio_in:eval_audio, audio_lengths_in:eval_length, index_in:eval_indicies})) + " ETA: " + str(time_remaining)
#save phonetic vectors so tensorboard can see them:
all_words = []
word_file_counts = {}
#concat the training words:
for w in dx.train_words:
	this_words = dx.train_words[w]
	if w in dx.eval_words:
		this_words = this_words + dx.eval_words[w]
	for f in this_words:
		all_words.append((w,f))
	word_file_counts[w] = len(this_words)
#iterate through the batches, and mean all of the files for each word into one vector:
word_vectors = {}
for i in range(0,len(all_words),50):
	print str(i) + "/" + str(len(all_words))
	batch = all_words[i:i+batch_size]
	batch_audio = []
	batch_lens = []
	for j in range(batch_size):
		l,a = dx.getFile(batch[j][1], 60000)
		batch_audio.append(a)
		batch_lens.append(l)
		if batch[j][0] not in word_vectors:
			word_vectors[batch[j][0]] = np.zeros([dx.phonetic_vector_size])
	batch_dense = dense.eval(feed_dict={audio_in:batch_audio, audio_lengths_in:batch_lens})
	for j in range(batch_size):
		word_vectors[batch[j][0]] = word_vectors[batch[j][0]] + (batch_dense[j]/word_file_counts[batch[j][0]])
#formatting so tensorflow saver can understand:
word_vector_list = []
words = []
for w in word_vectors:
	word_vector_list.append(word_vectors[w])
	words.append(w)
#push calculated phonetic vectors into a new variable and init:
word_vecs = tf.Variable(np.asarray(word_vector_list), name="vecs")
tf.global_variables_initializer().run()
#make folder
os.system('rm -rf logs/ 2> /dev/null')
os.system('mkdir logs')
#write the words to the file:
with open('logs/metadata.tsv', 'w') as metadata_file:
	for x in words:
		metadata_file.write(x + "\n")
#save the vectors:
saver = tf.train.Saver([word_vecs])
sess.run(word_vecs.initializer)
saver.save(sess, os.path.join('logs', 'words.ckpt'))
#tensorboard config:
config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = word_vecs.name
embedding.metadata_path = 'metadata.tsv'
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(tf.summary.FileWriter('logs'), config)