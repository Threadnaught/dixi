import json
import bson
import argparse
import sys
#JSON/ARGPARSE:
def saveJson(name, vars):
	with open(name, 'w') as outfile:
		json.dump(vars, outfile)
def loadJson(name):
	with open(name, 'r') as infile:
		return json.load(infile)
def saveBson(name, vars):
	with open(name, 'w') as outfile:
		outfile.write(bson.dumps(vars))
def loadBson(name):
	with open(name, 'r') as infile:
		return bson.loads(infile.read())
argcfg = loadJson('args.json')
args = []
if sys.argv[0] in argcfg:
	parser = argparse.ArgumentParser(description=argcfg[sys.argv[0]]['description'])
	for a in argcfg[sys.argv[0]]['args']:
		parser.add_argument(a['name'], nargs=a['nargs'], default=a['default'])
	args = vars(parser.parse_args())
#EVERYTHING ELSE: (import tensorflow after above to avoid annoying output when using --help)
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
#misc:
def pinRandom():
	random.seed(0)
#TENSORFLOW FUNCS:
phonetic_vector_size=8
def addFFT(input):
	stfts = tf.contrib.signal.stft(input, frame_length=256, frame_step=64, fft_length=256)
	return ((tf.log(tf.abs(stfts) + 1e-6) + 15) / 30), tf.imag(stfts)
def addStretch(input, length=800):
	return tf.image.resize_images(input, [length, input.shape[-2]])
def addFrontend(audio_in, audio_lengths=None, stretch=True, addNoise=True, noiseIntensity = 0.3):
	batch_size = audio_in.shape[0]
	if addNoise:
		audio_in = audio_in + tf.random_uniform(np.shape(audio_in), -0.5 * noiseIntensity, 0.5 * noiseIntensity)
	audio_fft, _ = addFFT(audio_in)
	audio_reshape = tf.reshape(audio_fft, [audio_fft.shape[0], -1, audio_fft.shape[2], 1])
	if not stretch:
		return audio_reshape
	window_lengths = tf.cast(tf.ceil(tf.cast(audio_lengths, tf.float32) / 64.0), tf.int32)
	audio_split = tf.split(audio_reshape, batch_size)
	audio_stretched = []
	for i in range(len(audio_split)):
		sliced = tf.slice(audio_split[i][0], [0, 0, 0], [window_lengths[i], int(audio_fft.shape[2]), 1])
		audio_stretched.append([addStretch(sliced)])
	return tf.concat(audio_stretched, 0)
def addConvPoolLayers(audio_fft, arrays=None, stride=2):
	#load arrays:
	conv_1_filter_arr = tf.random_uniform([stride,stride,1,2]) if arrays == None else arrays[0]
	conv_1_bias_arr = tf.random_uniform([2]) if arrays == None else arrays[1]
	conv_2_filter_arr = tf.random_uniform([stride,stride,2,4]) if arrays == None else arrays[2]
	conv_2_bias_arr = tf.random_uniform([4]) if arrays == None else arrays[3]

	conv_1_filter = tf.Variable(conv_1_filter_arr)
	conv_1_bias = tf.Variable(conv_1_bias_arr)
	conv_1 = tf.nn.leaky_relu(tf.nn.conv2d(input=audio_fft, filter=conv_1_filter, strides=[1,stride,stride,1],padding="SAME") + conv_1_bias)

	conv_1_pool = tf.nn.max_pool(conv_1, [1,stride,stride,1], [1,stride,stride,1], "SAME")

	conv_2_filter = tf.Variable(conv_2_filter_arr)
	conv_2_bias = tf.Variable(conv_2_bias_arr)
	conv_2 = tf.nn.leaky_relu(tf.nn.conv2d(input=conv_1_pool, filter=conv_2_filter, strides=[1,stride,stride,1],padding="SAME") + conv_2_bias)

	conv_2_pool = tf.nn.max_pool(conv_2, [1,stride,stride,1], [1,stride,stride,1], "SAME")

	return conv_2_pool, (conv_1_filter, conv_1_bias, conv_2_filter, conv_2_bias)
def addDense(input, widths=None, arrs=None):
	weights = []
	biases = []
	if arrs != None:
		for i in range(0, len(arrs), 2):
			weights.append(arrs[i])
			biases.append(arrs[i+1])
	elif widths != None:
		for i in range(0, len(widths)-1):
			weights.append(tf.random_uniform([widths[i], widths[i+1]]))
			biases.append(tf.zeros([widths[i+1]]))
	else:
		raise Exception('No widths or arrays, undefined shape')
	cur = input
	ret = []
	for i in range(len(weights)):
		this_weights = tf.Variable(weights[i])
		this_biases = tf.Variable(biases[i])
		cur = tf.nn.leaky_relu(tf.matmul(cur, this_weights) + this_biases)
		ret = ret + [this_weights, this_biases]
	return (cur, ret)
def addClassifier(input, arrays=None):
	classifier_weights_arr = tf.random_uniform([int(input.shape[1]), len(train_words)]) if arrays == None else arrays[0]
	classifier_biases_arr = tf.random_uniform([len(train_words)]) if arrays == None else arrays[1]
	classifier_weights = tf.Variable(classifier_weights_arr)
	classifier_biases = tf.Variable(classifier_biases_arr)

	classified = addClassifierWithVars(input, classifier_weights, classifier_biases)

	return classified, (classifier_weights, classifier_biases)
def addClassifierWithVars(input, weights, biases):
	return tf.matmul(input, weights) + biases
def extractArrays(vars):
	return [x.eval().tolist() for x in vars]
#AUDIO FILE MANIPULATION:
global_sample_rate = 44100
words = None
train_words = None
eval_words = None
def getFile(word, sample_count):
	rate, d = wavfile.read(word)
	if rate != global_sample_rate:
		raise "bad sample rate"
	return len(d), np.concatenate([d, np.zeros([sample_count - len(d)])])
def callThruGetFile(p):
	return getFile(p[0], p[1])
def loadWords(source="synth", eval_ratio = 20):
	global train_words
	global eval_words
	global words
	train_words = loadJson(source + "/index.json")
	sampled = random.sample(train_words, len(train_words) / eval_ratio)
	eval_words = {}
	for sample in sampled:
		i = random.randrange(0, len(train_words[sample]))
		eval_words[sample] = [train_words[sample].pop(i)]
	words = train_words.keys()
def getBatch(batch_size, train=True):
	ret = []
	while len(ret) < batch_size:
		c = random.randrange(len(words))
		#print c
		if (not train) and (words[c] not in eval_words):
			continue
		#if c not in ret:
		ret.append(c)
	return ret
def getAudio(batch_indicies, train=True, sample_count=60000):
	source = train_words if train else eval_words
	params = [(random.choice(source[words[word]]), sample_count) for word in batch_indicies]
	received = [callThruGetFile(x) for x in params]
	lens = []
	ret = []
	for item in received:
		lens.append(item[0])
		ret.append(item[1])
		if item[0] == 0:
			print "0 length file"
	return lens, ret
#PLOTTING:
def plot(data, filename, set_range=True):
	tgt = None
	if set_range:
		tgt = plt.contourf(np.transpose(data), [0.1 * float(x) for x in range(10)])
	else:
		tgt = plt.contourf(np.transpose(data))
	cbar = plt.colorbar(tgt)
	plt.savefig(filename)
	plt.clf()
