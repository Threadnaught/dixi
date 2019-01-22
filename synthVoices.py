import dixi as dx
import os
import random
import sys

with open('synth/google-10000-english.txt', 'r') as word_file:
	words = [x for x in word_file.read().split('\n') if x]
word_index = {}
voices = dx.loadJson("synth/voices.json")
pitch_count = 4
#clear existing words:
os.system('rm synth/words/*.wav synth/index.json 2> /dev/null')
os.system('mkdir synth/words/ 2> /dev/null')

def dynamicPrint(str):
	a = ''
	for _ in range(50): a = a + ' '
	sys.stdout.write('\r' + a)
	sys.stdout.flush()
	sys.stdout.write('\r' + str)
	sys.stdout.flush()
#create words:
for i in range(int(dx.args['subset'])):
	w = words[i]
	dynamicPrint(str(i) + '/' + dx.args['subset'] + ': ' + w)
	command = ""
	for v in voices:
		for _ in range(pitch_count):
			pitch = random.randrange(10, 90)
			fname = 'synth/words/' + w + "-" + v + '-' + str(pitch) + '.wav'
			command = command + ('espeak -p ' + str(pitch) + ' -v ' + v + ' "' + w + '" --stdout | ffmpeg -i pipe:0 -ar 44100 -af silenceremove=0:0:0:1:5:-25dB  "' + fname + '" 2> /dev/null &')
			if w not in word_index:
				word_index[w] = [fname]
			else:
				word_index[w].append(fname)
	os.system(command[:-1])
dx.saveJson("synth/index.json", word_index)
