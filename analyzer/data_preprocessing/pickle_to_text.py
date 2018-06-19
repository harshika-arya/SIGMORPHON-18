import pickle
import os
import sys

sentences = pickle.load(open('./pickle-dumps/sentences_intra_20', 'rb'))

with open('all_sentences.txt', 'w', encoding='utf-8') as f:
	for i in sentences:
		sentence = ' '.join(str(e) for e in i)
		f.write(sentence+'\n')
