import os
import re
import pickle 

path = "/media/harshika/Work/Projects/Hindi-morph-analyzer/dataset/HDTB_pre_release_version-0.05/IntraChunk/CoNLL/utf/news_articles_and_heritage/Development"

cnt = 0
sentences = []
rootwords = []
features = []
n_files = 0

sentences = pickle.load(open('./pickle-dumps/sentences_intra_20', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra_20', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra_20', 'rb'))

print(sentences[:10])
X1 = [item for sublist in sentences for item in sublist]
print(X1[:10])
"""
for filename in os.listdir(path)[:20]:
	n_files += 1
	with open(os.path.join(path, filename)) as fn:
		
		words = []
		roots = []
		tags = []
		for line in fn:
			line = line.rstrip()	
			# import pdb
			# pdb.set_trace()
			if(line): # keep adding words till blank line
				lis = re.split(r'\t+', line.rstrip('\t'))
			
				if cnt == 5:
					print(words)
				words.insert(len(words),lis[1])
				roots.insert(len(roots), lis[2])
				tags.insert(len(tags), lis[5])
				continue

			else: # encounter a blank line; add all previous words to form a sentence
				
				# clear() deletes the references to the lists
				# so make copy of lists 
				tempwords = []
				temproots = []
				temptags = []
				for i in range(len(words)):
					tempwords.append(words[i])
					temproots.append(roots[i])
					temptags.append(tags[i])

				sentences.append(tempwords)
				rootwords.append(temproots)
				features.append(temptags)
			
				cnt += 1
				words.clear()
				roots.clear()
				tags.clear()

print("total files: ", n_files)
print(cnt)
print("total sentences: ", len(sentences))
print(len(rootwords))
print(len(features))

pickle.dump(sentences, open('sentences_dev_20', 'wb'))
pickle.dump(rootwords, open('rootwords_dev_20', 'wb'))
pickle.dump(features, open('features_dev_20', 'wb'))

######## calculate stats #########

# mean sentence len
slen = 0
for s in sentences:
	#print(s)
	slen += len(s)
print("Mean len: ", slen/len(sentences))

# no of unique words
all_words = [item for sentence in sentences for item in sentence]
print("Total words: ", len(all_words))
words_set = set(all_words)
print("Unique words: ", len(words_set))
"""


