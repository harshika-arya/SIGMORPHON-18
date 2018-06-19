import os
import re
import pickle 

path = "/media/harshika/Work/Projects/SIGMORPHON/baseline/datasets"

cnt = 0
sentences = []
rootwords = []
features = []


sentences = pickle.load(open('sentences_train_low', 'rb'))
rootwords = pickle.load(open('rootwords_train_low', 'rb'))
features = pickle.load(open('features_train_low', 'rb'))

print(sentences[:10])
X1 = [item for sublist in sentences for item in sublist]
print(X1[:10])
"""
with open("english-dev") as fn:
        
    words = []
    roots = []
    tags = []
    for line in fn:
        line = line.strip('\n')
        if(line): # keep adding words till blank line
            wf, lemma, msd = line.split('\t')
            words.insert(len(words), wf)
            roots.insert(len(roots), lemma)
            tags.insert(len(tags), msd)
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

print(cnt)
print("total sentences: ", len(sentences))
print(len(rootwords))
print(len(features))

pickle.dump(sentences, open('sentences_dev', 'wb'))
pickle.dump(rootwords, open('rootwords_dev', 'wb'))
pickle.dump(features, open('features_dev', 'wb'))

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


