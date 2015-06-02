import os
import re
import csv
import nltk # remember to install nltk and download punkt from nltk.download()
import numpy as np
import nltk.data
import sys

def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):    
    # Thanks to Prof. Chris Potts for this function
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []    
    rownames = []
    for line in reader:        
        rownames.append(line[0])            
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

def get_data(return_type="wv"):
	GLOVE_MAT, GLOVE_VOCAB, _ = build('data_processing/data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

	path = "/Users/nishithkhandwala/Desktop/TextualReconstructor/data_processing/wiki/"
	all_wiki_files = os.listdir(path)
	all_wiki_files = all_wiki_files[1:] # to remove .DS_Store

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	all_wiki_lines = []
	for wiki_file in all_wiki_files:
		all_lines = open(path + wiki_file, 'r').readlines()
		for line in all_lines:
			if line[0] != "<":
				line = line.replace('\n', '')
				if line != "":
					all_wiki_lines.append(line)

	all_sentences = []
	for line in all_wiki_lines:
		try:
			sentences = tokenizer.tokenize(line) 
		except:
			pass
		for sentence in sentences:
			all_sentences.append(sentence)

	if return_type == "char":
		return all_sentences

	all_wvi = []
	num_sentences = len(all_sentences)
	sentences_missed = 0
	num_seen = 0
	for sentence in all_sentences:
		try:
			wvi_sentence = []
			tokens = nltk.word_tokenize(sentence)
			for token in tokens:
				token = token.lower()
				wvi_sentence.append(GLOVE_VOCAB.index(token))
			all_wvi.append(wvi_sentence)
		except:
			sentences_missed += 1
			pass
		num_seen += 1
		if num_seen % 10000 == 0:
			print "Seen: %d" % num_seen

	print "Number of sentences obtained: %d" % (num_sentences - sentences_missed)
	print "Total number of sentences: %d" % num_sentences
	return all_wvi


