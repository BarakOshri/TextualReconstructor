import csv
import nltk # remember to install nltk and download punkt from nltk.download()
import numpy as np

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

GLOVE_MAT, GLOVE_VOCAB, _ = build('data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

input_sentences = ["I love this project.", "This is a sentence", "The weather is great today"]
wvi_sentences = [] # 2d array
for sentence in input_sentences:
    tokens = nltk.word_tokenize(sentence)
    wvi_sentence = []
    for token in tokens:
        token = token.lower() 
        wvi_sentence.append(GLOVE_VOCAB.index(token))
    wvi_sentences.append(wvi_sentence)

