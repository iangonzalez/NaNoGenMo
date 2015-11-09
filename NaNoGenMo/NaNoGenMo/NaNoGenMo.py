# NANOGENMO main code

import randsent
import re
import nltk.data
import random

bs_grammar = randsent.read_grammar("republic_rebuffs_grammar.txt")

corpus_file = "data/the_republic.txt"

print("reading corpus.")
corpus_string = ""
with open(corpus_file) as corpus:
    for line in corpus:
        if re.search(r"\S", line):
            corpus_string += line

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sent_detector.tokenize(re.sub(r"\n", " ", corpus_string))

for i,sent in enumerate(sentences):
    #replace fawning replies
    if len(sent.split(" ")) < 4 and re.search(r"yes|certain|true|just|so|obvious|socrates|sure|exact|clear", sent.lower()):
        sentences[i] = (randsent.randsentence("ROOT", bs_grammar))

    #intersperse likes into questions
    if sent[-1] == "?" and len(sent.split(" ")) > 3: # sentence is a question
        sent = sent.split(" ")
        sent[random.randint(0, len(sent)-2)] +=  ", like,"
        sentences[i] = " ".join(sent)
        

with open("outputs/output1.txt", "w") as outf:
    outf.write("\n".join(sentences))