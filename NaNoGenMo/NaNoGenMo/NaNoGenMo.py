# NANOGENMO main code

import randsent
import re
import nltk.data
import random



corpus_file = "data/the_republic.txt"

print("reading The Republic corpus.")
corpus_string = ""
with open(corpus_file) as corpus:
    for line in corpus:
        if re.search(r"\S", line):
            corpus_string += line

# use nlkt to tokenize the corpus into sentences
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sent_detector.tokenize(re.sub(r"\n", " ", corpus_string))


print("adding fun edits to corpus.")

# read in the CFG for generating the confrontational replies to socrates.
bs_grammar = randsent.read_grammar("republic_rebuffs_grammar.txt")

for i,sent in enumerate(sentences):
    #replace fawning replies with profane rebuffs.
    if len(sent.split(" ")) < 4 and re.search(r"yes|certain|true|just|so|obvious|socrates|sure|exact|clear", sent.lower()):
        sentences[i] = (randsent.randsentence("ROOT", bs_grammar))

    #intersperse likes into questions
    if sent[-1] == "?" and len(sent.split(" ")) > 3: # sentence is a question
        sent = sent.split(" ")
        sent[random.randint(0, len(sent)-2)] +=  ", like,"
        sentences[i] = " ".join(sent)
        
# output the sentences to output1.txt
with open("outputs/output1.txt", "w") as outf:
    outf.write("\n".join(sentences))