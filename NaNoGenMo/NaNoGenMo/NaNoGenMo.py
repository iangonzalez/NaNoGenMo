# NANOGENMO main code

import randsent
import re
import nltk.data
import random
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class LanguageGenerator:
    def __init__(self):
        self.corpus_file = None
        self.corpus_string = None
        self.gen_sentences = None
        self.markov_model = None

    def readCorpusFile(self, corpus_file):
        # read the corpus of text from a file
        with open(corpus_file) as corpus:
            corpus_strings = [line for line in corpus if re.search(r"\S", line)]
        self.corpus_string = "".join(corpus_strings)

    def sentenceTokenizeCorpus(self):
        # tokenize the corpus into sentences using the ntlk.punkt module
        if self.corpus_string is None:
            return None

        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return sent_detector.tokenize(re.sub(r"\n", " ", self.corpus_string))

    def crassifySocraticDialogue(self):
        sentences = self.sentenceTokenizeCorpus()
        if sentences is None:
            return None

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

        self.gen_sentences = sentences
        return sentences

    def writeOutput(self, outfile="outputs/output1.txt"):
        # output the sentences to output1.txt
        with open(outfile, "w") as outf:
            outf.write("\n".join(self.gen_sentences))


    def trainMarkovChain(self, n = 1):
      
        self.markov_model = defaultdict(lambda : defaultdict(int))
        sentences = self.sentenceTokenizeCorpus()
        if sentences is None:
            return None

        print("Training markov model on corpus.")

        word_tokenizer = RegexpTokenizer(r"\w+")

        for sentence in sentences:
            words = word_tokenizer.tokenize(sentence)
            last_word = "#"

            for word in words:
                if last_word is not None:
                    self.markov_model[last_word][word] += 1
                last_word = word

            self.markov_model[last_word]["#"] += 1

    def genMarkovSentences(self, n = 1):
        if self.markov_model is None:
            return None
           
        print("Generating markov random sentences.")

        self.gen_sentences = []
        
        for i in range(n):
            cur_token = "#"
            sentence = []
            while True:
                sum_freqs = sum([self.markov_model[cur_token][key] for key in self.markov_model[cur_token]])
                rand_choice = random.randint(0, sum_freqs)
                
                # make random weighted choice for next possible words.
                running_sum = 0
                for key in self.markov_model[cur_token]:
                    running_sum += self.markov_model[cur_token][key]
                    if running_sum >= rand_choice:
                        sentence.append(key)
                        break

                cur_token = key
                if key.startswith("#"):
                    break

            self.gen_sentences.append(" ".join(sentence))

        return self.gen_sentences


class GeneratorUnitTests:
    def crassifyTest(self):
        langGenerator = LanguageGenerator()
        corpus_file = "data/the_republic.txt"

        langGenerator.readCorpusFile(corpus_file)
        langGenerator.crassifySocraticDialogue()
        langGenerator.writeOutput("output/output1.txt")

    def markovLanguageTest(self):
        langGenerator = LanguageGenerator()
        corpus_file = "data/the_republic.txt"
        langGenerator.readCorpusFile(corpus_file)
        langGenerator.trainMarkovChain()
        print(langGenerator.genMarkovSentences(n=1))


if __name__ == '__main__':
    tester = GeneratorUnitTests()
    tester.crassifyTest()
    tester.markovLanguageTest()