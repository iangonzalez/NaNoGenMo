# NANOGENMO main code

import randsent
import re
import sys
import math
import nltk.data
import random
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class LanguageGenerator:
    def __init__(self, file_to_read=None):
        self.corpus_file = None
        self.corpus_string = None
        self.gen_sentences = None
        self.markov_model = None
        self.ngram_degree = None
        self.corpus_sentences = None
        self.dialogue_tuples = None

        if file_to_read:
            self.readCorpusFile(file_to_read)



    def readCorpusFile(self, corpus_file):
        # read the corpus of text from a file
        with open(corpus_file) as corpus:
            corpus_strings = [line for line in corpus if re.search(r"\S", line)]
        self.corpus_string = "".join(corpus_strings).decode("utf-8")

    def sentenceTokenizeCorpus(self):
        # tokenize the corpus into sentences using the ntlk.punkt module
        if self.corpus_string is None:
            return None

        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.corpus_sentences = sent_detector.tokenize(re.sub(r"\n", " ", self.corpus_string))
        return self.corpus_sentences

    def filterSentencesForDialogue(self):
        # try to find only question+answer dialogue in the corpus
        if self.corpus_sentences is None:
            return None

        self.dialogue_tuples = []
        for i, sent in enumerate(self.corpus_sentences):
            if "?" in sent:
                if (not self.dialogue_tuples or sent != self.dialogue_tuples[-1][0]):
                    self.dialogue_tuples.append((sent, self.corpus_sentences[i+1]))

        return self.dialogue_tuples

    def crassifySocraticDialogue(self):
        # needs to generate a copy of sentence list (so as not to modify it unnecessarily)
        sentences = self.corpus_sentences
        if sentences is None:
            sentences = sentenceTokenizeCorpus()
        sentences = sentences[:]

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
        with open(outfile, "w+") as outf:
            outf.write("\n".join(self.gen_sentences).encode("utf-8"))


    def trainMarkovChain(self, n = 1):

        self.ngram_degree = n
      
        self.markov_model = defaultdict(lambda : defaultdict(int))

        sentences = self.corpus_sentences
        if sentences is None:
            sentences = self.sentenceTokenizeCorpus()

        print("Training markov model on corpus.")

        word_tokenizer = RegexpTokenizer(r"\w+")

        for sentence in sentences:
            words = word_tokenizer.tokenize(sentence)
            last_word_list = ["#"] * n

            for word in words:
                last_token = " ".join(last_word_list)
                
                self.markov_model[last_token][word] += 1
                
                last_word_list.append(word)
                last_word_list = last_word_list[1:]

            last_token = " ".join(last_word_list)
            self.markov_model[last_token]["#"] += 1

    def genMarkovSentences(self, num_sentences = 1):
        if self.markov_model is None:
            return None
           
        print("Generating markov random sentences.")

        self.gen_sentences = []
        
        for i in range(num_sentences):
            cur_token_list = ["#"] * self.ngram_degree
            sentence = []
            while True:
                #print(sentence, len(sentence))
                cur_token = " ".join(cur_token_list)
                sum_freqs = sum([self.markov_model[cur_token][key] for key in self.markov_model[cur_token]])
                rand_choice = random.randint(0, sum_freqs)
                
                # make random weighted choice for next possible words.
                running_sum = 0
                for key in self.markov_model[cur_token]:
                    running_sum += self.markov_model[cur_token][key]
                    if running_sum >= rand_choice:
                        sentence.append(key)
                        break

                cur_token_list.append(key)
                cur_token_list = cur_token_list[1:]

                if key.startswith("#"):
                    self.gen_sentences.append(" ".join(sentence))
                    break
                elif len(sentence) > 50:
                    break

        return self.gen_sentences

    # cuts a random sentence, then finds a matching sentence to use as an ending to that sentence
    # can be passed an optional second set of sentences from a different corpus to use as endings
    def sentenceSnipNShuffle(self, num_sentences = 1, sentence_set_2 = None, preserve_punct = False):
        sentences = self.corpus_sentences
        if sentences is None:
            sentences = self.sentenceTokenizeCorpus()
        sentences = sentences[:]

        word_tokenizer = RegexpTokenizer(r"\w+")

        self.gen_sentences = []

        for i in range(num_sentences):
            start_sent_str = random.choice(sentences)
            start_sent = word_tokenizer.tokenize(start_sent_str)

            # choose place to cut initial sentence based on normal distribution centered on the middle of sentence.
            mu = float(len(start_sent))/2.0                 # center = average
            sigma = math.sqrt(float(len(start_sent))/8.0)   # stdev = 1/6 sentence length 
            cut_loc = random.normalvariate(mu, sigma)
            while cut_loc < 0 or cut_loc >= len(start_sent):
                cut_loc = random.normalvariate(mu, sigma)

            first_half = start_sent[:int(math.floor(cut_loc))]
            if not first_half:
                continue  # nothing to do if first half is empty

            joining_word = first_half[-1]

            # find all candidate sentences that have the joining word (last word of first half)
            if sentence_set_2 is None:
                candidate_sents = [sent for sent in sentences if (sent.find(" " + joining_word + " ") != -1 and sent != start_sent_str)]
            else:
                # use the passed set of sentences as endings if it exists:
                candidate_sents = [sent for sent in sentence_set_2 if sent.find(" " + joining_word + " ") != -1]
                
            if not candidate_sents:
                continue
            
            # matches the lowest index matching word in the chosen candidate sentence
            second_sent_str = random.choice(candidate_sents)
            second_sent = word_tokenizer.tokenize(second_sent_str)
            try:
                second_half = second_sent[second_sent.index(joining_word)+1:]
            except:
                continue

            if not preserve_punct:
                self.gen_sentences.append(" ".join(first_half + second_half))
            else:
                idx1 = start_sent_str.find(joining_word) + len(joining_word)
                while idx1 < len(start_sent_str) and start_sent_str[idx1] != " ":
                    idx1 += 1
                first_half = start_sent_str[:idx1]
                
                idx2 = second_sent_str.find(joining_word) + len(joining_word)
                while idx2 < len(second_sent_str) and second_sent_str[idx2] != " ":
                    idx2 += 1

                second_half = second_sent_str[idx2:]
                self.gen_sentences.append(first_half + second_half)

        return self.gen_sentences
            

# Unit tests for the language generator class
class GeneratorUnitTests:
    def crassifyTest(self):
        langGenerator = LanguageGenerator()
        corpus_file = "data/the_republic.txt"

        langGenerator.readCorpusFile(corpus_file)
        langGenerator.crassifySocraticDialogue()
        langGenerator.writeOutput("output/output1.txt")

    def markovLanguageTest(self):
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/the_republic.txt")
        langGenerator.trainMarkovChain(n=2)
        langGenerator.genMarkovSentences(num_sentences = 100)
        langGenerator.gen_sentences = [sent.split(" #")[0] + "." for sent in langGenerator.gen_sentences]
        langGenerator.writeOutput("outputs/republic_markov_output.txt")

    def snipSentenceTest(self):
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/the_republic.txt")
        print(langGenerator.sentenceSnipNShuffle())

    def snipSentenceTest2sources(self):
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/the_republic.txt")

        langGenerator2= LanguageGenerator()
        langGenerator2.readCorpusFile("data/sherlock_holmes.txt")
        langGenerator2.sentenceTokenizeCorpus()

        langGenerator.sentenceSnipNShuffle(num_sentences = 1, sentence_set_2 = langGenerator2.corpus_sentences)
        print(langGenerator.gen_sentences)

    def dialogueFilterTest(self):
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/the_republic.txt")
        langGenerator.sentenceTokenizeCorpus()

        langGenerator2= LanguageGenerator()
        langGenerator2.readCorpusFile("data/sherlock_holmes.txt")
        langGenerator2.sentenceTokenizeCorpus()

        langGenerator.filterSentencesForDialogue()
        langGenerator2.filterSentencesForDialogue()

        with open("data/republic_questions.txt", "w") as outf:
            questions = map(lambda tup : tup[0], langGenerator.dialogue_tuples)
            outf.write("\n".join(questions).encode("utf-8"))

        with open("data/holmes_questions.txt", "w") as outf:
            questions = map(lambda tup : tup[0], langGenerator2.dialogue_tuples)
            outf.write("\n".join(questions).encode("utf-8"))

        with open("data/republic_answers.txt", "w") as outf:
            answers = map(lambda tup : tup[1], langGenerator.dialogue_tuples)
            outf.write("\n".join(answers).encode("utf-8"))

        with open("data/holmes_answers.txt", "w") as outf:
            answers = map(lambda tup : tup[1], langGenerator2.dialogue_tuples)
            outf.write("\n".join(answers).encode("utf-8"))

    def dialogueMasherTest(self):
        # generate mashed questions
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/republic_questions.txt")
        langGenerator.sentenceTokenizeCorpus()

        langGenerator2= LanguageGenerator()
        langGenerator2.readCorpusFile("data/holmes_questions.txt")
        langGenerator2.sentenceTokenizeCorpus()

        langGenerator.sentenceSnipNShuffle(num_sentences = 1000, sentence_set_2 = langGenerator2.corpus_sentences, preserve_punct = True)
        langGenerator2.sentenceSnipNShuffle(num_sentences = 1000, sentence_set_2 = langGenerator.corpus_sentences, preserve_punct = True)

        langGenerator.writeOutput("outputs/mashed_questions.txt")
        langGenerator2.writeOutput("outputs/mashed_questions.txt")

        # generate mashed answers
        langGenerator = LanguageGenerator()
        langGenerator.readCorpusFile("data/republic_answers.txt")
        langGenerator.sentenceTokenizeCorpus()

        langGenerator2 = LanguageGenerator()
        langGenerator2.readCorpusFile("data/holmes_answers.txt")
        langGenerator2.sentenceTokenizeCorpus()

        langGenerator.sentenceSnipNShuffle(num_sentences = 1000, sentence_set_2 = langGenerator2.corpus_sentences, preserve_punct = True)
        langGenerator2.sentenceSnipNShuffle(num_sentences = 1000, sentence_set_2 = langGenerator.corpus_sentences, preserve_punct = True)

        langGenerator.writeOutput("outputs/mashed_answers.txt")
        langGenerator2.writeOutput("outputs/mashed_answers.txt")


# weighted random choice from a list of corpuses (lists of sentences or other tokens).
def randomPickFromCorpuses(corpuses, weights):
    assert len(corpuses) == len(weights)
    weightsum = float(sum(weights))
    weights = [float(wt)/weightsum for wt in weights]
    weights.sort()
    #print(weights)
    choice = random.uniform(0.0, 1.0)
    runningsum = 0.0
    for i in range(len(weights)):
        runningsum += weights[i]
        if runningsum <= choice:
            break
    corpus = corpuses[i]

    return random.choice(corpus)

    

if __name__ == '__main__':
    tester = GeneratorUnitTests()
    if "-test" in sys.argv:
        tester.crassifyTest()
        tester.markovLanguageTest()
        tester.snipSentenceTest2sources()
        tester.dialogueFilterTest()
        tester.dialogueMasherTest()
    else:
        # create the final combined dialogue.
        republic_questions_corp = LanguageGenerator("data/republic_questions.txt").sentenceTokenizeCorpus()
        holmes_questions_corp = LanguageGenerator("data/holmes_questions.txt").sentenceTokenizeCorpus()

        republic_answers_corp = LanguageGenerator("data/republic_answers.txt").sentenceTokenizeCorpus()
        holmes_answers_corp = LanguageGenerator("data/holmes_answers.txt").sentenceTokenizeCorpus()

        mashed_questions_corp = LanguageGenerator("outputs/mashed_questions.txt").sentenceTokenizeCorpus()
        mashed_answers_corp = LanguageGenerator("outputs/mashed_answers.txt").sentenceTokenizeCorpus()

        question_corps = [republic_questions_corp, holmes_questions_corp, mashed_questions_corp]
        answer_corps = [republic_answers_corp, holmes_answers_corp, mashed_answers_corp]

        final = []
        for i in range(2000):
            final.append(randomPickFromCorpuses(question_corps, [1, 1, 3]))
            final.append(randomPickFromCorpuses(answer_corps, [1, 1, 3]))

        with open("outputs/holmesXsocrates_draft1.txt", "w") as outf:
            outf.write("\n".join(final).encode("utf-8"))
