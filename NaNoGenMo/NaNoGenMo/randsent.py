# Random sentence generator using CFGs

import sys
import random

# function to read CFG from the given file. Will return a dict
# with LHS's as keys and lists of (odds, rule) pairs as values.
def read_grammar(grm_fname):
    grm = {}
    with open(grm_fname) as grmfile:
        grm_lines = grmfile.readlines()
        for line in grm_lines:
            # skip lines that start with #:
            if line[0] == "#":
                continue

            line = line.split()

            # skip blank lines:
            if len(line) <= 1:
                continue

            # read in the relative odds of each rule and store it:
            odds = int(line[0])
            if grm.has_key(line[1]):
                grm[line[1]].append((odds, line[2:]))
            else:
                grm[line[1]] = [(odds, line[2:])]
    return grm

# function to generate a random sentence starting from a symbol in the given
# grammar grm. accepts the treeprint option, which will add parentheses to
# clarify the sentence structure.
def randsentence(symbol, grm, treeprint = False):
    # nonterminal case (when symbol is a LHS in the grammar)
    if grm.has_key(symbol):
        # pick a random rule from the possible rules in the CFG:
        rules = grm[symbol]
        odds_total = sum([rule[0] for rule in rules])

        rand_pick = random.randint(1, odds_total)
        running_tot = 0
        picked_rule = []

        # this for loop picks one of the rules based on a rand int
        for rule in rules:
            running_tot += rule[0]
            if rand_pick <= running_tot:
                picked_rule = rule[1]
                break

        # return the result of recursing onto each symbol in the rule:
        if treeprint:
            # with the treeprint option, add parens and the parent symbol to
            # elucidate sentence structure
            sentence = ""
            for sym in picked_rule:
                sentence += " " + randsentence(sym, grm, True) + " "
            sentence = "(" + symbol + " " + sentence + ")"
            return sentence
        else:
            # without the tree print option, simply seprate the results by
            # spaces:
            sentence = []
            for sym in picked_rule:
                sentence.append(randsentence(sym, grm))
            return " ".join(sentence)

    # terminal case: do nothing
    else:
        return symbol

# Main driver code:
if __name__ == '__main__':
    # with no -t option, read in the grammar and print n sentences without printree flag:
    if len(sys.argv) == 3:
        grm_fname = sys.argv[1]
        niters = int(sys.argv[2])

        grammar = read_grammar(grm_fname)
        for i in range(0, niters):
            print randsentence("ROOT", grammar)

    # with the -t option, read grammar and print n sentences with printree flag:
    elif len(sys.argv) == 4:
        grm_fname = sys.argv[2]
        niters = int(sys.argv[3])
        grammar = read_grammar(grm_fname)

        if sys.argv[1] == "-t":
            for i in range(0, niters):
                print randsentence("ROOT", grammar, True)
        else:
            print "error: option must be [-t]"

    # anything else is an error
    else:
        print "error: incorrect number of arguments"