#NaNoGenMo Submission, 2015

This repository contains my code and my results for NaNoGenMo 2015.

##Idea

Create a new text in the style of both Plato's Republic and The Adventures of Sherlock Holmes. For some reason, the idea of these two together delights me.
Ideally, the socratic dialogue will become combined with the Holmes-Watson dialogue in entertaining ways.
A current draft of the text can be found in `outputs/holmesXSocrates_draft1.txt`. The dialogue nicely combines the styles
from both of the dialogues, and has the wacky sound I was looking for.

##Implementation

Methods being used currently:
- Markov modeling (bigram, trigram)
- Snipping + pasting sentences together.
- Identifying Socratic dialogue, adding "basic" (slang sense) language


##Code
`randsent.py` generates random sentences given a CFG formatted like `sample_grammar.txt`.
`NaNoGenMo.py` contains the main methods in the LanguageGenerator class.

##Data

I'm using the full text of The Republic and The Adventures of Sherlock Holmes.
The plaintexts were obtained with no licensing restrictions from project Gutenberg.