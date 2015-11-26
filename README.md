#NaNoGenMo Submission, 2015

This repository contains my code and my results for NaNoGenMo 2015.

##Idea

Create a new text in the style of both Plato's Republic and The Adventures of Sherlock Holmes. For some reason, the idea of these two together delights me.
Ideally, the socratic dialogue will become combined with the Holmes-Watson dialogue in entertaining ways.
I'm not sure yet, but I think I want the final product to focus on these dialogues -- I might also modify them to sound more "modern" (for maximum discord).

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