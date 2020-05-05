# Automated-confirmation-of-protein-function-annotation-using-NLP

The logistic regression, support vector machine using linear kernel and RCNN using LSTM unit models are in Models directory. 

RCNN code are modified from https://github.com/roomylee/rcnn-text-classification


Pretrained word embedding and sentence embedding models: https://github.com/ncbi-nlp/BioSentVec

Tensorflow:

pip install tensorflow==1.4

numpy version 1.15.2

# uni-tests:
https://www.uniprot.org/uniprot/F8SJR0
publication titles:
pseudomonas aeruginosa generalized transducing phage phipa is new member of phikzlike group of jumbo phages and infects model laboratory strains and clinical isolates from cystic fibrosis patients

phage nucleus and tubulin spindle are conserved among large pseudomonas phages

viral capsid trafficking along treadmilling tubulin filaments in bacteria

run ./unit_test

output:

negative

rcnn [0 0 0]

lr [0 0 1]

svm [0 0 0]
