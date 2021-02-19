# Automated-confirmation-of-protein-function-annotation-using-NLP

The logistic regression, support vector machine using linear kernel and RCNN using LSTM unit models are in Models directory. 

RCNN code are modified from https://github.com/roomylee/rcnn-text-classification


Pretrained word embedding and sentence embedding models: https://github.com/ncbi-nlp/BioSentVec

Tensorflow:

pip install tensorflow==1.4

numpy version 1.15.2

# More info on environment setup:
```
git clone https://github.com/epfml/sent2vec.git 
pip install Cython

  *   download the pretrained model BioSentVec: BioSentVec_PubMed_MIMICIII-bigram_d700.bin
  *   installed fasttext as below (fasttext was installed successfully)
  
$ wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
$ unzip v0.9.2.zip
$ cd fastText-0.9.2
$ make

(pass_venv) jin@pass:/opt/pass/machine-learning-deploy$ ls
__pycache__      ensemble-eval.py  fastText-0.9.2  unit-classification.py
data_helpers.py  eval.py           sent2vec        v0.9.2.zip
(pass_venv) jin@pass:/opt/pass/machine-learning-deploy$ cd sent2vec/
(pass_venv) jin@pass:/opt/pass/machine-learning-deploy/sent2vec$ pip install .

```

# Model Training:
./cmd-lstm-ns

Alternatively, run the command below:
python train-ns-title.py --cell_type "lstm" --pos_dir "data/title.pos-ns" --neg_dir "data/title.neg-ns" --word2vec "/home/paperspace/Documents/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin" --word_embedding_dim 200 --context_embedding_dim 150 --hidden_size 150

LSTM means using LSTM unit; "ns" stands for non-stemmed words; "pos-ns" stands for positive non-stemmed data; "neg-ns" stands for negative non-stemmed data. The directory of "word2vec" should be modified based on the actual directory of the pretrained "BioWordVec_PubMed_MIMICIII_d200.vec.bin". 

# Model Evaluation: 
./runevalemsemble

Alternatively, run the command below:
python ensemble-eval.py --pos_dir "data/title.pos-ns" --neg_dir "data/title.neg-ns" --batch_size 32 --checkpoint_dir "/home/paperspace/Documents/RCNN-421-BioSentVec/runs-ns/1587761106/checkpoints" 

The directory to checkpoint in the command should be modified. Also, inside ensemble-eval.py, the directory of SVM and logistic regression models (which can be accessed in /Models) should be modified.

# Uni-tests:
run ./unit_test

Alternatively, run:
  
python unit-classification.py --checkpoint_dir "/home/paperspace/Documents/RCNN-421-BioSentVec/runs-ns/1587761106/checkpoints" --unknown_dir "data/P71009-pub"

The content of data/P71009-pub can be checked in the data directory. P71009 is the entry identifier of the protein from Swiss-Prot. Note: remember change the model directories accordingly inside unit-classification.py

Example output:
P71009-pub (3 publications) <br />
rcnn: <br />
[1 1 0] <br />
logistic regression: <br />
[1 1 1] <br />
svm: <br />
[0 1 0] <br />
voting: <br />
[2 3 1] <br />
final: <br />
[1 1 0] <br />
Two of publications predicted as positive.  <br />
