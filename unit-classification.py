import tensorflow as tf
import numpy as np
import os
import data_helpers

# use BioSent2Vec and traditional ml on titles
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score
import glob
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import sent2vec
import sys

neg_path = '/home/paperspace/Documents/RCNN-421-BioSentVec/data/title.neg-ns'
pos_path = '/home/paperspace/Documents/RCNN-421-BioSentVec/data/title.pos-ns'
model_path = '/home/paperspace/Documents/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
model = sent2vec.Sent2vecModel()

try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')

# load classic ml models
abs_dir = "/home/paperspace/Documents/BioSentVec/"
lr_dir = abs_dir + "lg_models/"
rf_dir = abs_dir + "rf_models/"
svm_dir = abs_dir + "svm_models_linear/"# svm_models_linear/c_0.1_svm.sav

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("unknown_dir", "data/unknown", "Path of unknown data")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (Default: 1)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

def eval():
    """
    with tf.device('/cpu:0'):
        x_text, y= data_helpers.load_data_and_labels(FLAGS.unknown_dir, FLAGS.unknown_dir)
    
    print("x_text = " + str(x_text))
    print("y=" + str(y))
    sys.exit(0)
    """
    # x_text = ["identification required acyltransferase step biosynthesis phosphatidylinositol mannosides mycobacterium species", "identification isolation cloning bacillus thuringiensis cryiac toxinbinding protein midgut lepidopteran insect heliothis virescens"]
    with open(FLAGS.unknown_dir) as unknown_f:
       x_text = [line.rstrip() for line in unknown_f]
    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    print(x_eval.shape)
    x = x_eval
    print(list(x_eval))

    # prepare the data from one single file
    unknown_arrays_700 = np.zeros((len(x_text), 700)) # the sum of all the training examples (neg+pos)
    
    with open(FLAGS.unknown_dir) as unknown:
        i = 0
        for line in unknown:
    	    sentence_vector = model.embed_sentence(line)
    	    unknown_arrays_700[i] = sentence_vector
    	    i = i + 1

    lr_model_1 = pickle.load(open(lr_dir + "0.01_lr.sav", 'rb'))
    svm_model_1 = pickle.load(open(svm_dir + "c_0.1_svm.sav", 'rb'))

    Y_test_pred_lr_1 = (lr_model_1.predict_proba(unknown_arrays_700) >= 0.25).astype(int)
    Y_test_pred_lr_1 = Y_test_pred_lr_1[:, 1]
    Y_test_pred_s1 = svm_model_1.predict(unknown_arrays_700)
    print(Y_test_pred_lr_1)
    print(Y_test_pred_s1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            print("here\n")
            print("rcnn:")
            print(all_predictions.astype(int))
            print("logistic regression:")
            print(Y_test_pred_lr_1.astype(int))
            print("svm:")
            print(Y_test_pred_s1.astype(int))
            final_pred = np.add(Y_test_pred_lr_1.astype(int), Y_test_pred_s1.astype(int))
            final_pred = np.add(final_pred, all_predictions.astype(int))
            print(final_pred)
            #final_pred[final_pred >= 2.0] = 1.0
            final_pred[final_pred < 2] = 0
            final_pred[final_pred > 0] = 1
            final_pred=final_pred.astype(int) 
            print("final:")
            print(final_pred)




def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
