''' text preprocessing pipeline'''
''' Links:
https://stackoverflow.com/questions/8009882/how-to-read-a-large-file-line-by-line-in-python
https://machinelearningmastery.com/clean-text-machine-learning-python/
https://stackoverflow.com/questions/14301056/concatenating-lists-in-python-3
https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
https://stackoverflow.com/questions/9495007/indenting-code-in-sublime-text-2
'''
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os
from os import listdir
from os.path import isfile, join

stop_words = stopwords.words('english')
porter = PorterStemmer()

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "TJ-data-2"
abs_file_path = os.path.join(script_dir, rel_path)
onlyfiles = [f for f in listdir(abs_file_path) if isfile(join(abs_file_path, f)) and f.endswith(".txt")]

for text_file in onlyfiles:
	filename = join(abs_file_path, text_file)
	with open(filename) as f:
		for line in f:
			paragraph = []
			# split the abstract into sentences
			sentences = sent_tokenize(line)
			# split into words
			for sentence in sentences:
				tokens = word_tokenize(sentence)
				# convert to lower case
				tokens = [w.lower() for w in tokens]
				# remove punctuation from each word
				table = str.maketrans('', '', string.punctuation)
				stripped = [w.translate(table) for w in tokens]
				# remove remaining tokens that are not alphabetic
				words = [word for word in stripped if word.isalpha()]
				# filter out stop words
				words = [w for w in words if not w in stop_words]
				# stemming of words
				#stemmed = [porter.stem(word) for word in words] # for one single sentence
				paragraph.extend(words)
			# print the paragraph list element to one line to a new file as the preprocessed file
			with open("/Users/jintao/Documents/2019-Spring/Research/NLP-research/TJ-data-2/cleaned-nostem/" + text_file + "-prep", "a") as myfile:
				for item in paragraph:
					myfile.write(item + " ")
				myfile.write("\n")
				




