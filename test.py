import sys
import numpy as np
from nltk.tokenize import TweetTokenizer
from utils.my_functions import parse_xml_data, get_polarity_lexicon, get_polarity_counts
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.sparse import hstack

################
# DATA LOADING #
################

# Paths
train_data_path = "data/train.xml"
dev_data_path = "data/dev.xml"
test_data_path = "data/test.xml"
polarity_lex_path = "utils/ElhPolar_esV1.lex"

# Load training data
train_tweets, train_labels, train_ids = parse_xml_data(train_data_path)
# Load development data
dev_tweets, dev_labels, dev_ids = parse_xml_data(dev_data_path)
# Load test data
test_tweets, test_labels, test_ids = parse_xml_data(test_data_path)
# Load polarity lexicon dictionary
lex_dict = get_polarity_lexicon(polarity_lex_path)
# Get polarity word counts for each tweet
train_polarities = get_polarity_counts(train_tweets, lex_dict)
dev_polarities = get_polarity_counts(dev_tweets, lex_dict)
test_polarities = get_polarity_counts(test_tweets, lex_dict)

#####################
# Classifier config #
#####################

# Build tokenizer (removes upper case )
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
# Make a callable function for the vectorizer
tok_func = lambda s: tokenizer.tokenize(s)

#############################
# VECTORIZER and CLASSIFIER #
#############################

vectorizer = HashingVectorizer(tokenizer=tok_func, ngram_range=(1,1))

# Vectorize the tweets
train_vectors = vectorizer.fit_transform(train_tweets)
dev_vectors = vectorizer.transform(dev_tweets)
test_vectors = vectorizer.transform(test_tweets)

# Add lexicon information
train_vectors = hstack((train_vectors, train_polarities))
dev_vectors = hstack((dev_vectors, dev_polarities))
test_vectors = hstack((test_vectors, test_polarities))

classifier = LinearSVC(C=0.1)

#########
# TRAIN # 
#########

classifier.fit(train_vectors, train_labels)

############
# EVALUATE #
############

# Get predictions
dev_preds = classifier.predict(dev_vectors)

accuracy = metrics.accuracy_score(dev_labels, dev_preds)
macro = metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='macro')

print(f"\tDev results: accuracy={accuracy} - macro={macro}")

########
# TEST #
########

# Get predictions
test_preds = classifier.predict(test_vectors)

with open("test_predictions.txt", "+w") as f:
    for tweet_id, pred in zip(test_ids, test_preds):
        f.write(f"{tweet_id}\t{pred}\n")

