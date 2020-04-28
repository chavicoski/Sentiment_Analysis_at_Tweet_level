import numpy as np
from nltk.tokenize import TweetTokenizer
from utils.my_functions import parse_xml_data, get_polarity_lexicon, get_polarity_counts
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
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
# Load polarity lexicon dictionary
lex_dict = get_polarity_lexicon(polarity_lex_path)
# Get polarity word counts for each tweet
train_polarities = get_polarity_counts(train_tweets, lex_dict)
dev_polarities = get_polarity_counts(dev_tweets, lex_dict)

######################
# Experiments config #
######################

# Build tokenizer (removes upper case )
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
# Make a callable function for the vectorizer
tok_func = lambda s: tokenizer.tokenize(s)

# Auxiliary variables to store the best config
best_accuracy = 0
best_config = []
# Range of different vectorizers
vectorizer_types = range(4)
# Kernel types for classifier
kernel_types = ["linear", "poly", "rbf", "sigmoid"]
# Regularization param for classifier
C = 0.1
# Auxiliary variables to store the best model
best_config = []
best_acc = 0

'''
Loop for trying diferent combinations if vectorizers and classifiers
'''
for vectorizer_type in vectorizer_types:
    for kernel_type in kernel_types:

        ##############
        # VECTORIZER #
        ##############

        if vectorizer_type == 0:
            vectorizer_name = "CountVectorizer"
            vectorizer = CountVectorizer(tokenizer=tok_func, ngram_range=(1,1))

        elif vectorizer_type == 1:
            vectorizer_name = "HashingVectorizer"
            vectorizer = HashingVectorizer(tokenizer=tok_func, ngram_range=(1,1))

        elif vectorizer_type == 2:
            vectorizer_name = "TfidfTransformer"
            vectorizer = Pipeline([("count", CountVectorizer(tokenizer=tok_func, ngram_range=(1,1))),
                                   ("tfid", TfidfTransformer())])

        elif vectorizer_type == 3:
            vectorizer_name = "TfidfVectorizer"
            vectorizer = TfidfVectorizer(tokenizer=tok_func, ngram_range=(1,1))

        # Apply vectorizer to train data
        train_vectors = vectorizer.fit_transform(train_tweets)
        # Apply vectorizer to train data
        dev_vectors = vectorizer.transform(dev_tweets)

        # Add the polarity counts to the vectors
        train_vectors = hstack((train_vectors, train_polarities))
        dev_vectors = hstack((dev_vectors, dev_polarities))

        ##############
        # CLASSIFIER #
        ##############

        # Build the classifier
        if kernel_type == "linear":
            classifier = LinearSVC(C=C)
        else:
            classifier = SVC(C=C, kernel=kernel_type)

        #########
        # TRAIN # 
        #########

        classifier.fit(train_vectors, train_labels)

        ############
        # EVALUATE #
        ############

        # Get predictions
        dev_preds = classifier.predict(dev_vectors)

        # Compute stats of the results
        accuracy = metrics.accuracy_score(dev_labels, dev_preds)
        macro = metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='macro')

        # Show stats
        print(f"\nResults of vectorizer {vectorizer_name} with kernel type {kernel_type}:")
        print(f"acc   = {accuracy}")
        print(f"macro = {macro}")
        #print(f"micro = {metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='micro')}")
        #print(metrics.classification_report(dev_labels, dev_preds))

        # Check if we get a new best model to store it
        if accuracy > best_acc:
            best_config = [vectorizer_name, kernel_type]
            best_acc = accuracy

# Show the best model
print(f"\nThe best model config with {best_acc} of accuracy is:")
print(f"\tvectorizer = {best_config[0]}")
print(f"\tkernel type = {best_config[1]}")
