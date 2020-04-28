import sys
import numpy as np
from nltk.tokenize import TweetTokenizer
from utils.my_functions import parse_xml_data, get_polarity_lexicon, get_polarity_counts, get_one_hot_labels
from utils.models import get_dnn_model
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from scipy.sparse import hstack
from tensorflow.keras.callbacks import ModelCheckpoint

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
# Get one hot labels for dnn
train_labels_onehot = get_one_hot_labels(train_labels)
dev_labels_onehot = get_one_hot_labels(dev_labels)

######################
# Experiments config #
######################

# Build tokenizer (removes upper case )
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
# Make a callable function for the vectorizer
tok_func = lambda s: tokenizer.tokenize(s)

# Auxiliary variables to store the best config
best_accuracy = 0
best_macro = None
best_config = []
# Range of different vectorizers
vectorizer_types = range(4)
# Kernel types for SVM classifier
kernel_types = ["linear", "poly", "rbf", "sigmoid"]
# Create the list of classifiers. All the SVM variants plus DNN
classifiers = kernel_types + ["dnn"]
# Regularization param for SVM classifier
C = 0.1
# DNN train parameters
batch_size = 64
epochs = 100
# Auxiliary variables to store the best model
best_config = []
best_acc = 0

'''
Loop for trying diferent combinations if vectorizers and classifiers
'''
for vectorizer_type in vectorizer_types:
    for classifier_type in classifiers:

        ##############
        # VECTORIZER #
        ##############

        if vectorizer_type == 0:
            vectorizer_name = "CountVectorizer"
            vectorizer = CountVectorizer(tokenizer=tok_func, ngram_range=(1,1))

        elif vectorizer_type == 1:
            if classifier_type == "dnn": continue
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

        if classifier_type == "dnn":
            # From scipy sparse to regular array
            train_vectors = train_vectors.toarray()
            dev_vectors = dev_vectors.toarray()

        ##############
        # CLASSIFIER #
        ##############

        # Build the classifier
        if classifier_type == "dnn":
            classifier = get_dnn_model(input_shape=train_vectors.shape[1:])
        elif classifier_type == "linear":
            classifier = LinearSVC(C=C)
        else:
            classifier = SVC(C=C, kernel=classifier_type)

        #########
        # TRAIN # 
        #########

        if classifier_type == "dnn":
            # Callback to store best model
            best_model_path = f"saved_models/{vectorizer_name}_bestloss"
            ckpt_callback = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
            classifier.fit(train_vectors, train_labels_onehot, batch_size, epochs, validation_data=(dev_vectors, dev_labels_onehot), callbacks=[ckpt_callback])
        else:
            classifier.fit(train_vectors, train_labels)

        ############
        # EVALUATE #
        ############

        # Load best model checkpoint (by validation loss)
        if classifier_type == "dnn":
            classifier.load_weights(best_model_path)
        # Get predictions
        dev_preds = classifier.predict(dev_vectors)

        # Compute stats of the results
        if classifier_type == "dnn":
            dev_labels_num = np.argmax(dev_labels_onehot, axis=1)
            dev_preds_num = np.argmax(dev_preds, axis=1)
            accuracy = metrics.accuracy_score(dev_labels_num, dev_preds_num)
            macro = metrics.precision_recall_fscore_support(dev_labels_num, dev_preds_num, average='macro')
        else:
            accuracy = metrics.accuracy_score(dev_labels, dev_preds)
            macro = metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='macro')

        # Show stats
        if classifier_type == "dnn":
            print(f"\nResults of vectorizer {vectorizer_name} using a dnn:")
        else:
            print(f"\nResults of vectorizer {vectorizer_name} using a SVM with kernel type {classifier_type}:")
        print(f"acc   = {accuracy}")
        print(f"macro = {macro}")
        #print(f"micro = {metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='micro')}")
        #print(metrics.classification_report(dev_labels, dev_preds))

        # Check if we get a new best model to store it
        if accuracy > best_acc:
            best_config = [vectorizer_name, classifier_type]
            best_acc = accuracy
            best_macro = macro

# Show the best model
print("\nThe best model config with is:")
print(f"\tvectorizer = {best_config[0]}")
print(f"\tclassifier = {best_config[1]}")
print(f"\tresults: accuracy={best_acc} - macro={best_macro}")
