from bs4 import BeautifulSoup

# Function to parse data from XML
def parse_xml_data(path):
    # load data from XML
    train = open(path).read()
    # Parse data
    train = BeautifulSoup(train, 'lxml')
    tweets = [twt.text.replace("\n", "") for twt in train.findAll('content')]
    labels = [labs.text for labs in train.findAll('value')]
    ids = [labs.text for labs in train.findAll('tweetid')] 

    return tweets, labels, ids


def get_polarity_lexicon(lex_path):
    lex_dict = {}
    with open(lex_path, "r") as lex_f:
        for line in lex_f.readlines():
            if not line.startswith("#"):
                words = line.split("\t")
                if len(words) == 2:
                    lex_dict[words[0]] = words[1][:-1]  # remove "\n" from polarity

    return lex_dict


def get_polarity_counts(tweets, lex_dict):
    pol_counts = []
    for tweet in tweets:
        aux_counts = [0, 0]
        for token in tweet.split(" "):
            tok_pol = lex_dict.get(token, "")
            if tok_pol == "positive":
                aux_counts[0] += 1
            elif tok_pol == "negative":
                aux_counts[1] += 1

        pol_counts.append(aux_counts)

    return pol_counts

if __name__ == "__main__":

    print("TEST FUNCIONS")

    print("\nparse_xml_data():")
    tweets, labels, ids = parse_xml_data('data/train.xml')
    for i in range(5):
        print(ids[i], labels[i], tweets[i])


    print("\nget_polarity_lexicon():")
    lex_dict = get_polarity_lexicon("utils/ElhPolar_esV1.lex")
    for i, (word, polarity) in enumerate(lex_dict.items()):
        if i == 5: break
        print(f"{word}: {polarity}")


    print("\nget_polarity_counts():")
    tweets_pol_counts = get_polarity_counts(tweets[:5], lex_dict)
    for pol_count in tweets_pol_counts:
        print(pol_count)
