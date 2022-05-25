"""This module analyses the results of the model"""


def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
    """
    classifier: trained classifier
    tag: particular tag
    tags_classes: a list of classes names from MultiLabelBinarizer
    index_to_words: index_to_words transformation

    return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print(f"Tag:{tag}")

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]
    ]
    top_negative_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]
    ]

    formatted_top_positive_words = ", ".join(top_positive_words)
    formatted_top_negative_words = ", ".join(top_negative_words)
    print(f"Top positive words:{formatted_top_positive_words}")
    print(f"Top negative words:{formatted_top_negative_words}\n")


def print_words_for_tags(classifier, mlb, reversed_vocab):
    """
    classifier: trained classifier
    mlb: a MultiLabelBinarizer
    reversed_vocab: an index_to_words transformation

    return nothing, just call another method for different tags
    """
    print_words_for_tag(classifier, "c", mlb.classes, reversed_vocab)
    print_words_for_tag(classifier, "c++", mlb.classes, reversed_vocab)
    print_words_for_tag(classifier, "linux", mlb.classes, reversed_vocab)
