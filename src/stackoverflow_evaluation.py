"""
Automatic evaluation of the generated StackOverflow dataset.
"""
# pylint: disable=W0612
# pylint: disable=R0914
# pylint: disable=R0801
import evaluation
import multilabel
import preprocessing
import transform_text_to_vector as transform


def run_generated():
    """
    Train the model.
    """

    # Preprocessing
    x_train, y_train, x_val, y_val, x_test = preprocessing.init_preprocessing(
        "generated"
    )
    # Print length of training, validation and test set
    print("Length of training set:", len(x_train))
    print("Length of validation set:", len(x_val))
    print("Length of test set:", len(x_test))
    tags_counts, words_counts = preprocessing.words_tags_count(x_train, y_train)

    # Transform text to vector
    ## Bag of words
    (
        x_train_mybag,
        x_val_mybag,
        x_test_mybag,
    ) = transform.bag_of_words(x_train, x_val, x_test, words_counts)
    ## TF-IDF
    (
        x_train_tfidf,
        x_val_tfidf,
        x_test_tfidf,
        tfidf_vocab,
        tfidf_reversed_vocab,
    ) = transform.initialize_tfidf_vectorizer(x_train, x_val, x_test)

    # Train model
    ## Init multilabel classifier
    mlb, y_train, y_val = multilabel.init_multilabel_classifier(
        y_train, y_val, tags_counts
    )
    ## Bag of words
    (
        y_val_predicted_labels_mybag,
        y_val_predicted_scores_mybag,
    ) = multilabel.multilabel_bag_of_words(x_train_mybag, y_train, x_val_mybag)
    ## TF-IDF
    (
        y_val_predicted_labels_tfidf,
        y_val_predicted_scores_tfidf,
    ) = multilabel.multilabel_tfidf(x_train_tfidf, y_train, x_val_tfidf)

    # Evaluate model
    ## Bag of words
    evaluation.print_evaluation_scores_bag_of_words(
        y_val, y_val_predicted_labels_mybag, is_stackoverflow=True
    )
    evaluation.print_roc_auc_score_bag_of_words(y_val, y_val_predicted_scores_mybag)
    ## TF-IDF
    evaluation.print_evaluation_scores_tfidf(
        y_val, y_val_predicted_labels_tfidf, is_stackoverflow=True
    )
    evaluation.print_roc_auc_score_tfidf(y_val, y_val_predicted_scores_tfidf)


if __name__ == "__main__":
    run_generated()
