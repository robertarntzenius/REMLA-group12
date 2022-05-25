"""This module trains a model based on questions on stackoverflow
and tries to assign a tag to a question"""
from src import evaluation, multilabel, preprocessing
from src import transform_text_to_vector as transform

# Preprocessing
X_train, y_train, X_val, y_val, X_test = preprocessing.init_preprocessing()
tags_counts, words_counts = preprocessing.words_tags_count(X_train, y_train)

# Transform text to vector
## Bag of words
(
    X_train_mybag,
    X_val_mybag,
    X_test_mybag,
) = transform.bag_of_words(X_train, X_val, X_test, words_counts)
## TF-IDF
(
    X_train_tfidf,
    X_val_tfidf,
    X_test_tfidf,
    tfidf_vocab,
    tfidf_reversed_vocab,
) = transform.initialize_tfidf_vectorizer(X_train, X_val, X_test)

# Train model
## Init multilabel classifier
mlb, y_train, y_val = multilabel.init_multilabel_classifier(y_train, y_val, tags_counts)
## Bag of words
(
    y_val_predicted_labels_mybag,
    y_val_predicted_scores_mybag,
) = multilabel.multilabel_bag_of_words(X_train_mybag, y_train, X_val_mybag)
## TF-IDF
(
    y_val_predicted_labels_tfidf,
    y_val_predicted_scores_tfidf,
) = multilabel.multilabel_tfidf(X_train_tfidf, y_train, X_val_tfidf)

# Evaluate model
## Bag of words
evaluation.print_evaluation_scores_bag_of_words(y_val, y_val_predicted_labels_mybag)
evaluation.print_roc_auc_score_bag_of_words(y_val, y_val_predicted_scores_mybag)
## TF-IDF
evaluation.print_evaluation_scores_tfidf(y_val, y_val_predicted_labels_tfidf)
evaluation.print_roc_auc_score_tfidf(y_val, y_val_predicted_scores_tfidf)

# Analysis
