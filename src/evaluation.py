from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score as roc_auc


def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


def print_evaluation_scores_bag_of_words(y_val, y_val_predicted_labels_mybag):
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)


def print_evaluation_scores_tfidf(y_val, y_val_predicted_labels_tfidf):
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


def print_roc_auc_score_bag_of_words(y_val, y_val_predicted_scores_mybag):
    print('Roc-auc: ', roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo'))


def print_roc_auc_score_tfidf(y_val, y_val_predicted_scores_tfidf):
    print('Roc-auc: ', roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo'))


# coefficients = [0.1, 1, 10, 100]
# penalties = ['l1', 'l2']

# for coefficient in coefficients:
#     for penalty in penalties:
#         classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)
#         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
#         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
#         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#
#
# test_predictions = classifier_tfidf.predict(X_test_tfidf)######### YOUR CODE HERE #############
# test_pred_inversed = mlb.inverse_transform(test_predictions)
#
# test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))