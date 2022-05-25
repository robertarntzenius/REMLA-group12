# pylint: disable=E0401
"""This module validates the data"""
import tensorflow_data_validation as tfdv

from src import preprocessing

train = preprocessing.read_data("../data/train.tsv")
validation = preprocessing.read_data("../data/validation.tsv")
# test = pd.read_csv("data/test.tsv", sep="\t")

train_stats = tfdv.generate_statistics_from_dataframe(train)
validation_stats = tfdv.generate_statistics_from_dataframe(validation)
tfdv.visualize_statistics(
    rhs_statistics=train_stats,
    lhs_statistics=validation_stats,
    rhs_name="TRAIN",
    lhs_name="VALIDATE",
)
train_schema = tfdv.infer_schema(train_stats)
tfdv.display_schema(train_schema)
anomalies = tfdv.validate_statistics(validation_stats, schema=train_schema)
tfdv.display_anomalies(anomalies)
