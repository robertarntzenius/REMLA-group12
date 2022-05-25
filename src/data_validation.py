# pylint: disable=E0401
"""This module validates the data"""
import tensorflow_data_validation as tfdv

from src import preprocessing


def main():
    train = preprocessing.read_data("../data/train.tsv")
    validation = preprocessing.read_data("../data/validation.tsv")
    # test = pd.read_csv("data/test.tsv", sep="\t")
    print(train.shape)
    train_stats = tfdv.generate_statistics_from_csv(data_location="../data/train.tsv", delimiter="\t")
    validation_stats = tfdv.generate_statistics_from_csv(data_location="../data/validation.tsv", delimiter="\t")
    stats = tfdv.get_statistics_html(
        rhs_statistics=train_stats,
        lhs_statistics=validation_stats,
        rhs_name="TRAIN",
        lhs_name="VALIDATE",
    )
    with open("../data/stats.html", "w") as f:
        f.write(stats)
    train_schema = tfdv.infer_schema(train_stats)
    tfdv.display_schema(train_schema)
    anomalies = tfdv.validate_statistics(validation_stats, schema=train_schema)
    tfdv.display_anomalies(anomalies)
    tfdv.write_stats_text(train_stats, "../data/stats.txt")
    tfdv.write_anomalies_text(anomalies, "../data/anomalies.txt")


if __name__ == "__main__":
    main()
