# pylint: disable=E0401,W1514
"""This module validates the data"""
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import _add_quotes
from tensorflow_metadata.proto.v0 import anomalies_pb2


def main():
    """
    Validate the data and create reports
    """
    # test = pd.read_csv("data/test.tsv", sep="\t")
    train_stats = tfdv.generate_statistics_from_csv(
        data_location="data/train.tsv", delimiter="\t"
    )
    validation_stats = tfdv.generate_statistics_from_csv(
        data_location="data/validation.tsv", delimiter="\t"
    )
    stats = tfdv.get_statistics_html(
        rhs_statistics=train_stats,
        lhs_statistics=validation_stats,
        rhs_name="TRAIN",
        lhs_name="VALIDATE",
    )
    with open("reports/stats.html", "w") as file:
        file.write(stats)
    train_schema = tfdv.infer_schema(train_stats)
    tfdv.display_schema(train_schema)
    anomalies = tfdv.validate_statistics(validation_stats, schema=train_schema)
    get_anomalies_markdown(anomalies)
    tfdv.write_stats_text(train_stats, "reports/stats.txt")


def get_anomalies_markdown(anomalies: anomalies_pb2.Anomalies):
    """
    Returns a DataFrame containing the input anomalies.
    Args: anomalies: An Anomalies protocol buffer.
    Returns: nothing, just create a md file of the anomalies.
    """
    anomaly_rows = []
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        anomaly_rows.append(
            [
                _add_quotes(feature_name),
                anomaly_info.short_description,
                anomaly_info.description,
            ]
        )
    if anomalies.HasField("dataset_anomaly_info"):
        anomaly_rows.append(
            [
                "[dataset anomaly]",
                anomalies.dataset_anomaly_info.short_description,
                anomalies.dataset_anomaly_info.description,
            ]
        )

    with open("reports/anomalies.md", "w") as anomd:
        anomd.write("## Anomalies report of the data")
        if len(anomaly_rows) == 0:
            anomd.write("No anomalies found!")
        else:
            for row in anomaly_rows:
                anomd.write(
                    f"Feature name: {row[0]}, "
                    f"Anomaly short description: {row[1]}, "
                    f"Anomaly ling description: {row[2]}"
                )


if __name__ == "__main__":
    main()
