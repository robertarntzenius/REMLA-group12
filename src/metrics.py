class MetricHandler:
    """Metric Handler."""
    def __init__(self):
        self.no_predictions = 0
        self.correct_predictions = 0
        self.no_tags_predicted = 0
        self.no_tags_suggested = 0
        self.new_tags_predicted = 0

    def new_prediction(self, tags):
        """Receive prediction data."""
        self.new_tags_predicted = len(tags)

    def feedback(self, correct):
        """Add new prediction data."""
        self.no_predictions += 1
        self.no_tags_predicted += self.new_tags_predicted

        if correct:
            self.correct_predictions += 1
            self.no_tags_suggested += self.no_tags_predicted

    def suggested(self, tags):
        """Add number of newly suggested tags."""
        self.no_tags_suggested += int((len(tags) - 1) / 4)

    def get_no_predictions(self):
        """Get number of total predictions."""
        return self.no_predictions

    def get_no_correct_predictions(self):
        """Get number of correct predictions."""
        return self.correct_predictions

    def get_no_tags(self):
        """Get tags data."""
        return self.no_tags_predicted, self.no_tags_suggested - self.no_tags_predicted
