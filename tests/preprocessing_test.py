# pylint: disable=E0401,R0801
"""This module runs tests for preprocessing.py"""
import unittest

from src import preprocessing


class TestPreprocessing(unittest.TestCase):
    """This class runs unittests"""

    def test_text_prepare(self):
        """
        Test the text_prepare function
        """
        examples = [
            "SQL Server - any equivalent of Excel's CHOOSE function?",
            "How to free c++ memory vector<int> * arr?",
        ]
        answers = [
            "sql server equivalent excels choose function",
            "free c++ memory vectorint arr",
        ]
        for ex, ans in zip(examples, answers):
            if preprocessing.text_prepare(ex) != ans:
                self.fail("Preprocessing failed")

    def test_get_most_common_tags(self):
        """
        Tests the get_most_common_tags function
        """
        x_train = ["python", "java", "c++", "c", "c#", "javascript"]
        y_train = ["python", "java", "c++", "c", "c#", "javascript"]
        expected = [("a", 4), ("c", 4), ("p", 2)]
        most_common_tag = preprocessing.get_most_common_tags_or_words(
            x_train, y_train, True
        )
        self.assertEqual(most_common_tag, expected)


if __name__ == "__main__":
    unittest.main()
