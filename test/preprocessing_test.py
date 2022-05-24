from src.preprocessing import *
import unittest


class TestPreprocessing(unittest.TestCase):

    def test_text_prepare(self):
        examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                    "How to free c++ memory vector<int> * arr?"]
        answers = ["sql server equivalent excels choose function",
                   "free c++ memory vectorint arr"]
        for ex, ans in zip(examples, answers):
            if text_prepare(ex) != ans:
                self.fail("Preprocessing failed")

    def test_get_most_common_tags(self):
        x_train = ["python", "java", "c++", "c", "c#", "javascript"]
        y_train = ["python", "java", "c++", "c", "c#", "javascript"]
        expected = [('a', 4), ('c', 4), ('p', 2)]
        most_common_tag = get_most_common_tags(x_train, y_train)
        self.assertEqual(most_common_tag, expected)


if __name__ == '__main__':
    unittest.main()
