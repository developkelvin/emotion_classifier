import unittest
from libs.model import *

class ModelTest(unittest.TestCase):
    def test_text_classifier(self):
        t = TextClassifier(session_nums=[1], include_neu=False)
        t.preprocess_data()
        t.load_model('models/text_cnn_lstm.h5')
        print(t.predict())