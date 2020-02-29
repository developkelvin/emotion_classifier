from libs.model import *

if __name__ == "__main__":
    t = TextClassifier(session_nums=[1], include_neu=False)
    t.preprocess_data()
    t.load_model('models/text_cnn_lstm.h5')
    t.predict()