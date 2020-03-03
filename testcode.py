from libs.model import *

if __name__ == "__main__":
    t = TextClassifier()
    t.load_model(model_path='models/text_cnn_lstm.h5', tokenizer_path='models/tokenizer_1.pickle', le_path='models/label_encoder_1.pickle')
    print(t.predict('Ses01F_impro01_F012'))