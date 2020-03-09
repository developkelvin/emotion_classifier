from libs.model import *

if __name__ == "__main__":
    t = TextClassifier()
    t.load_model(model_path='models/text/text_cnn_lstm.h5', tokenizer_path='models/text/tokenizer_1.pickle', le_path='models/text/label_encoder_1.pickle')
    print(t.predict('Ses01F_impro01_F012'))

    v = VideoClassifier("Ses01F_impro01_F000",session_nums=[1],include_neu=True)
    v.preprocess_data()
    v.load_model('models/video/4class_5_model')
    print(v.predict())

    a = AudioClassifier("audio.wav")
    a.load_model('models/text/model.hdf5')
    predc=a.predict()
    print(predc)

