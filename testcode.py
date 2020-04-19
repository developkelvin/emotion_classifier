from libs.model import *
from libs.voter import Voter

if __name__ == "__main__":
    # test_id = 'Ses01F_impro01_F012'

    # t = TextClassifier()
    # t.load_model(model_path='models/text/class3_model2.h5', tokenizer_path='models/text/class3_tokenizer.pickle', le_path='models/text/class3_label_encoder.pickle')
    # print(t.predict(test_id))

    v = VideoClassifier(test_id,session_nums=[1],include_neu=True)
    v.preprocess_data()
    v.load_model('models/video/test_model')
    print(v.predict())

    # a = AudioClassifier(f"{test_id}.wav")
    # a.load_model('models/audio/model.hdf5')
    # predc=a.predict()
    # print(predc)

    v = Voter(include_neu=False, text_weight=0.71, video_weight=0.72, audio_weight=0.7)
    print(v.voting('Ses01F_impro01_M013'))
    print(v.voting('Ses01F_impro02_F000'))
