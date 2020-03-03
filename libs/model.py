import os
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import nltk

from abc import *
    

class Classifier(metaclass=ABCMeta):
    # input으로 어떤 session을 사용할 지 입력해주어야 합니다.
    def __init__(self):
        pass

    # 저장된 Keras나 TF, Pytorch 모델을 불러오는 코드. 반드시 구현 필요
    @abstractmethod
    def load_model(self, model):
        pass

    # 데이터를 외부에서 불러오는 작업이 필요하다면 구현
    def load_data(self, model):
        pass

    # 데이터 전처리가 필요하다면 구현
    def preprocess_data(self):
        pass

    # 클래스를 예측하는 코드. 반드시 구현 필요
    # 리턴 값은 string 타입의 감정이 되도록 (ang, neu, hap, ...) 구현할 것. 0, 1, 2등 숫자를 리턴하면 클래스 구분 어려움
    @abstractmethod
    def predict(self):
        pass


class TextClassifier(Classifier):

    def __init__(self, session_nums=[1,2,3], base_dir='.', include_neu=True):
        """입력받은 값을 검증하고 필요한 데이터 및 라이브러리를 로드합니다.
        
        Arguments:
            session_nums {list} -- [실험에 사용할 세션을 지정합니다.]
        
        Keyword Arguments:
            base_dir {str} -- [작업 공간을 지정합니다.] (default: {'.'})
            include_neu {bool} -- [참조할 데이터셋에 중립 감정을 포함할지 설정합니다.] (default: {True})
        
        Raises:
            TypeError: session_nums 파라미터는 반드시 리스트로 받아야 합니다.
            Exception: []
        """

        if not isinstance(session_nums, list):
            raise TypeError('session no must be list type')
        else:
            self.session_nums = session_nums

        # for session_num in session_nums:
        #     self.make_text_dataset(session_num, include_neu=include_neu)

        # base_dir = os.path.join('drive', 'My Drive', 'chatbot')
        text_dir = os.path.join(base_dir, 'dataset',  'iemocap_text')

        if len(session_nums) == 1:
            if include_neu:
                fname = f'session{session_nums[0]}_text_neu.csv'
            else:
                fname = f'session{session_nums[0]}_text.csv'
            self.text_dset = pd.read_csv(os.path.join(text_dir, f'session{session_nums[0]}_text.csv')) # 미리 전처리 해 놓은 데이터
        elif len(session_nums) > 1:
            if include_neu:
                dir_list = [os.path.join(text_dir, f'session{no}_text_neu.csv') for no in session_nums]
            else:
                dir_list = [os.path.join(text_dir, f'session{no}_text.csv') for no in session_nums]
            base = pd.read_csv(dir_list[0])
            for path in dir_list[1:]:
                df = pd.read_csv(path)
                base = base.append(df, ignore_index=True)
            
            self.text_dset = base
        else:
            raise Exception('Unknown Error')
        
        # 필요한 nltk 라이브러리 다운
        # nltk.download('popular')

    def load_model(self, model_path='models/text_model.h5', tokenizer_path='models/tokenizer.pickle', le_path='models/le.pickle'):
        """pre-train 된 모델을 불러옵니다.
        
        Keyword Arguments:
            model_path {str} -- 불러올 모델이 저장된 경로를 입력합니다. (default: {'models/text_model.h5'})
            tokenizer_path {str} -- tokenizer를 불러올 경로를 입력합니다. (default: {'models/tokenizer.pickle'})
            le_path {str} -- label_encoder를 불러올 결로를 입력합니다. (default: {'models/le.pickle'})
        """
        # reference : https://www.tensorflow.org/guide/keras/save_and_serialize
        import pickle

        self.model = keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(le_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def preprocess_X(self, texts, maxlen=100):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer, LancasterStemmer
        stop_words = set(stopwords.words('english'))
        pst = PorterStemmer()
        # 토큰화
        tokens = []
        for txt in texts:
            token = word_tokenize(txt)
            non_stopwords = [pst.stem(t) for t in token if not t in stop_words]
            tokens.append(non_stopwords)
        if self.tokenizer is None:
            raise ValueError('tokenizer is not loaded')
        tokenizer = self.tokenizer
        sentences = tokenizer.texts_to_sequences(tokens)
        X = pad_sequences(sentences, maxlen=maxlen)
        return X

    def preprocess_y(self, labels):
        if self.label_encoder is None:
            raise ValueError('label encoder is not loaded')
        labels = self.label_encoder.transform(labels)
        labels = to_categorical(labels)
        y = np.asarray(labels)

        return y

    def get_data(self, script_id):
        dset = self.text_dset
        row = dset[dset['script_id'] == script_id]
        text = row['text']
        emotion = row['emotion']

        return text, emotion

    def predict(self, script_id):
        """한 개의 script id에 대한 감정 예측을 진행합니다.
        
        Arguments:
            script_id {str} -- 예측할 스크립트 id를 입력합니다. (ex. )
        
        Returns:
            str -- 예측된 클래스를 리턴합니다. 리턴 값은 ang, hap과 같은 문자열 타입입니다.
        """
        text, emotion = self.get_data(script_id)
        X_test = self.preprocess_X(text)
        y_pred = self.model.predict_classes(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        return y_pred[0]

    
class VideoClassifier(Classifier):
    def __init__(self):
        pass
    def load_model(self, model):
        pass

    def load_data(self, model):
        pass

    def preprocess_data(self):
        pass

    def predict(self):
        pass
    
class AudioClassifier(Classifier):
    def __init__(self):
        pass

    def load_model(self):
        pass

    def preprocess_data(self):
        pass
    
    def load_data(self, model):
        pass

    def predict(self):
        pass