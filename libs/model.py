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

    def __init__(self, session_nums, base_dir='.', include_neu=False):
        """입력받은 값을 검증하고 필요한 데이터 및 라이브러리를 로드합니다.
        
        Arguments:
            session_nums {[list]} -- [실험에 사용할 세션을 지정합니다.]
        
        Keyword Arguments:
            base_dir {str} -- [작업 공간을 지정합니다.] (default: {'.'})
            include_neu {bool} -- [중립 감정을 포함할지 설정합니다.] (default: {False})
        
        Raises:
            TypeError: [session_nums 변수는 반드시 리스트로 받아야 합니다.]
            Exception: []
        """

        if not isinstance(session_nums, list):
            raise TypeError('session no must be list type')
        else:
            self.session_nums = session_nums

        for session_num in session_nums:
            self.make_text_dataset(session_num, include_neu=include_neu)

        # base_dir = os.path.join('drive', 'My Drive', 'chatbot')
        text_dir = os.path.join(base_dir, 'dataset',  'iemocap_text')
        if len(session_nums) == 1:
            self.text_dset = pd.read_csv(os.path.join(text_dir, f'session{session_nums[0]}_text.csv')) # 미리 전처리 해 놓은 데이터
        elif len(session_nums) > 1:
            dir_list = [os.path.join(text_dir, f'session{no}_text.csv') for no in session_nums]
            base = pd.read_csv(dir_list[0])
            for path in dir_list[1:]:
                df = pd.read_csv(path)
                base.append(df, ignore_index=True)
            
            self.text_dset = base
        else:
            raise Exception('Unknown Error')

        all_labels = self.text_dset['emotion']
        le = LabelEncoder()
        self.le = le.fit(all_labels)
        
        # 필요한 nltk 라이브러리 다운
        nltk.download('popular')

    def load_model(self, model_path='models/text_model.h5'):
        """pre-train 된 모델을 불러옵니다.
        
        Keyword Arguments:
            model_path {str} -- 불러올 모델이 저장된 경로를 입력합니다. (default: {'models/text_model.h5'})
        """
        # reference : https://www.tensorflow.org/guide/keras/save_and_serialize
        self.model = keras.models.load_model(model_path)

    def preprocess_all_data(self):
        text_dset = self.text_dset
        def preprocess_X(texts, maxlen=100):
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

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(tokens)
            sentences = tokenizer.texts_to_sequences(tokens)
            len_of_sentences = [len(s) for s in sentences]
            sentence_maxlen = max(len_of_sentences)
            sentence_lenavg = sum(len_of_sentences)/len(len_of_sentences)
            print('문장의 개수:',len(len_of_sentences), '\n가장 긴 문장에 포함된 단어 수:',sentence_maxlen,'\n문장 평균 단어 수:', sentence_lenavg)
            X = pad_sequences(sentences, maxlen=maxlen)

            return X

        def preprocess_y(labels):
            labels = self.le.transform(labels)
            labels = to_categorical(labels)
            y = np.asarray(labels)

            return y
        
        train = text_dset[text_dset['use'] == 'train']
        val = text_dset[text_dset['use'] == 'validation']
        test = text_dset[text_dset['use'] == 'test']

        self.X_train = preprocess_X(train['text'])
        self.y_train = preprocess_y(train['emotion'])
        self.X_val = preprocess_X(val['text'])
        self.y_val = preprocess_y(val['emotion'])
        self.X_test = preprocess_X(test['text'])
        self.y_test = preprocess_y(test['emotion'])


    def preprocess_one_data(self, script_id):
        self.X_test = None
        self.y_test = None
        pass

    def predict_all_test(self):
        """[불러온 모델을 이용하여 클래스를 예측합니다.]
        
        Returns:
            [str] -- [예측된 클래스를 리턴합니다. 리턴 값은 ang, hap과 같은 문자열 타입입니다.]
        """
        pred = self.model.predict_classes(self.X_test)
        print(pred)
        pred = self.le.inverse_transform(pred)
        print(pred)
        return pred

    def predict(self, script_id):
        """한 개의 script id에 대한 감정 예측
        
        Arguments:
            script_id {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        self.preprocess_one_data(script_id)
        pred = self.model.predict_classes(self.X_test)
        pred = self.le.inverse_transform(pred)
        return pred

    def make_text_dataset(self, session_num, print_warn=False, include_neu=False):
        import os
        import glob
        import pandas as pd
        
        session_num = str(session_num)
        # Label 불러오기
        LABEL_PATH = os.path.join('dataset', 'master')
        LABEL_FNAME = f'session{session_num}-train-val-test.csv'

        fpath = os.path.join(LABEL_PATH, LABEL_FNAME)
        
        label = pd.read_csv(fpath)
        
        SCRIPT_PATH = os.path.join('dataset', 'IEMOCAP_full_release', f'Session{session_num}', 'dialog', 'transcriptions', '*')
        file_list = glob.glob(SCRIPT_PATH)
        file_list_script = [file for file in file_list if file.endswith(".txt")]

        rows = []

        for file in file_list_script:
            with open(file, 'r') as f:
                for line in f.readlines():
                    row = dict()
                    contents = line.split()
                    if len(contents) > 0:
                        try:
                            script_id = contents[0]
                            time_str = contents[1].replace('[','').replace(']','').replace(':','')
                            start_time = float(time_str.split('-')[0])
                            end_time = float(time_str.split('-')[1])
                            text = ' '.join(contents[2:])

                            row['script_id'] = script_id
                            row['start_time'] = start_time
                            row['end_time'] = end_time
                            row['text'] = text
                            row['label'] = ''

                            rows.append(row)
                        except ValueError as e:
                            if print_warn:
                                print('ValueError',e,file,line)
                        except IndexError as ie:
                            if print_warn:
                                print('IndexError', ie, line)
                            
        df = pd.DataFrame(rows)
        
        try:
            # name에 공백 제거
            df['script_id'] = df['script_id'].apply(lambda x:x.replace(' ', ''))
            label['name'] = label['name'].apply(lambda x:x.replace(' ', ''))
            
            if include_neu:
                pass
            else:
                # lable 중 neu 제외
                label = label[label['emotion'] != 'neu'].reset_index(drop=True)

            merged = pd.merge(df, label, left_on='script_id', right_on='name')[['script_id', 'start_time', 'end_time', 'text', "emotion", "use"]]
            if merged.shape[0] != label.shape[0]:
                print(f'merged : {merged.shape}, df : {df.shape}, label : {label.shape}')
                raise ValueError('size is changed before and after merged')
            
            text_dir = os.path.join('dataset',  'iemocap_text')
            merged.to_csv(os.path.join(text_dir, f'session{session_num}_text.csv'), index=False)
        except KeyError as ke:
            print('Data Not Exist') # 텍스트 데이터를 못불러와서 df가 비어있음

class VideoClassifier(Classifier):
    def __init__(self):
        pass
    def load_model(self, model):
        return super().load_model(model)

    def load_data(self, model):
        pass

    def preprocess_data(self):
        pass

    def predict(self):
        pass
    
class AudioClassifier(Classifier):
    def __init__(self):
        pass

    def load_model(self, model):
        return super().load_model(model)

    def preprocess_data(self):
        pass
    
    def load_data(self, model):
        pass

    def predict(self):
        pass