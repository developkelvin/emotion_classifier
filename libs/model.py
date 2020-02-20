import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import nltk
    

class Classifier:
    def __init__(self, cwd='dataset', master_fname='IEMOCAP_session1.csv'):
        super().__init__()
        master_fpath = os.path.join(cwd, master_fname)
        self._master_file = pd.read_csv(master_fpath)

    def load_model(self, model):
        self.__model = model

    def load_data(self, model):
        raise NotImplementedError()

    def preprocess_data(self):
        pass

    def predict(self):
        raise NotImplementedError()


class TextClassifier(Classifier):


    def __init__(self,):
        super().__init__(data_path=data_path)
        nltk.download('popular')

    def load_model(self, model):
        return super().load_model(model)


    def preprocess_data(self):
        base_dir = os.path.join('drive', 'My Drive', 'chatbot')
        text_dir = os.path.join(base_dir, 'dataset',  'iemocap_text')
        text_dset = pd.read_csv(os.path.join(text_dir, 'session1_text.csv'))
        texts = text_dset['text']
        labels = text_dset['emotion']
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
        print('num_of_sentence:',len(len_of_sentences), '\nmax_length:',sentence_maxlen,'\naverage_length:', sentence_lenavg)
        X = pad_sequences(sentences, maxlen=100)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = to_categorical(labels)
        y = np.asarray(labels)

        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def predict(self):
        pred = self.__model.predict(self.X_test)

        return pred

class VideoClassifier(Classifier):
    def __init__(self, data_path='dataset'):
        super().__init__(data_path=data_path)

    def load_model(self, model):
        return super().load_model(model)

    def load_data(self, model):
        pass

    def preprocess_data(self):
        pass

    def predict(self):
        pass
    
class AudioClassifier(Classifier):
    def __init__(self, data_path='dataset'):
        super().__init__(data_path=data_path)

    def load_model(self, model):
        return super().load_model(model)

    def preprocess_data(self):
        pass
    
    def load_data(self, model):
        pass

    def predict(self):
        pass