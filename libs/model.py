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
import librosa
from keras.models import load_model
    

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
    def __init__(self,video_name,session_nums=[1, 2, 3],include_neu=True):

        from moviepy.editor import VideoFileClip

        # 현재 작업 폴더
        self.work_directory = os.getcwd()
        self.work_directory = self.work_directory + '/dataset/'

        # 자료들이 담길 폴더 생성
        self.temp_directory = self.work_directory + 'iemocap_video/image_model/'
        if not os.path.isdir(self.temp_directory):
            os.makedirs(self.work_directory + 'iemocap_video/image_model')

        # 결과 저장 폴더(현재 작업 폴더 내에 생성
        self.result_directory = self.temp_directory + 'clip/'
        if not os.path.isdir(self.result_directory):
            os.makedirs(self.temp_directory + 'clip')

        # 클립이 프레임으로 쪼개져서 저장될 폴더
        self.frame_directory = self.temp_directory + 'Frames/'
        if not os.path.isdir(self.frame_directory):
            os.makedirs(self.temp_directory + 'Frames')

        # 프레임이 crop되고 저장될 폴더
        self.crop_directory = self.temp_directory + 'Crop/'
        if not os.path.isdir(self.crop_directory):
            os.makedirs(self.temp_directory + 'Crop')

        # crop이미지가 전처리 되고 저장될 폴더
        self.gray_224_directory = self.temp_directory + 'Preprocessing/'
        if not os.path.isdir(self.gray_224_directory):
            os.makedirs(self.temp_directory + 'Preprocessing')

        # gray이미지가 histogram equalization되고 저장될 폴더
        self.hist_directory = self.temp_directory + 'hist/'
        if not os.path.isdir(self.hist_directory):
            os.makedirs(self.temp_directory + 'hist')

        # 원본 video가 있는 폴더
        video_directory = self.work_directory + 'iemocap_video/video/'

        self.target_clip_name = video_name
        self.target_clip_emotion = ''

        self.include_neu = include_neu

        print("Processing - Video split to Clip and Frame")

        # csv파일을 읽어 참조할수있는 list 생성

        data = pd.read_csv(
            self.work_directory + 'master/' + f"session{session_nums[0]}-train-val-test.csv")

        start_list = data['start']
        end_list = data['end']
        title_list = data['name']

        video_index = 0
        for index, title in enumerate(title_list):
            if title == video_name:
                video_index = index

        # video파일을 start time, end time 이용 clip 생성
        # 이곳 경로의 경우 DB있는 폴더 경로로 해주시면 됩니다.
        clip = VideoFileClip(video_directory + video_name[:-5] + ".avi").subclip(start_list[video_index],
                                                                                 end_list[video_index])

        # 생성된 clip 저장
        clip.to_videofile(self.result_directory + video_name + ".mp4", codec="libx264", temp_audiofile='temp-audio.m4a',
                          remove_temp=True, audio_codec='aac')

    def read_clip(self,subset_dir):
        # 폴더 안에있는 처리된 이미지들의 이름을 읽어 한 비디오에서 나오는 각 클립 2개를 구분해주는 코드입니다.
        filename_list = os.listdir(subset_dir)
        file_name_re = filename_list[0].replace("_" + filename_list[0].split('_')[-1], "")
        file_name_re = file_name_re.replace("_" + file_name_re.split('_')[-1], "")

        i = 0
        clip_list = []
        clip_num_list = []
        for j, filename in enumerate(filename_list):
            if i == 0:
                file_name_re = filename.replace("_" + filename.split('_')[-1], "")
                file_name_re = file_name_re.replace("_" + file_name_re.split('_')[-1], "")
                clip_list.append(file_name_re)
                clip_num_list.append(j)
                i = i + 1
            file_name_re = filename.replace("_" + filename.split('_')[-1], "")
            file_name_re = file_name_re.replace("_" + file_name_re.split('_')[-1], "")

            if file_name_re not in clip_list:
                clip_list.append(file_name_re)
                clip_num_list.append(j)
                i = i + 1

        clip_num_list_2 = []

        for j in range(len(clip_num_list)):
            if j == len(clip_num_list) - 1:
                clip_num_list_2.append(len(filename_list) - clip_num_list[j])
            else:
                clip_num_list_2.append(clip_num_list[j + 1] - clip_num_list[j])

        return clip_list, clip_num_list, clip_num_list_2

    def read_subset(self,subset_dir, file_start_num, file_frame_num, file_now_frame):
        # histogram처리된 이미지가 저장되어있는 폴더를 읽어서 모델에 넣기위해 라벨링 해주는 부분입니다.

        from skimage.io import imread
        from skimage.transform import resize

        # Read trainval data
        filename_list = os.listdir(subset_dir)

        set_size = 1

        # Pre-allocate data arrays
        X_set = np.empty((set_size, 224, 224, 3), dtype=np.float32)  # (N, H, W, 3)
        check = 0
        for i, filename in enumerate(filename_list):
            if i >= file_start_num and i < file_start_num + file_frame_num:
                if i == file_start_num + file_now_frame:
                    file_path = os.path.join(subset_dir, filename)
                    img = imread(file_path)  # shape: (H, W, 3), range: [0, 255]
                    img = resize(img, (224, 224, 3), mode='constant').astype(np.float32)  # (256, 256, 3), [0.0, 1.0]
                    X_set[check] = img
                    check = check + 1

            if check != 0:
                break;

        return X_set

    def find_max(self,y_list):
        maxValue = y_list[0]
        max_i = 0
        for i in range(1, len(y_list)):
            if maxValue < y_list[i]:
                maxValue = y_list[i]
                max_i = i
        return max_i

    def split_Female_Male(self,name):
        # 각 동영상이 측정되는 사람(왼쪽)과 오른쪽 사람이 성별이 다르기에 두 얼굴을 검출해서 두개 각자 저장할수 있도록 이름을 바꾸어주는 함수입니다.
        if name.split('_')[-3][0] == 'F':
            temp = name.split('_')[-3]
            change = name.split('_')[-3]
            change = change.replace("F", "M")
            name = name.split(temp)[0] + change + name.split(temp)[1]
        else:
            temp = name.split('_')[-3]
            change = name.split('_')[-3]
            change = change.replace("M", "F")
            name = name.split(temp)[0] + change + name.split(temp)[1]

        return name

    def load_model(self, model_path):

        from keras import optimizers

        json_file = open(model_path + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(model_path + ".h5")
        self.model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True),
                             metrics=['accuracy'])

    def preprocess_data(self):

        import face_recognition
        import cv2
        from PIL import Image

        source_directory_files = os.listdir(self.result_directory)

        for source_directory_file in source_directory_files:
            source_file_path = "%s/%s" % (self.result_directory, source_directory_file)
            vid = cv2.VideoCapture(source_file_path)
            index = 1

            while (True):
                ret, frame = vid.read()

                if not ret:
                    break
                name = self.frame_directory + source_directory_file[:-4] + '_frame_' + str(index) + '.jpg'
                cv2.imwrite(name, frame)

                index += 1

        source_directory = self.frame_directory
        output_directory = self.crop_directory

        source_directory_files = os.listdir(source_directory)

        for i, source_directory_file_name in enumerate(source_directory_files):
            source_file_path = "%s/%s" % (source_directory, source_directory_file_name)

            image = face_recognition.load_image_file(source_file_path)
            faces = face_recognition.face_locations(image)
            extracted_image_file_paths = []
            j = 1
            for p in range(len(faces)):
                if p == 0:
                    top, right, bottom, left = faces[p]
                    extracted_image = Image.fromarray(image[top:bottom, left:right])
                    extracted_image_file_path = source_directory_file_name[:-4] + '-' + str(j) + ".jpg"
                    extracted_image.save("%s/%s" % (output_directory, extracted_image_file_path), cmap='gray')
                    extracted_image_file_paths.append(extracted_image_file_path)
                    j = j + 1
                else:
                    top, right, bottom, left = faces[p]
                    extracted_image = Image.fromarray(image[top:bottom, left:right])
                    extracted_image_file_path = self.split_Female_Male(source_directory_file_name)[:-4] + '-' + str(
                        j) + ".jpg"
                    extracted_image.save("%s/%s" % (output_directory, extracted_image_file_path))
                    extracted_image_file_paths.append(extracted_image_file_path)
                    j = j + 1

        source_directory = self.crop_directory
        source_directory_files = os.listdir(source_directory)

        j = 1

        for source_directory_file in source_directory_files:
            source_file_path = "%s/%s" % (source_directory, source_directory_file)
            img = cv2.imread(source_file_path)
            img_height, img_width = img.shape[:2]
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_height >= 224:
                resi_img = cv2.resize(gray_img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            else:
                resi_img = cv2.resize(gray_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

            name = self.gray_224_directory + source_directory_file[:-4] + ".jpg"
            cv2.imwrite(name, resi_img)
            j = j + 1

        source_directory = self.gray_224_directory
        source_directory_files = os.listdir(source_directory)

        j = 1

        for source_directory_file in source_directory_files:
            source_file_path = "%s/%s" % (source_directory, source_directory_file)
            img = cv2.imread(source_file_path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
            img2 = clahe.apply(gray_image)
            name = self.hist_directory + source_directory_file[:-4] + ".jpg"
            cv2.imwrite(name, img2)
            j = j + 1

    def predict(self):
        testval_dir = self.hist_directory

        if self.include_neu:
            file_name_list, file_start_num_list, file_frame_num_list = self.read_clip(testval_dir)

            for k in range(len(file_name_list)):
                y_frame = []
                y_label = [0, 0, 0, 0]
                for file_now_frame in range(file_frame_num_list[k]):
                    X_testval = self.read_subset(testval_dir, file_start_num_list[k], file_frame_num_list[k],
                                                 file_now_frame)
                    y = self.model.predict(X_testval)
                    y_s = self.find_max(y[0])
                    y_frame.append(y_s)
                    y_label[y_s] = y_label[y_s] + 1
                final_y = self.find_max(y_label)
                final_label = ""
                if final_y == 0:
                    final_label = "neu"
                elif final_y == 1:
                    final_label = "hap"
                elif final_y == 2:
                    final_label = "ang"
                elif final_y == 3:
                    final_label = "sad"

                if file_name_list[k] == self.target_clip_name:
                    self.target_clip_emotion = final_label
                    print(self.target_clip_name  + " : "  + final_label)
            return self.target_clip_emotion
        else:
            file_name_list, file_start_num_list, file_frame_num_list = self.read_clip(testval_dir)

            for k in range(len(file_name_list)):
                y_frame = []
                y_label = [0, 0, 0, 0]
                for file_now_frame in range(file_frame_num_list[k]):
                    X_testval = self.read_subset(testval_dir, file_start_num_list[k], file_frame_num_list[k],
                                                 file_now_frame)
                    y = self.model.predict(X_testval)
                    y_s = self.find_max(y[0])
                    y_frame.append(y_s)
                    y_label[y_s] = y_label[y_s] + 1
                final_y = self.find_max(y_label)
                final_label = ""
                if final_y == 0:
                    final_label = "hap"
                elif final_y == 1:
                    final_label = "ang"
                elif final_y == 2:
                    final_label = "sad"

                if file_name_list[k] == self.target_clip_name:
                    self.target_clip_emotion = final_label
                    print(self.target_clip_name  + " : "  + final_label)
            return self.target_clip_emotion


class AudioClassifier(Classifier):
    def __init__(self,audio):
        self.wav=audio


    def load_model(self,path):
        self.model = load_model(path)


    def score(self,predictions):
        fear = 0
        surprise = 0
        nutral = 0
        angry = 0
        sad = 0
        happy = 0
        EMOTIONS = ['Sad', 'Angry', 'Happy']
        for i in range(len(predictions)):
            sad += predictions[i][0]
            angry += predictions[i][1]
            # nutral += predictions[i][2]
            happy += predictions[i][2]
        # score = [nutral, angry, sad, happy]
        score = [angry, sad, happy]
        index = np.argmax(score)
        return EMOTIONS[index]

    def windows(self,data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size / 2)

    def extract_features_array(self,filename, bands=60, frames=41):
        window_size = 512 * (frames - 1)
        log_specgrams = []
        sound_clip, s = librosa.load(filename)

        for (start, end) in self.windows(sound_clip, window_size):
            if (len(sound_clip[int(start):int(end)]) == int(window_size)):
                signal = sound_clip[int(start):int(end)]

                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

        return np.array(features)


    def predict(self):

        feature_x = self.extract_features_array(self.wav, bands=60, frames=41)

        predictions = self.model.predict(feature_x)

        # score(predictions)
        # predictions = prediction[0]
        # ind = np.argpartition(predictions[0], -2)[-2:]
        # ind[np.argsort(predictions[0][ind])]
        # ind = ind[::-1]
        # print "Actual:", actual, " Top guess: ", EMOTIONS[ind[0]], " (",round(predictions[0,ind[0]],3),")"
        # print "2nd guess: ", EMOTIONS[ind[1]], " (",round(predictions[0,ind[1]],3),")"
        # print(predictions)
        # index = np.argmax(predictions)
        # print("-------------------------------------------------------")
        # print(filename[:-4] + " Predicted : " + score(predictions))
        # print("=======================================================\n")
        return self.score(predictions)
