# emotion_classifier

## How to Implement
1. Repository를 Fork 혹은 다운로드 한 후, libs 폴더의 model.py 파일을 엽니다.
2. 본인에게 해당하는 Classifier 클래스를 확인합니다. (ex. VideoClassifier, AudioClassifier 등)
3. 그 중에서 필수적으로 구현해야 하는 메소드는 이전에 학습시킨 모델을 불러오는 load_model()과 script_id를 입력받아 이에 해당하는 클래스를 예측하는 predict() 메소드 입니다.
4. 다른 코드는 어떻게 작성하셔도 상관없지만 predict()메소드를 사용했을 때 'ang', 'hap', 'sad'와 같은 클래스를 string형태로 반환해 주어야 합니다.
5. 주호님이 train/validation/test로 나누어놓은 데이터는 dataset/master 폴더에 있습니다. 만약 필요하다면 참조하도록 코드를 작성하시면 됩니다.
6. 부수적으로 필요한 메타데이터가 있을 수 있습니다. 이는 dataset 폴더에 자유롭게 놓으시면 됩니다. (ex. 저는 iemocap_text라는 메타데이터 csv파일을 저장해두었습니다.)
7. dataset 폴더에 IEMOCAP_full_release/Session1/....와 같은 형태로 원본 데이터가 있다고 가정합니다.(실제 개발할 때 저기에 데이터셋을 두고 작업하시면 됩니다.)

## Example
잘 실행되는지 확인하기 위해 다음과 같은 테스트 코드를 사용할 수 있습니다.
```
t = TextClassifier()
t.load_model(model_path='models/text_cnn_lstm.h5')
print(t.predict('Ses01F_impro01_F012')) # 'sad'
```
