from libs.voter import Voter
import pandas as pd
import numpy as np
import os
if __name__ == "__main__":
    # class3
    # session 1만 테스트

    master = pd.read_csv(os.path.join('dataset', 'master', 'session1-train-val-test.csv'))
    master = master[['name', 'emotion', 'use']]

    master = master[master['use']=='test']

    master_class3 = master[master['emotion']!='neu'][['name', 'emotion']]

    v = Voter(include_neu=False, text_weight=0.71, video_weight=0.72, audio_weight=0.7)
    y_pred = []
    y_true = master_class3['emotion']

    for i in range(master_class3.shape[0]):
        name = master_class3.iloc[i]['name']
        print(name)
        y_pred.append(v.voting(name))

    import pickle
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    with open('y_pred.pickle', 'wb') as f:
        pickle.dump(y_pred, f)

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
        

    