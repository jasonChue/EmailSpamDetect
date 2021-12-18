import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.porter import PorterStemmer

try:
    data = pd.read_csv('emails.csv')
except:
    print("Some Error when get data from Dataset!")
finally:
    def pre_process():
        stopwords = nltk.corpus.stopwords.words('english')
        data.drop_duplicates(inplace = True)
        data['text'] = data['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ', text)).apply(
            lambda x: (x.lower()).split())
        data['text']=data['text'].map(lambda text: text[1:])
        ps = PorterStemmer()
        corpus = data['text'].apply(lambda text_list: ' '.join(list(map(lambda word: ps.stem(word), (
            list(filter(lambda text: text not in set(stopwords), text_list)))))))
        print(corpus.head(5))
        return corpus
    def draw_curve(y_test):
        pred_prob1 = model.predict_proba(x_test)
        fpr, tpr, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
        auc_score = roc_auc_score(y_test, pred_prob1[:,1])
        plt.plot(fpr, tpr, linestyle='dashed',color='orange', label='Multinomial Naive Bayes\nAccuracy {0:.4f}%'.format(auc_score))
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('ROC',dpi=300)
        plt.show();

    corpus = pre_process()
    X = data['text']
    y = data['spam']
    cv=CountVectorizer()
    X = cv.fit_transform(corpus.values).toarray()
    x_train, x_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    pickle.dump(model, open("spam.pkl", "wb"))
    pickle.dump(cv, open("vectorizer.pkl", "wb"))
    print('Created data file successfully')
    draw_curve(y_test)