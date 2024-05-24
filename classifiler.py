import os

from nltk.corpus import stopwords
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

# # read positive
# fileObj = codecs.open( "pos.txt", "r", "utf_8_sig" )
# positive = fileObj.read()
# fileObj.close()
# positive = [value for value in positive.split('\r\n') if value]
#
# # read negative
# fileObj = codecs.open( "neg.txt", "r", "utf_8_sig" )
# negative = fileObj.read()
# fileObj.close()
# negative = [value for value in negative.split('\r\n') if value]

reader = CategorizedPlaintextCorpusReader('C:/Users/Timovey/Study/IAL/docs2',  r'(?!\.).*\.txt',
    cat_pattern=os.path.join(r'(neg|pos)', '.*'))

negative = ''
for file_id in reader.fileids(categories =['neg']):
    negative += reader.raw(fileids=file_id)
negative = [value for value in negative.split('\r\n') if value]

positive = ''
for file_id in reader.fileids(categories =['pos']):
    positive += reader.raw(fileids=file_id)
positive = [value for value in positive.split('\r\n') if value]

stopwords_1 = list(set(stopwords.words('russian')))

results = []
reviews = [(review, "1") for review in positive] + [(review, "0") for review in negative]

def calculate(texts, labels, use_stop_words, ngram_range, test_size):
    #Обучающая и тестовая выборки
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)

    if use_stop_words:
        vectorizer = CountVectorizer(stop_words=stopwords_1, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range)

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Обучение классификатора
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Предсказание
    predict = classifier.predict(X_test_vectorized)

    #Возращаем точность
    return accuracy_score(y_test, predict)

def run(reviews, total_test_size, using_stop_word, gram):
    textsWithCategory = reviews
    for test_size in total_test_size:
        random.shuffle(textsWithCategory)

        texts = [text for text, category in textsWithCategory]
        labels = [category for text, category in textsWithCategory]

        # Запуск эксперимента
        acc = calculate(texts, labels, using_stop_word, gram, test_size)

        stwrds = "Со стоп-словами; " if using_stop_word else "Без стоп-слов; "

        ngrm = "Грамма: "  + str(gram) + "; "

        test_size = "Размер  " + str(round(test_size * 100)) + "%"

        # Сохранение результатов
        print(test_size + "; Точность: " + str(round(acc, 2)) + "; " + stwrds + ngrm)


#размеры
total_test_size = [0.1, 0.15, 0.2, 0.25, 0.3]

run(reviews, total_test_size, True, (1, 1))
run(reviews, total_test_size, False, (1, 1))
run(reviews, total_test_size, True, (1, 2))
run(reviews, total_test_size, False, (1, 2))
