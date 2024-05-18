from nltk.corpus import stopwords
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


targets = ['хороший', 'отличный', 'супер', 'шикарный', 'плохой', 'ужасный',  'горячий', 'занижен']

# read positive
fileObj = codecs.open( "pos.txt", "r", "utf_8_sig" )
positive = fileObj.read()
fileObj.close()
positive = [value for value in positive.split('\r\n') if value]

# read negative
fileObj = codecs.open( "neg.txt", "r", "utf_8_sig" )
negative = fileObj.read()
fileObj.close()
negative = [value for value in negative.split('\r\n') if value]

stopwords_1 = list(set(stopwords.words('russian')))

results = []
reviews = [(review, "1") for review in positive] + [(review, "0") for review in negative]

def run_experiment(texts, labels, use_stop_words, ngram_range, test_size):
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)

    # Векторизация текста
    if use_stop_words:
        vectorizer = CountVectorizer(stop_words=stopwords_1, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Обучение классификатора
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Оценка точности классификации
    accuracy = accuracy_score(y_test, classifier.predict(X_test_vectorized))

    return accuracy


def run_experiments():
    i = 0
    for stop_words in using_stop_words:
        for ngram_range in total_ngram_range:
            for test_size in total_test_size:
                i = i + 1
                textsWithCategory = reviews

                random.shuffle(textsWithCategory)

                texts = [text for text, category in textsWithCategory]
                labels = [category for text, category in textsWithCategory]

                # Запуск эксперимента
                accuracy = run_experiment(texts, labels, stop_words, ngram_range, test_size)

                stwrds = ""
                if stop_words:
                    stwrds = "Использовались стоп-слова; "
                else:
                    stwrds = "Не использовались стоп-слова; "

                ngrm = ""
                if ngram_range == (1,1):
                    ngrm = "Униграмма; "
                else:
                    ngrm = "Биграмма; "

                ts = "Размер тестовой выборки " + str(test_size * 100) + "%"

                # Сохранение результатов
                results.append("Итерация: " + str(i) + "; Точность: " + str(accuracy) + "; " + stwrds + ngrm + ts)


# Параметры экспериментов
using_stop_words = [True, False]
total_ngram_range = [(1, 1), (1, 2)]


total_test_size = [0.1, 0.15, 0.2, 0.25, 0.3]

# Запуск экспериментов
run_experiments()

for i in range(0, 20):
    print(results[i])