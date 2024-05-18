from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.draw.dispersion import dispersion_plot
import matplotlib.pyplot as plt
import codecs

stemmer = SnowballStemmer("russian")

targets_words = ['хороший', 'отличный', 'супер', 'шикарный', 'плохой', 'ужасный',  'горячий', 'занижен']
targets_stemmer_words = [stemmer.stem(word) for word in targets_words]

# read positive
fileObj = codecs.open( "pos.txt", "r", "utf_8_sig" )
positive = fileObj.read()
fileObj.close()

# read negative
fileObj = codecs.open( "neg.txt", "r", "utf_8_sig" )
negative = fileObj.read()
fileObj.close()

def get_words(text, is_stop):
    tokens = word_tokenize(text, 'russian')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    stop_words = set(stopwords.words('russian'))
    if(is_stop):
        return [word for word in stemmed_words if word not in stop_words]
    else:
        return stemmed_words

#dispersion
def plot_dispersion(reviews, is_stop, title):
    words = get_words(reviews, is_stop)
    ax = dispersion_plot(words, targets_stemmer_words, False, title)
    ax.set_yticks(list(range(len(targets_words))), reversed(targets_words))
    plt.show()

#lexical
def lexical_diversity(reviews, stop):
    words = get_words(reviews, stop)
    return len(set(words)) / len(words)

#cumulative
def cumulative_frequency(reviews, stop, title):
    words = get_words(reviews, stop)
    freq_dist = FreqDist(words)
    freq_dist.plot(50, cumulative=True, title=title)
    plt.show()

#selection
def selection(reviews, stop):
    length = 7
    min_freq = 5
    words = get_words(reviews, stop)
    freq_dist = FreqDist(words)
    selected_words = [word for word, freq in freq_dist.items() if len(word) >= length and freq >= min_freq]
    return selected_words

plot_dispersion(positive, False, "Позитив без стоп слов")
plot_dispersion(negative, False, "Негатив без стоп-слов")

plot_dispersion(positive, True, "Позитив со стоп-словами")
plot_dispersion(negative, True, "Негатив со стоп-словами")

print("Лекс. разн. поз. отзывов без стоп-слов:", lexical_diversity(positive, False))
print("Лекс. разн. негатив. отзывов без стоп-слов:", lexical_diversity(negative, False))

print("Лекс. разн. поз. отзывов со стоп-словами:", lexical_diversity(positive, True))
print("Лекс. разн. негатив. отзывов со стоп-словами:", lexical_diversity(negative, True))


cumulative_frequency(positive, False, "Кум. график поз. отзывов без стоп-слов")
cumulative_frequency(negative, False, "Кум. график негатив. отзывов без стоп-слов")

cumulative_frequency(positive, True, "Кум. график поз. отзывов со стоп-словами")
cumulative_frequency(negative, True, "Кум. график негатив. отзывов со стоп-словами")

print("Отбор слов для поз. отзывов со стоп-словами:", selection(positive, True))
print("Отбор слов для негатив. отзывов со стоп-словами:", selection(negative, True))
print("Отбор слов для поз. отзывов без стоп-слов:", selection(positive, False))
print("Отбор слов для негатив отзывов без стоп-слов:", selection(negative, False))