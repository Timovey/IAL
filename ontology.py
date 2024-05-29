from nltk import word_tokenize
from owlready2 import *
import types
import codecs
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian")

fileObj = codecs.open( "pos.txt", "r", "utf_8_sig" )
positive = fileObj.read()
fileObj.close()
positive = [value for value in positive.split('\r\n') if value]

# read negative
fileObj = codecs.open( "neg.txt", "r", "utf_8_sig" )
negative = fileObj.read()
fileObj.close()
negative = [value for value in negative.split('\r\n') if value]


onto = get_ontology("C:/Users/Timovey/Desktop/ontology.owl")

stopWordsRu = set(stopwords.words('russian'))
my_stop_words = ['.', ',', '\'', '(', ')', '-', '«', '»', '?', '!', ':', ';', '—']
stopWords = sorted(list(stopWordsRu) + my_stop_words)

with onto:
    class Review(Thing):
        pass

    class Theme(Thing):
        pass

    class Word(Thing):
        pass

    class has_tone(ObjectProperty):
        domain = [Review]
        range = [Theme]


    class has_review(ObjectProperty):
        domain = [Theme]
        range = [Review]
        inverse_property = has_tone


    class has_word(ObjectProperty):
        domain = [Review]
        range = [Word]

    class includes_in_review(ObjectProperty):
        domain = [Word]
        range = [Review]
        inverse_property = has_word

    def create_review(theme, reviews):
        for i in range(len(reviews)):
            NewReview = Review(theme.name +'_review_' + str(i))
            NewReview.has_tone.append(theme)

            tokens = word_tokenize(reviews[i], 'russian')
            stemmed_words = [stemmer.stem(word) for word in tokens]
            words = [word for word in stemmed_words if word not in stopWords and len(word) > 2]

            for j in range(len(words)):
                NewWord = Word(str(words[j]))
                NewReview.has_word.append(NewWord)

    PositiveTheme = Theme('positive')
    NegativeTheme = Theme('negative')
    create_review(PositiveTheme, positive)
    create_review(NegativeTheme, negative)


onto.save(file="ontology1.owl", format="rdfxml")
