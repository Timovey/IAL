from gensim import models
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
import pymorphy2
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DOC_PATTERN = r'.*\.txt'
CAT_PATTERN = r'(\w+)/.*'
my_stop_words = ['все', 'это', 'весь', 'игра', 'упаковка', 'всё', 'процессор', 'завестись']

sentTokenizer = PunktSentenceTokenizer()
tokenizer = RegexpTokenizer(r'\w+')
morph = pymorphy2.MorphAnalyzer()
stopWordsRu = set(stopwords.words('russian'))
stopWords = sorted(list(stopWordsRu) + my_stop_words)
# r'.*(pos|neg)\.txt'

reader = CategorizedPlaintextCorpusReader('C:/Users/Timovey/Study/IAL/docs/', DOC_PATTERN, cat_pattern=CAT_PATTERN)

textCollection = []
for file_id in reader.fileids():
    textCollection.append(reader.raw(fileids=file_id))

textCollTokens = []
for text in textCollection:
    sentList = [sent for sent in sentTokenizer.tokenize(text)]
    tokens = [word for sent in sentList for word in tokenizer.tokenize(sent.lower())]
    lemmedTokens = []
    for token in tokens:
        lemmedTokens.append(morph.parse(token)[0].normal_form)
    goodTokens = [token for token in lemmedTokens if token not in stopWords]
    textCollTokens.append(goodTokens)

textCollDictionary = corpora.Dictionary(textCollTokens)
textCollDictionary.filter_extremes(no_below=1)
textCollDictionary.save_as_text('textCollDictionary2.txt')



textCorpus = [textCollDictionary.doc2bow(doc) for doc in textCollTokens]
nTopics = 2
model = models.ldamodel.LdaModel(corpus=textCorpus, num_topics=
nTopics, id2word=textCollDictionary)

textTopicsMtx = np.zeros(shape=(len(textCorpus), nTopics), dtype=float)
for k in range(len(textCorpus)):
    for tpcId, tpcProb in model.get_document_topics(textCorpus[k]):
        textTopicsMtx[k, tpcId] = tpcProb


cloud = WordCloud(background_color='white', width=2500,
height=1800, max_words=5, colormap='tab10', prefer_horizontal=1.0)
topics = model.show_topics(num_topics=nTopics, num_words=5,formatted=False)
fig, ax = plt.subplots(2, 1, figsize=(2, 3), sharex=True, sharey=True)

for i, ax in enumerate(ax.flatten()):
    fig.add_subplot(ax)
    topicWords = dict(topics[i][1])
    cloud.generate_from_frequencies(topicWords, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Тема ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.axis('off')
plt.savefig('myfig.png')