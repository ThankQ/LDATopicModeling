import gensim
from gensim import corpora
import nltk
#nltk.download("stopwords")
#nltk.download("wordnet")
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import wikipedia

doc_complete = []
import requests
import json
for i in range(28):
    f = open("frankenstein/ch" + str(i+1) + ".txt", "r")
    doc_complete.append(f.read())
    print(i)


"""
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."
doc_complete = [doc1, doc2, doc3, doc4, doc5]
"""

#LDA code taken from https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
#clean documents
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
stop.add('-')
stop.add('also')
stop.add('yet')
stop.add('said')
stop.add('could')
stop.add('would')
stop.add('shall')
stop.add('still')
stop.add('up')
stop.add('upon')
stop.add('it')
f = open('stoplist_final.txt')
for i in f:
    stop.add(i.strip())
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=50)


for i in ldamodel.print_topics(num_topics=6, num_words=10):
    print(i)
#print(ldamodel.print_topics(num_topics=10, num_words=10))

