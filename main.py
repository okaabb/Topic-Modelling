import nltk
import string
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

SAMPLE_SIZE = 200
max_K = 50
optimal_K = 10

stop_words = set(stopwords.words('english'))
punctuation = string.punctuation
df = pd.read_csv('articles1.csv')
text = df.sample(SAMPLE_SIZE).reset_index()['content']

lemmatizer = WordNetLemmatizer()
corpus = []

for i in text:
  raw = str(i).lower()
  tokens = word_tokenize(raw)
  stopped_tokens = [raw for raw in tokens if not raw in stop_words and raw not in string.punctuation and len(raw)>=2]
  lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens]
  new_lemma_tokens = [raw for raw in lemma_tokens]
  corpus.append(' '.join(new_lemma_tokens))

print("Preprocessing finished")

cv = TfidfVectorizer()
corupsFit = cv.fit_transform(corpus)
tfidfmatrix = pd.DataFrame(corupsFit.toarray(),columns=cv.get_feature_names())


step = 1
iterations = range(2, max_K + 1, step)

Squared_Sum_Error = []
for k in iterations:
  Squared_Sum_Error.append(KMeans(n_clusters=k).fit(tfidfmatrix).inertia_)

print("Done Clustering for optimal K")

f, ax = plt.subplots(1, 1)
ax.plot(iterations, Squared_Sum_Error, marker='o')
ax.set_xlabel('Clusters')
ax.set_xticks(iterations)
ax.set_xticklabels(iterations)
ax.set_ylabel('Squared-Sum-Error')

clusters = KMeans(n_clusters=k).fit_predict(corupsFit)
max_label = max(clusters)
max_items = np.random.choice(range(corupsFit.shape[0]), size=SAMPLE_SIZE, replace=False)

pca = PCA(n_components=2).fit_transform(corupsFit[max_items, :].todense())

idx = np.random.choice(range(pca.shape[0]), size=SAMPLE_SIZE, replace=False)
label_subset = clusters[max_items]
label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]
f, ax = plt.subplots(1, 1)
ax.scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
ax.set_title('PCA Cluster Plot')

print("Finished plot")

plt.show()