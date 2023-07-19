from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from joblib import Memory
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
def clean(text):
  text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  texter = re.sub(r"<br />", " ", text)
  texter = re.sub(r"&quot;", "\"",texter)
  texter = re.sub('&#39;', "\"", texter)
  texter = re.sub('\n', " ", texter)
  texter = re.sub(' u '," you ", texter)
  texter = re.sub('`',"", texter)
  texter = re.sub(' +', ' ', texter)
  texter = re.sub(r"(!)\1+", r"!", texter)
  texter = re.sub(r"(\?)\1+", r"?", texter)
  texter = re.sub('&amp;', 'and', texter)
  texter = re.sub('\r', ' ',texter)
  clean = re.compile('<.*?>')
  texter = texter.encode('ascii', 'ignore').decode('ascii')
  texter = re.sub(clean, '', texter)
  if texter == "":
    texter = ""
  return texter
def penn2morphy(penntag): # Mapping of position tag is required to use with WordNetLemmatizer
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'
def lemmatize_sent(word_list):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(word_list)]

def contain_digit(word):
  return any(ch.isdigit() for ch in word)

def compile_vocabulary_lemma(document):
    sentence_list = sent_tokenize(
        document)  # To get sentences from document, returns them as a list of sentences, step a
    vocabulary = []
    stop_words = text.ENGLISH_STOP_WORDS
    tokenizer = CountVectorizer().build_analyzer()

    for sentence in sentence_list:
        word_list = tokenizer(sentence)  # To get words from sentence, step b

        processed_word_list = lemmatize_sent(word_list)  # position tag then lemmatize words in a sentence step c and d

        new_word_list = []
        for word in processed_word_list:
            if ((word not in stop_words) and (
            not (contain_digit(word)))):  # discard stop words and words that contain a number step e
                new_word_list.append(word)

        vocabulary = vocabulary + new_word_list
    return list(set(vocabulary))  # Return vocabulary that consists the word once,step f

def stem_sent(word_list):
    porter = PorterStemmer()
    return [porter.stem(word.lower())
            for word in (word_list)]


def compile_vocabulary_stem(document):
  sentence_list = sent_tokenize(document) #To get sentences from document, returns them as a list of sentences, step a
  vocabulary = []
  stop_words = text.ENGLISH_STOP_WORDS
  tokenizer = CountVectorizer().build_analyzer()
  for sentence in sentence_list:
    word_list = tokenizer(sentence) # To get words from sentence, step b

    processed_word_list = stem_sent(word_list)  # stem words in a sentence step d

    new_word_list = []
    for word in processed_word_list:
      if ((word not in stop_words) and (not (contain_digit(word)))): # discard stop words and words that contain a number step e
        new_word_list.append(word)

    vocabulary = vocabulary + new_word_list
  return list(set(vocabulary)) #Return vocabulary that consists the word once,step f

class clean_pipeline(TransformerMixin, BaseEstimator):
    def __init__(self):
        return None
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        for i in range(0,len(X)):
            X_copy.iloc[i] = clean(X_copy.iloc[i])
        return X_copy

df=pd.read_csv("E:\Downloads\Project1-Classification.csv")
train, test = train_test_split(df[["full_text","root_label"]], test_size=0.2)
best_gamma = 1000
best_l = 100000
best_l2 = 100000
pipeline = Pipeline([
    ('clean_pip', clean_pipeline()),
    ('count_vectorizer',CountVectorizer(min_df=3,
                             analyzer=compile_vocabulary_lemma)),
    ('tfidf',TfidfTransformer()),
    ('dim_red',TruncatedSVD(n_components=5,random_state = 42)),
    ('model',SVC(kernel = "linear",C = best_gamma))
],memory=Memory(location="lsdm_project1", verbose=10))

param_search = [
     {
         'clean_pip': [clean_pipeline()],
          'count_vectorizer__min_df': [3,5],
          'count_vectorizer__analyzer': [compile_vocabulary_lemma,compile_vocabulary_stem],
          'dim_red': [TruncatedSVD(n_components=5,random_state = 42),NMF(n_components=5, init='random', random_state=42,tol=1e-3,max_iter=100)],
          'dim_red__n_components': [5,30,80],
          "model":[ SVC(kernel = "linear",C = best_gamma),
                    LogisticRegression(penalty='l1', C=best_l, solver = "liblinear",max_iter=450),
                    LogisticRegression(penalty='l2', C=best_l2, solver = "liblinear",max_iter=450),
                     GaussianNB()]
     }
]


grid_search = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_search, scoring='accuracy',verbose=10)
grid_search.fit(train["full_text"], train["root_label"])
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv("lsdm_project_gridsearch.csv", encoding='utf-8', index=False)