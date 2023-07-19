import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
np.random.seed(42)
random.seed(42)
# Use here if csv file is not uploaded at the files section
#uploaded = files.upload()
#df = pd.read_csv(io.BytesIO(uploaded['Project1-Classification.csv']))

def receiver_op_char(model, test_set, test_labeling, true):
    predictions = model.predict(test_set)
    acc = metrics.accuracy_score(test_labeling, predictions)
    prec = metrics.precision_score(test_labeling, predictions, pos_label=true)
    recall = metrics.recall_score(test_labeling, predictions, pos_label=true)
    f1 = metrics.f1_score(test_labeling, predictions, pos_label=true)

    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(prec))
    print('Recall: ' + str(recall))
    print('F-1 Score: ' + str(f1))

    # plt.figure()
    metrics.plot_confusion_matrix(model, test_set, test_labeling)

    # plt.figure()
    metrics.plot_roc_curve(model, test_set, test_labeling, pos_label=true)

    return acc, prec, recall,f1
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#net = Net()


df = pd.read_csv('E:\Downloads\Project1-Classification.csv')

full_text = df["full_text"]
num_alnum_full_text =[]
for i in range(0,full_text.shape[0]):
    sample = full_text.iloc[i]
    count = sum(ch.isalnum() for ch in sample )
    num_alnum_full_text.append(count)

num_class_leaf_label = df['leaf_label'].value_counts()
num_class_root_label = df['root_label'].value_counts()

from sklearn.model_selection import train_test_split
train, test = train_test_split(df[["full_text","root_label","leaf_label","keywords"]], test_size=0.2)

import re


def feature_engineer(df, embeddings, glove_dim):
    """Takes in a dataframe and embeddings and converts them to an array the same
    length x 300 which uses the GLoVE Embeddings"""

    features = np.empty((0, glove_dim), 'float32')
    for row in range(0, len(df)):
        keywords = ast.literal_eval(df.iloc[row]["keywords"])
        # keywords = lemmatize_words(keywords)
        feature = np.zeros(glove_dim)
        for keyword in keywords:
            if keyword in embeddings:
                feature += embeddings[keyword]
        features = np.vstack((features, feature))
    return features
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


# Clean each data sample(train and test)
for i in range(0, train.shape[0]):
    sample = train.iloc[i]["full_text"]
    train.iloc[i]["full_text"] = clean(sample)

for i in range(0, test.shape[0]):
    sample = test.iloc[i]["full_text"]
    test.iloc[i]["full_text"] = clean(sample)

embeddings_dict = {}
dimension_of_glove = 300

with open("E:\Downloads\glove.6B.300d.txt",'r', encoding = "utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

import ast
keywords_list = []
for i in range(0,df["keywords"].shape[0]):
    keywords_list.append(ast.literal_eval(df.iloc[i]["keywords"]))

trainFeature = feature_engineer(train, embeddings_dict,dimension_of_glove)
testFeature = feature_engineer(test, embeddings_dict,dimension_of_glove)
#%% Define the labels used for binary classification
label_train = train["root_label"]
label_test= test["root_label"]

classifier = LogisticRegression(penalty='l2', solver = "liblinear",max_iter=400)
classifier.fit(trainFeature,label_train)
accuracy,precision,recall,f1score = receiver_op_char(classifier,testFeature,label_test,"sports")

glove_accs = []
for glove_fname, dimension_of_glove in zip(["glove.6B.50d.txt","glove.6B.100d.txt","glove.6B.200d.txt","glove.6B.300d.txt"],[50,100,200,300]):
    embeddings_dict = {}
    with open("E:/E_Python/gd_rp_jl/glove/" + glove_fname, 'r',encoding="utf8") as f:
        for line in f:
           values = line.split()
           word = values[0]
           vector = np.asarray(values[1:], "float32")
           embeddings_dict[word] = vector
    trainFeature = feature_engineer(train, embeddings_dict,dimension_of_glove)
    testFeature = feature_engineer(test, embeddings_dict,dimension_of_glove)
    classifier = LogisticRegression(penalty='l2', solver = "liblinear",max_iter=500)
    classifier.fit(trainFeature,label_train)
    accGLoVE,precGLoVE,recallGLoVE,f1GLoVE = receiver_op_char(classifier,testFeature,label_test,"sports")
    glove_accs+=[accGLoVE]
#%%
plt.figure()
plt.plot([50,100,200,300], glove_accs,"b*--")
plt.xlabel("GLoVE length")
plt.ylabel("Testing accuracy (%)")
#%% 13 - UMAP Visualization
import umap.umap_ as umap
import seaborn as sns
# generate UMAP embeddings
reducer = umap.UMAP()
rand_uniform = StandardScaler().fit_transform(np.random.uniform(size=trainFeature.shape))
scaled_data = StandardScaler().fit_transform(trainFeature)
UMAP_first = reducer.fit_transform(scaled_data)
UMAP_second = reducer.fit_transform(rand_uniform)

#%% Plotting glove based embeddings
plt.figure(figsize=(10,5))
x=0
for label in label_train.unique():
    indices = [i for i in range(0,len(label_train)) if label_train.iloc[i]==label]
    plt.scatter(
    UMAP_first[indices, 0],
    UMAP_first[indices, 1],
    s=5,
    c=sns.color_palette()[x],
    label=label)
    x=1
plt.gca().set_aspect('equal', 'datalim')
plt.title('Glove', fontsize=12)
plt.legend()

#%% Plotting of random vectors
plt.figure(figsize=(10,5))
x=0
for label in label_train.unique():
    indices = [i for i in range(0,len(label_train)) if label_train.iloc[i]==label]
    plt.scatter(
    UMAP_second[indices, 0],
    UMAP_second[indices, 1],
    s=5,
    c=sns.color_palette()[x],label=label)
    x=1
plt.gca().set_aspect('equal', 'datalim')
plt.title('Random', fontsize=12)
plt.legend()
