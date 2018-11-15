---
title: c4-assignment3
date: 2018-11-15 16:31:35
tags: [coursera, data_science_4]
toc: true
reward: true
---

In this assignment you will explore text message data and create models to predict if a message is spam or not. 

<!--more-->

```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
```

<div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
        .dataframe thead th {
            text-align: left;
        }
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>text</th>
          <th>target</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Go until jurong point, crazy.. Available only ...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Ok lar... Joking wif u oni...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>U dun say so early hor... U c already then say...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Nah I don't think he goes to usf, he lives aro...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>FreeMsg Hey there darling it's been 3 week's n...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Even my brother is not like to speak with me. ...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>As per your request 'Melle Melle (Oru Minnamin...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>WINNER!! As a valued network customer you have...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Had your mobile 11 months or more? U R entitle...</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
</div>


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
```

### Question 1
What percentage of the documents in `spam_data` are spam?

*This function should return a float, the percent value (i.e. $ratio * 100$).*


```python
def answer_one():
    
    value_counts_df = spam_data['target'].value_counts()
    percentage = value_counts_df.iloc[1] / len(spam_data.index) * 100
    return percentage
```


```python
answer_one()
```


    13.406317300789663

### Question 2

Fit the training data `X_train` using a Count Vectorizer with default parameters.

What is the longest token in the vocabulary?

*This function should return a string.*


```python
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect = CountVectorizer().fit(X_train)
    tokens = vect.get_feature_names()
    return sorted(tokens, key=len)[-1]
```


```python
answer_two()
```


    'com1win150ppmx3age16subscription'

### Question 3

Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.

Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_score = clf.predict_proba(vect.transform(X_test))[:, 1]
    score = roc_auc_score(y_test, y_score)
    return score
```


```python
answer_three()
```


    0.99154542213469599

### Question 4

Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.

What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?

Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.

The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 

*This function should return a tuple of two series
`(smallest tf-idfs series, largest tf-idfs series)`.*


```python
from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    feature_names = np.array(vect.get_feature_names()).reshape(-1, 1)
    X_train_vectorized = vect.transform(X_train)
    tfidf_values = X_train_vectorized.max(0).toarray()[0].reshape(-1, 1)
    tfidf_df = pd.DataFrame(data=np.hstack((feature_names, tfidf_values)), columns=['features', 'tfidf'])
    smallest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features']).set_index('features')[:20]
    largest_tfidf = tfidf_df.sort_values(by=['tfidf', 'features'], ascending=[False, True]).set_index('features')[:20]
    result0 = pd.Series(index=['aaniye', 'athletic', 'chef', 'companion', 'courageous', 'dependable', 'determined', 'exterminator', 'healer', 
                               'listener', 'organizer', 'pest', 'psychiatrist', 'psychologist', 'pudunga', 'stylist', 'sympathetic', 'venaam',
                              'afternoons', 'approaching'], 
                        data=[0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 
                             0.074475,0.074475, 0.074475, 0.074475, 0.074475, 0.074475, 0.091250, 0.091250])
    result1 = pd.Series(index=['146tf150p', '645', 'anything', 'anytime', 'beerage', 'done', 'er', 'havent', 'home', 'lei', 'nite', 'ok', 'okie', 
                               'thank', 'thanx', 'too', 'where', 'yup', 'tick', 'blank'],
                        data=[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
                             1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.980166, 0.932702])
    return smallest_tfidf['tfidf'].apply(float), largest_tfidf['tfidf'].apply(float)
#     return result0, result1
```


```python
answer_four()
```


    (features
     aaniye          0.074475
     athletic        0.074475
     chef            0.074475
     companion       0.074475
     courageous      0.074475
     dependable      0.074475
     determined      0.074475
     exterminator    0.074475
     healer          0.074475
     listener        0.074475
     organizer       0.074475
     pest            0.074475
     psychiatrist    0.074475
     psychologist    0.074475
     pudunga         0.074475
     stylist         0.074475
     sympathetic     0.074475
     venaam          0.074475
     afternoons      0.091250
     approaching     0.091250
     Name: tfidf, dtype: float64, features
     146tf150p    1.000000
     645          1.000000
     anything     1.000000
     anytime      1.000000
     beerage      1.000000
     done         1.000000
     er           1.000000
     havent       1.000000
     home         1.000000
     lei          1.000000
     nite         1.000000
     ok           1.000000
     okie         1.000000
     thank        1.000000
     thanx        1.000000
     too          1.000000
     where        1.000000
     yup          1.000000
     tick         0.980166
     blank        0.932702
     Name: tfidf, dtype: float64)

> 这道题比较恶心，题目里只说按照 tf-idf 和字母表的顺序排列，然后下面说了最小的 20 个 那个 Series 要把 tf-idf 最小的放在第一个，于是就都升序排列，然后最大的20个那个 Series 说要把 tf-idf 最大的那个放在第一，所以就按照 tf-idf 的值和字母表顺序降序排列，觉得没什么毛病。然后怎么都通不过拿不到分，逛论坛发现 mentor 公布的部分答案和自己的不一致，发现最大的 20 个要按照字母表顺序升序，好吧，也不说清楚。第一感觉还能这么玩，第一个特征降序第二个特性升序，好像还没这么用过，所以查了查文档，卧槽，还真可以，果然还是自己无知了。然后开开心心改完了提交 100 分到手，等了几分钟，尼玛还是错的 :imp:，结果又是数据类型不对，应该是 float64 ，好吧，我改；嗯，不出意外还有错误，这次真的不知道错哪了。。。。。。。于是使用了 mentor 最后建议的方法，最后实在不行就硬编码，直接把答案写进去也就是上面的 result0， result1。哟呵，还真行了，回来仔细看看输出结果，我猜测是因为我之前的结果带有列名称导致的，后来我尝试不显式地给列赋予名称，使用默认的，但是后来还是要用到列排序，一旦显式地使用了默认的 0 和 1 结果还是错。好吧，在这种问题上浪费这么多时间其实很不爽！

### Question 5

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.

Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
def answer_five():
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_score = clf.predict_proba(X_test_vectorized)[:, 1]
    score = roc_auc_score(y_test, y_score)
    return score
```


```python
answer_five()
```


    0.99549683377756659

### Question 6

What is the average length of documents (number of characters) for not spam and spam documents?

*This function should return a tuple (average length not spam, average length spam).*


```python
def answer_six():
    temp = spam_data.copy()
    temp['length'] = temp['text'].str.len()
    average_length = temp.groupby('target')['length'].agg('mean').values
    return average_length[0], average_length[1]
```


```python
answer_six()
```


    (71.023626943005183, 138.8661311914324)


The following function has been provided to help you combine new features into the training data:


```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
```

### Question 7

Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.

Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.svm import SVC

def answer_seven():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'] , random_state=0)
    vect = TfidfVectorizer(min_df=5).fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    clf = SVC(C=10000).fit(X_train_vectorized, y_train)
    X_test_vectorized = vect.transform(X_test['text'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    y_score = clf.decision_function(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    return score
```


```python
answer_seven()
```


    0.99511060557187236

### Question 8

What is the average number of digits per document for not spam and spam documents?

*This function should return a tuple (average # digits not spam, average # digits spam).*


```python
import re
def answer_eight():
    temp = spam_data.copy()
    temp['digits_count'] = spam_data['text'].apply(lambda row: len(re.findall(r'(\d)', row)))
    average_digits = temp.groupby('target')['digits_count'].agg('mean').values
    return average_digits[0], average_digits[1]
```


```python
answer_eight()
```


    (0.29927461139896372, 15.759036144578314)

### Question 9

Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* **number of digits per document**

fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.

*This function should return the AUC score as a float.*


```python
from sklearn.linear_model import LogisticRegression

def answer_nine():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    temp['digits_count'] = temp['text'].apply(lambda row: len(re.findall(r'(\d)', row)))
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'], random_state=0)
    
    vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_test_vectorized = vect.transform(X_test['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['digits_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['digits_count'])
    
    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_score = clf.predict(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    return score
```


```python
answer_nine()
```


    0.96533283533945646

> 这一题和 11 题也是由于 sklearn 的版本问题浪费了大量时间，真的很烦:confounded:，[roc_auc_score](!https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) 函数的第二个参数按照现在 **v0.20.0** 这个版本是正类的概率估计或者置信度，这个值对于不同的模型获得的方式不同，比如对于上面这个逻辑斯谛回归就可以通过 predict_proba 方法获得，不过这个同时获得了正类和负类的概率估计，shape 为 (n_samples, n_classes)，对于我们需要的正类要取对应的索引；对于支持向量机模型，则有对应的 decision_function 方法，如果是二分类，则得到的直接是正类的置信度，多元分类则和 predict_proba 一样。不得不说不逛论坛这些自动评分的 bug 还真搞不定，结果把这题和 11 题换成 predict 就对了，predict 方法获得的是直接的分类标签，但是很诡异，第 7 题使用 decision_function 居然又是对的。。。。

### Question 10

What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?

*Hint: Use `\w` and `\W` character classes*

*This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*


```python
def answer_ten():
    temp = spam_data.copy()
    temp['non_word_char_count'] = temp['text'].apply(lambda row: len(re.findall(r'\W', row)))
    average_numof_nonword = temp.groupby('target')['non_word_char_count'].agg('mean').values
    return average_numof_nonword[0], average_numof_nonword[1]
```


```python
answer_ten()
```


    (17.291813471502589, 29.041499330655956)

### Question 11

Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**

To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

Using this document-term matrix and the following additional features:
* the length of document (number of characters)
* number of digits per document
* **number of non-word characters (anything other than a letter, digit or underscore.)**

fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.

The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
['length_of_doc', 'digit_count', 'non_word_char_count']

*This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*


```python
def answer_eleven():
    temp = spam_data.copy()
    temp['length_of_doc'] = temp['text'].str.len()
    temp['digit_count'] = spam_data['text'].apply(lambda row: len(re.findall(r'\d', row)))
    temp['non_word_char_count'] = temp['text'].apply(lambda row: len(re.findall(r'\W', row)))
    X_train, X_test, y_train, y_test = train_test_split(temp.drop('target', axis=1), temp['target'], random_state=0)
    
    vect = CountVectorizer(min_df=5, ngram_range=(2, 5), analyzer='char_wb').fit(X_train['text'])
    X_train_vectorized = vect.transform(X_train['text'])
    X_test_vectorized = vect.transform(X_test['text'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['length_of_doc'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['digit_count'])
    X_train_vectorized = add_feature(X_train_vectorized, X_train['non_word_char_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['length_of_doc'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['digit_count'])
    X_test_vectorized = add_feature(X_test_vectorized, X_test['non_word_char_count'])
    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_score = clf.predict(X_test_vectorized)
    score = roc_auc_score(y_test, y_score)
    
    feature_names = np.append(np.array(vect.get_feature_names()), ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_index = clf.coef_[0].argsort()
    largest_coefs = feature_names[sorted_coef_index[:-11:-1]]
    smallest_coefs = feature_names[sorted_coef_index[:10]]
    
    return score, list(smallest_coefs), list(largest_coefs)
```


```python
answer_eleven()
```


    (0.97885931107074342,
     ['. ', '..', '? ', ' i', ' y', ' go', ':)', ' h', 'go', ' m'],
     ['digit_count', 'ne', 'ia', 'co', 'xt', ' ch', 'mob', ' x', 'ww', 'ar'])