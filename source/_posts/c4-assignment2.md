---
title: c4-assignment2
date: 2018-11-08 15:42:32
tags: [coursera, data_science_4]
toc: true
reward: true
---

Introduction to NLTK
In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

<!--more-->

## Part 1 - Analyzing Moby Dick


```python
import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)
```

### Example 1

How many tokens (words and punctuation symbols) are in text1?

*This function should return an integer.*


```python
def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()
```


    254989

### Example 2

How many unique tokens (unique words and punctuation) does text1 have?

*This function should return an integer.*


```python
def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()
```


    20755

### Example 3

After lemmatizing the verbs, how many unique tokens does text1 have?

*This function should return an integer.*


```python
from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()
```


    16900

### Question 1

What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)

*This function should return a float.*


```python
def answer_one():
    
    
    return example_two() / example_one()

answer_one()
```


    0.08139566804842562

### Question 2

What percentage of tokens is 'whale'or 'Whale'?

*This function should return a float.*


```python
from nltk.book import FreqDist
def answer_two():
    
#     moby_tokens = nltk.word_tokenize(moby_raw.lower())
#     temp = nltk.Text(moby_tokens)
    dist = FreqDist(moby_tokens)
    return (dist['whale'] + dist['Whale']) / len(moby_tokens) * 100

answer_two()
```


    0.4125668166077752

### Question 3

What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

*This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*


```python
def answer_three():
    
    dist = FreqDist(text1)
    sort_dist = sorted(dist.items(), key=lambda d: d[1], reverse=True)
    return sort_dist[:20]

answer_three()
```


    [(',', 19204),
     ('the', 13715),
     ('.', 7308),
     ('of', 6513),
     ('and', 6010),
     ('a', 4545),
     ('to', 4515),
     (';', 4173),
     ('in', 3908),
     ('that', 2978),
     ('his', 2459),
     ('it', 2196),
     ('I', 2097),
     ('!', 1767),
     ('is', 1722),
     ('--', 1713),
     ('with', 1659),
     ('he', 1658),
     ('was', 1639),
     ('as', 1620)]

### Question 4

What tokens have a length of greater than 5 and frequency of more than 150?

*This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*


```python
def answer_four():
    
    dist = FreqDist(text1)
    result = [w for w, d in dist.items() if len(w) > 5 and d > 150]
    return sorted(result)

answer_four()
```


    ['Captain',
     'Pequod',
     'Queequeg',
     'Starbuck',
     'almost',
     'before',
     'himself',
     'little',
     'seemed',
     'should',
     'though',
     'through',
     'whales',
     'without']

### Question 5

Find the longest word in text1 and that word's length.

*This function should return a tuple `(longest_word, length)`.*


```python
def answer_five():
    
    sort_by_length = sorted(text1, key=len, reverse=True)
    return (sort_by_length[0], len(sort_by_length[0]))

answer_five()
```


    ("twelve-o'clock-at-night", 23)

### Question 6

What unique words have a frequency of more than 2000? What is their frequency?

"Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."

*This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*


```python
def answer_six():
    
    dist = FreqDist(text1)
    sort_dist = sorted(dist.items(), key=lambda d: d[1], reverse=True)
    result = [(d, w) for w, d in sort_dist if w.isalpha() and d > 2000]
    return result

answer_six()
```


    [(13715, 'the'),
     (6513, 'of'),
     (6010, 'and'),
     (4545, 'a'),
     (4515, 'to'),
     (3908, 'in'),
     (2978, 'that'),
     (2459, 'his'),
     (2196, 'it'),
     (2097, 'I')]

### Question 7

What is the average number of tokens per sentence?

*This function should return a float.*


```python
def answer_seven():
    
    sentences = nltk.sent_tokenize(moby_raw)
    sentence_length = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    return sum(sentence_length) / len(sentence_length)

answer_seven()
```


    25.881952902963864

### Question 8

What are the 5 most frequent parts of speech in this text? What is their frequency?

*This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*


```python
import pandas as pd
def answer_eight():
    
#     moby_words = [w for w in moby_tokens if w.isalpha()]
    pos = [pos for w, pos in nltk.pos_tag(moby_tokens)]
    dist = FreqDist(pos)
    results = sorted(dist.items(), key=lambda d: d[1], reverse=True)[:5]
#     没办法，答案不接受int64，所以只好进行没必要的转换
    df = pd.DataFrame(results)
    df[1] = df[1].astype('int32')
    
    return list(zip(df[0], df[1]))

answer_eight()
```


    [('NN', 32730), ('IN', 28657), ('DT', 25867), (',', 19204), ('JJ', 17620)]

## Part 2 - Spelling Recommender

For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

*Each of the three different recommenders will use a different distance measure (outlined below).

Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.


```python
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance, edit_distance
from nltk.util import ngrams
import pandas as pd

correct_spellings = words.words()
spellings_series = pd.Series(correct_spellings)
```

### Question 9

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    result = []
#     for entry in entries:
#         dis = 1
#         recommendation = None
#         for correct_spelling in correct_spellings:
#             if entry[0] == correct_spelling[0]:
#                 jd = jaccard_distance(set(ngrams(entry, 3)), set(ngrams(correct_spelling, 3)))
# #                 print(jd)
#                 if jd < dis:
#                     dis = jd
#                     print('%s and %s distance: %d' % (entry, correct_spelling, dis))
#                     recommendation = correct_spelling
#         result.append(recommendation)
    for entry in entries:
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = [(jaccard_distance(set(ngrams(entry, 3)), set(ngrams(word, 3))), word) for word in spellings]
        closest = min(distances)
        result.append(closest[1])
    return result
    
answer_nine()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel/__main__.py:18: DeprecationWarning: generator 'ngrams' raised StopIteration
    
    ['corpulent', 'indecence', 'validate']

### Questin 10

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    result = []
    for entry in entries:
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = [(jaccard_distance(set(ngrams(entry, 4)), set(ngrams(word, 4))), word) for word in spellings]
        closest = min(distances)
        result.append(closest[1])
    return result
answer_ten()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel/__main__.py:5: DeprecationWarning: generator 'ngrams' raised StopIteration
    
    ['cormus', 'incendiary', 'valid']

### Question 11

For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

**[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**

*This function should return a list of length three:
`['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*


```python
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    result = []
    for entry in entries:
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = [(edit_distance(entry, word, transpositions=True), word) for word in spellings]
        closest = min(distances)
        result.append(closest[1])
    return result
    
answer_eleven()
```


    ['corpulent', 'intendence', 'validate']