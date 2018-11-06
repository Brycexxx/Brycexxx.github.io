---
title: c4-assignment1
date: 2018-11-06 18:27:25
tags: [coursera, data_science_4]
toc: true
reward: true
---

In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 

Each line of the [dates.txt](https://github.com/Brycexxx/data/blob/master/dates.txt) file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

<!--more-->

The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 

Here is a list of some of the variants you might encounter in this dataset:
* 04/20/2009; 04/20/09; 4/20/09; 4/3/09
* Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
* 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
* Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
* Feb 2009; Sep 2009; Oct 2010
* 6/2008; 12/2009
* 2009; 2010

Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
* Assume all dates in xx/xx/xx format are mm/dd/yy
* Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
* If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
* If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
* Watch out for potential typos as this is a raw, real-life derived dataset.

With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

For example if the original series was this:

    0    1999
    1    2010
    2    1978
    3    2015
    4    1985

Your function should return this:

    0    2
    1    4
    2    0
    3    1
    4    3

Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.

*This function should return a Series of length 500 and dtype int.*


```python
import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)
```




    0         03/25/93 Total time of visit (in minutes):\n
    1                       6/18/85 Primary Care Doctor:\n
    2    sshe plans to move as of 7/8/71 In-Home Servic...
    3                7 on 9/27/75 Audit C Score Current:\n
    4    2/6/96 sleep studyPain Treatment Pain Level (N...
    5                    .Per 7/06/79 Movement D/O note:\n
    6    4, 5/18/78 Patient's thoughts about current su...
    7    10/24/89 CPT Code: 90801 - Psychiatric Diagnos...
    8                         3/7/86 SOS-10 Total Score:\n
    9             (4/10/71)Score-1Audit C Score Current:\n
    dtype: object




```python
# 贪心，希望可以一次性解决，结果提取了比较多没用的数字出来又很难剔除，只好作罢
import re
pattern = r'(\d{1,2}|(Jan|Feb|Mar|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)?[-/\s\.,]*(\d{1,2}[a-z]{,2}|(Jan|Feb|Mar|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)?[-\s/\.,]*(\d{2,4})'
ptn = re.compile(pattern)
```


```python
# 初步分析，按日期模式分段
first_interval = df.iloc[0: 125] # mm/dd/yy
second_interval = df.iloc[125: 194] # 20 Mar 2009
third_interval = df.iloc[194: 228] # March 20, 2009   September. 15, 2011
forth_interval = df.iloc[228: 343] # Feb 2009    Feb, 2009
fifth_interval = df.iloc[343: 455] # 12/2009
sixth_interval = df.iloc[455:]  #2010
```


```python
def date_sorter():
    
    word_map_digits = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    def add_19(df):
        two_digits_year = df['year'].str.len() <= 2
        df['year'][two_digits_year] = df['year'][two_digits_year].apply(lambda x: '19' + x)
        return df
    
    def word_to_digits(row):
        mon = row['month'][:3]
        row['month'] = word_map_digits[mon]
        return row
    
    def str_to_int(row):
        row = [int(s) for s in row]
        return row
    
    first_interval = df.iloc[0: 125] # mm/dd/yy
    second_interval = pd.concat([df.iloc[125: 194], df.iloc[228: 343]]) # 20 Mar 2009  Feb 2009    Feb, 2009
    third_interval = df.iloc[194: 228] # March 20, 2009   September. 15, 2011
    forth_interval = df.iloc[343: 455] # 12/2009
    fifth_interval = df.iloc[455:]  #2010
    
    first_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
    second_pattern = r'(\d{1,2})?[\s\.,]{,2}((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s\.,]{,2}(\d{2,4})'
    third_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)[\s\.,]{,2}(\d{1,2}[a-z]{,2})[\s\.,]{,2}(\d{2,4})'
    forth_pattern = r'(\d{1,2})/(\d{2,4})'
    fifth_pattern = r'(\d{4})'
    
    first = first_interval.str.extractall(first_pattern)
    first = first.drop(1, level=1).rename(columns={0: 'month', 1: 'day', 2: 'year'})
    first.index = first.index.droplevel(1)
    
    second = second_interval.str.extractall(second_pattern)
    second = second.drop(1, level=1).rename(columns={0: 'day', 1: 'month', 2: 'year'})
    second.index = second.index.droplevel(1)
    second = second.apply(word_to_digits, axis=1)
    
    third = third_interval.str.extractall(third_pattern)
    third = third.drop(1, level=1).rename(columns={0: 'month', 1: 'day', 2: 'year'})
    third.index = third.index.droplevel(1)
    third = third.apply(word_to_digits, axis=1)
    
    forth = forth_interval.str.extractall(forth_pattern)
    forth = forth.drop(1, level=1).rename(columns={0: 'month', 1: 'year'})
    forth.index = forth.index.droplevel(1)
    forth['day'] = 1
    
    fifth = df.iloc[455:].str.extractall(fifth_pattern)
    fifth = fifth.drop(1, level=1).rename(columns={0: 'year'})
    fifth.index = fifth.index.droplevel(1)
    fifth['day'] = 1
    fifth['month'] = 1
    
    treated_df = pd.concat([first, second.loc[125: 194], third, second.loc[228: 343], forth, fifth])
    treated_df = add_19(treated_df)
    treated_df.fillna(value=1, inplace=True)
    treated_df = treated_df.apply(str_to_int, axis=1)
    treated_df.sort_values(by=['year', 'month', 'day'], inplace=True)
    treated_df.reset_index(inplace=True)
    
    return treated_df['index']
```

**1个不常用的 pandas 用法：**

上面利用 droplevel 删除多层索引的第二层，第二层的出现根据题目数据具体提取的结果来看还是有一行提取出了两个日期数据，那首先要找到是哪一行出了问题，确定这两个日期是否有一个是无效的。举个例子：

```python
import numpy as np
df_test = pd.DataFrame(np.arange(12).reshape(4, 3), index=[[0, 0, 1, 1], [0, 1, 0, 1]], columns=['a', 'b', 'c'])
```
得到：
<img src="https://raw.githubusercontent.com/Brycexxx/BlogComments/master/20181106184819.jpg"/>
所以要定位到上面这个 DataFrame 里面二级索引等于 1 的数据，中间尝试过很多次 iloc 和 loc 来定位，但是多层索引还是没搞明白怎么用，google 百度也没能弄明白，最后在 stackoverflow 上找到利用 index.get_level_values 来定位，效果如下：
<img src="https://raw.githubusercontent.com/Brycexxx/BlogComments/master/20181106190741.jpg"/>
代码：
```python
df_test.index.names = ['m', 'n'] # 给多级索引添加名字
df_test[df_test.index.get_level_values('n') == 1]
```
后来经过尝试，不加名字，直接传入 1 也是可以的。

事实上，上面这个方法依然没有达到我想要的效果，只是恰好可以满足这个题目，如果要定位到 m=1，n=1 这一行怎么办，经过漫长的查找依然无果，所以就先放着吧。。。。。

通过以上方式定位发现 72 行果然提取出了一个无效数据，于是要删除掉，很自然的想到 drop 方法，于是找来文档，美滋滋，正好有关于多级索引删除数据的例子：

<img src="https://raw.githubusercontent.com/Brycexxx/BlogComments/master/20181106191714.jpg"/>

看到上面例子里索引都是字符串，心中不免出现一丝担忧，我的可都是 int 型的，果不其然报错了。。。如下：

```python
df_test.drop(index=1, level=1)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-26-b161031ed499> in <module>()
----> 1 df_test.drop(index=1, level=1)

TypeError: drop() got an unexpected keyword argument 'index'
```

不过这个错误信息有点诡异啊，居然是没有这个参数，难以置信，好吧，用官方例子试试：

```python
df.drop(index='length', level=1)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-25-ce3d559c5e21> in <module>()
----> 1 df.drop(index='length', level=1)

TypeError: drop() got an unexpected keyword argument 'index'
```

emmm.....文档骗人啊！！！好吧，估计是版本的问题。。。。

心累，放弃这个方法了

重新开始各种 google 百度，不知道过了多长时间，对 drop 又有了新发现，drop('xxx', level=1) ，然后赶紧试了试，结果奏效了，于是成功删除二级索引等于 1 的那条数据，这个也算是恰巧是这个问题了，如果其他行二级索引等于 1 的数据是有效的那这个方法就没法用了。

再次尝试官方文档：

```python
df.drop('length', level=1)
```

<img src="https://raw.githubusercontent.com/Brycexxx/BlogComments/master/20181106193447.jpg"/>

呵呵，没毛病，算了，至于 index 参数怎么用以后遇到再说吧 :sleeping:

pandas 真的是太强大，只能怪自己太渣 :joy:



