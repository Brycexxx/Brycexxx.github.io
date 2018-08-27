---
title: CS212-Lesson4
date: 2018-08-26 20:55:43
tags: [test, python, cs212]
---

优达学城 CS212 第四课
“倒水”问题
<!--more-->
## 1 - Water Pouring Problem ##
### 1.1 - 问题描述 ###
给两个容积固定的杯子 A 和 B ，假设 A 和 B 的容积分别为 4 和 9 。杯子上并没有准确刻度，你可以将两只杯子的水互相转移，或者在水龙头处接水，或者将水完全倒掉。问题是如果给定 A 和 B 的最后状态假如 $(0,6)$，找出正确的操作步骤使得两杯的状态达到要求。
### 1.2 - 问题分析 ###
通过问题描述我们可以发现主要有以下6种操作：将 A 置空，将 A 倒入 B 中，将 A 装满；将 B不进行同样的操作。尽管我们知道如何操作，但是我们却不知道到底要操作多少次。不同于 $ODD+ODD==EVEN$数字映射问题，此种问题我们很清楚地知道只需要为字母挑选 5 个不同的数字来进行验证，也就是说每个候选答案的长度都是相同的。而这个问题，我们并不确定答案的长度，并不知道到底要操作多少次。所以这是一个搜索或者说探索问题，我们无法通过简单的穷举法进行解决。

前面我们说到共有 6 种不同的操作，也就是说从最初始的状态开始，每当我们面临旧状态比如 $(4,0)$（表示两杯中水量）我们都可以采取不同的动作达到一个新状态。这里有一个问题，当新状态比如 $(4,9)$ 再次产生 $(4,0)$ 时，那么我们的程序很可能就进入了无限的循环之中。所以对于探索问题我们需要设置一个已探索的标记`explored`用以记录已经探索获得状态防止重复探索陷入无限循环，由于状态不允许重复，故采用集合 set 存储。同时由于我们需要得到的结果是如何操作能达到问题给定的状态，所以还需要一个容器 frontier 用以存储操作顺序，并返回最终结果。对于给定的状态没有正确的答案的情况，为了保证返回值的一致性我们应该返回一个空列表，用以表示没有合适的路径；程序应该在 frontier 中没有待探索的路径的时候终止循环探索。

### 1.3 - 代码实现 ###
通过分析知道，首先我们需要实现针对当前状态产生后继新状态的函数，这个函数需要返回我们需要的新状态以及达到新状态需要采取的操作，所以采取字典作为返回值的数据结构：
```python
def successors(x, y, X, Y):
    """
    这里 x， y 是当前 A 和 B 的水量状态，X, Y是 A 和 B 的容积
    """
    # x, y 首先得符合基本条件，即不大于容器容积
    assert x <= X and y <= Y
    return {((x+y, 0) if x+y<=X else (x+(X-x), y-(X-x))): "X<-Y",
            ((0, x+y) if x+y<=Y else (x-(Y-y), y+(Y-y))): "X->Y",
            (0, y): "empty X", (x, 0): "empty Y",
            (X, y): "full X", (x, Y): "full Y"}
```
下面实现程序的主体部分：
```python
def pourWater(X, Y, goal, start=(0, 0)):
    # 定义搜索失败的返回值
    Failure = []
    # 如果初始状态就是我们的目标那么无需进行搜索
    if goal == start:
        return [start]
    # 定义已探索状态集合
    explored = set()
    # 定义待探索路径列表
    frontier = [[start]]
    while frontier:
        # 取出第一个路径进行探索
        path = frontier.pop(0)
        # 取出该探索路径当前的状态
        x, y = path[-1]
        for (state, action) in successor(x, y, X, Y).items():
            if state not in explored:
                explored.add(state)
                # 产生的新状态未被探索，所以添加到当前路径形成新路径
                path2 = path + [action, state]
                if state == goal:
                    return path2
                # 如果产生的新状态未达到目标，则将新路径加入带探索路径
                else:
                    frontier.append(path2)
    return Failure
```
### 1.4 - 代码测试###
假设我们给定目标 goal = (0, 6) ，初始状态 start = (4, 0) ：
```python
>>> print(pourWater(4, 9, (0, 6), (4, 0)))
[(4, 0), 'X->Y', (0, 4), 'full Y', (0, 9), 'X<-Y', (4, 5), 'empty X', (0, 5), 'X<-Y', (4, 1), 'empty X', (0, 1), 'X<-Y', (1, 0), 'full Y', (1, 9), 'X<-Y', (4, 6), 'empty X', (0, 6)]
```
程序给出了正确的操作流程。
### 1.5 - 小结 ###
该问题的主要需要厘清以下几个点：

 - 搞清问题的本质，这不是一个简单的组合问题，简单的穷举法并不适用；
 - 清楚地找出每次共有多少种操作方式，这是探索的基础；
 - 注意循环探索中可能出现无限循环的情况，需要设置标记记录。

## 2 - Bridge Problem ##
### 2.1 - 问题描述 ###
设有这么一座桥，一天晚上有4个人想过桥，这座桥每次只能通过 1 个或者 2 个人，并且需要手电筒。每个人通过桥所花的时间不同，假设以上 4 人分别需要的时间为 1 分钟、2 分钟、5 分钟、10 分钟。问题是找到过桥花费时间最短的方式。
### 2.2 - 问题分析 ###
仔细分析不难发现，这个问题与前面的“倒水”问题是十分相似的，每一种过桥的顺序也就是说候选答案的长度都是不一致的，所以这也是一个探索问题。所以我们需要为每个状态找到一个合理的表达方式。同样，在这个问题里，也存在`state`和`path`的概念，一个`state`里面应该包含岸两边的人员分布、手电筒在哪边；此处对于人的表达很方便，每个人的速度不同，那么我们自然可以直接通过速度来表示这个人，但是我们选择哪一种数据结构来表达呢？（tuple, list, set, frozenset），事实上这几种都是可以的，都可以较为方便地添加或删除。其中`tuple`和`frozenset`是可哈希的也就是不可变的。对于状态我们通过三个元素来表达，`here`和`there`分别表示岸两边的人员，`t`用来表达花费的时间；由于在整个搜索过程中，状态的长度是不需要改变的，所以我们选则元组来表达，即`(here, there, t)`。由于由于`frozenset`的可哈希的性能优势，以及可方便地添加、删除元素，所以选择`frozenset`来表达内部的`here`和`there`；比如：$here=frozenset([1, 2, 5, 10, 'light'])$，$there=frozenset()$。搞定状态的表达之后，我们还需要搞清楚如何产生后继状态，这些问题都清楚之后，那么和前面的“倒水”问题就差不多了。
### 2.3 - successors 的实现 ###
对于后继状态，首先我们可以分成两种情况，即根据`light`在岸的哪一边，因为只有在有灯的情况下才能够过桥，所以第一步按照有无灯分为`here->there`和`here<-there`；第二步，这里拿`here->there`来说明，过桥又分为两种情况，即一个人过还是两个人同时过，如果按照这种方式来，那么`successors`函数将会变得非常繁琐，首先进行一个人的遍历，然后进行两个人的遍历，最后将两次结果合并。所以这里便体现出`set`的优势，主要思想就是从`here`里面随便取出两个，放入`set`中，如果是两个不同的人，那么 set 的长度为 2 ，如果取出的两个是同一个人，那么 set 将自动去掉一个，长度为 1 。 人员的流动主要通过`set`的`|`和`-`实现，极为方便。时间，如果是两个人的话，就去其中较大的。代码实现如下：
```python
def successors(state):
    here, there, t = state
    if 'light' in here:
        return dict((here - frozenset([a, b, 'light']), 
                    there | frozenset([a, b, 'light']), 
                    t + max(a, b)), (a, b, '->') 
                    for a in here if a is not 'light' 
                    for b in here if b is not 'light')
    else:
        return dict((here | frozenset([a, b, 'light']), 
                    there - frozenset([a, b, 'light']), 
                    t + max(a, b)), (a, b, '<-') 
                    for a in there if a is not 'light' 
                    for b in there if b is not 'light')
```
### 2.4 - bridgeProblem 的实现 ###
这里不同于“倒水”问题，倒水问题是要找到最少的步骤解决问题，而这里需要找出用时最短的方法，自然地我们会想到通过对每一种方法花费时间进行排序。所以定义下面的函数：
```python
def elapsedTime(path):
    return path[-1][2]
```
下面实现 bridgeProblem 部分：
```python
def bridge_problem(here):
    """Modify this to test for goal later: after pulling a state off frontier,
    not when we are about to put it on the frontier."""
    Fail = []
    here = frozenset(here) | frozenset(['light'])
    explored = set() # set of states we have visited
    # State will be a (people-here, people-there, time-elapsed)
    frontier = [ [(here, frozenset(), 0)] ] # ordered list of paths we have blazed
    while frontier:
        path = frontier.pop(0)
        here1, there1, time = path[-1]
        if not here1 or here1 == set(['light']):
            return path
        for (state, action) in bsuccessors(path[-1]).items():
            if state not in explored:
                here, there, t = state
                explored.add(state)
                path2 = path + [action, state]
                frontier.append(path2)
                frontier.sort(key=elapsed_time)
    return Fail
```
上面代码中比较关键的地方在于判断是否达到目标的位置，“倒水”问题中，目标的判断是放在 for 循环内部的，如果这里不进行改变，实际上我们会错过最短的过桥时间，比如当前`frontier`中，存在的三条`path`的所花的时间分别为 12，13，14，这时我们会拿出时间为 12 的`path`继续探索，假如在这一步我们采取了一个花费时间为 5 的`action`，结果达到了目标，人都过桥了，那么这条新`path`没有加入到`frontier`中进行排序，直接作为最短路径返回，如果花费时间为 13 的`path`下一个状态花费的时间为 2 ，那么这条`path`的所花时间实际上会比已经返回的时间更短，也就是说我们错过了最短时间的过桥方法。

这里我们的解决方式是，将判断的语句提到 for 循环之外，这样就会将新产生的`path`和已经存在的`path`放在一起进行时间排序。这样，上面花费 17 的`path`就会被放到最后，保证捕获错过最优的`path`
### 2.5 - 小结 ###
从上面的分析知道，这个问题的关键在于我们需要考虑时间，不同于最短步长问题，最短步长问题每加上一个新的`state`就是一个步长，也就是说每个步长的权重是相等的，而最短时间问题就相当于是给不同的`state`赋予一个不同的权重权重的不同也就造成了尽管两条`path`的总长度一致，但总的时间花费不一致，如果不将判断部分放到`for`循环之外，就会可能得到花费时间更长的`path`。
## 3 - 总结 ##
这节课主要讲了两类不同的探索问题，总步长最短问题和总时间最短问题。这两类问题的共同点在于都无法通过简单的遍历所有情况来得到结果，需要一步一步地向前推进探索新状态，所有它们有一个共同的模式：首先我们需要考虑状态的表达，构造一个`successors`函数，然后我们需要一个`frontier`来存放候选的所有`path`，同时我们需要一个`explored`来存放已经探索过的状态，防止进入无限循环探索状态。不同点，我们要关注问题是要求哪个参数最短，针对这个问题，我们需要稍加思考，对代码做出相应的改动
