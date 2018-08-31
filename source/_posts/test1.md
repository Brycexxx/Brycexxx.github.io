---
title: CS212-Lesson3
date: 2018-08-26 20:31:20
tags: [test, python, cs212]
toc: true
reward: true
---

优达学城 CS212 第三课
多参数问题
<!--more-->

## 1 - 多参数函数 ##
### 1.1 - 修改函数需要考虑的因素
有些函数固定只含有一个参数或者两个参数，在某些时候我们需要同时传入多个参数实现相同功能的会十分不方便，举个例子：
```python
# 自定义一个加法函数
def sum(x, y):
    return x + y
```
这个时候我们就只能传入两个参数，实现两个数的加法，而很多时候我们需要实现多个数相加，这时上面的sum函数就不能满足我们的要求（当然python内置的sum函数是可以多数相加的）。

在真实情况下，我们需要考虑两个因素。因为这个函数会和其他许多函数产生关联，被其他函数调用。比如如果我们直接修改了传入参数的数目，那么其他函数通过两个参数调用的方式便发生错误；或者我们修改了返回值的形式，同样会造成其他函数的错误。所有修改的第一个原则就是保证**向后兼容（backward compatible）**，即保证原有的调用方式依然有效。

第二个需要考虑的因素就是**外部修改（external）**和**内部修改（internal）**。如果我们进行了内外部修改，但是保证其他函数的正确调用，那么此时仍是向后兼容的。比如：
```python
def seq(x, y): return ('seq', x, y)

# 假设进行修改后，可接受多个参数
def seq(x, y, z):
   #return ('seq', x, y, z)                                 --1
    return ('seq', x, ('seq', y, z))                        --2
```
为什么在修改后我们采取第二种返回形式呢，因为其他函数获取此函数返回的形式就是第二种，你给它返回4个值，它就不知道了。

### 1.2 - 内外部修改函数
下面我们进行`seq`函数的修改，函数由`seq(x, y)`变为`seq(x, *args)`。相应地，函数内部也要进行修改：
```python
def seq(x, *args):
    # 保持两个参数时依然可以调用，并和之前的结果一致
    if len(args) == 1:
        return ('seq', x, args[0])
    else:
        return ('seq', x, seq(*args))
```
可以看到，其实这样修改不难，但是依然要写多行代码重新定义`seq`函数，如果我们还存在例如`alt(x, y)`，那么我们要对`alt`做出和`seq`函数同样的内外部修改。这时我们就违背了“不要重复自己”原则（Don't repeat Yourself），代码出现了大量冗余。
```python
def alt(x, y): return ('alt', x, y)
```
那么最好的方式是什么呢？函数具有很好的组合效果，所以我们可以定义一个函数，传入要修改的函数，然后在此函数内部进行调用，最后返回一个可以接受多个参数的新函数。定义函数`n_ary(f)`：
```python
def n_ary(f):
    """
    Given binary function f(x, y), return an n_ary function such that f(x, y, z) = f(x, f(y,z)), etc. Also allow f(x) = x.
    """
    def n_ary_f(x, *args):
        return x if not args else f(x, n_ary_f(*args))
    return n_ary_f
```
对于上面的`*args`我我们可能被并不是很清楚函数到底是怎么处理的。通过试验我们发现其实`*args`其实把传入的第一个参数之外的多个参数打包成了一个元组（tuple），以下是试验过程:
```python
def seq(x, y): return ('seq', x, y)

def n_ary(f):
    def n_ary_f(x, *args):
    # 打印查看args到底是什么形式
        print(args)
        return x if not args else f(x, n_ary_f(*args))
    return n_ary_f

# 打印一下修改之前的结果
print(seq('a', 'b'))
print('*'*40)
# 将seq传入修改函数内返回一个可接受多参数的新函数
mulSeq = n_ary(seq)
print(mulSeq('a', 'b', 'c', 'd'))
------------------------------------------------------------
以下是试验结果：
修改之前的结果：('seq', 'a', 'b')
args在递归过程中的变化：('b', 'c', 'd')
                        ('c', 'd')
                        ('d',)
                        ()
最终的返回结果：('seq', 'a', ('seq', 'b', ('seq', 'c', 'd')))
```
通过上面的结果我们可以发现args以元组的形式进入内部，当再次传入递归函数内，通过`*`将元组解开，并拿出一个放到第一个参数位置，剩下的重新以元组的形式再次传入函数内部。

### 1.3 - 装饰器
事实上，上述修改函数的函数我们有一个术语叫做**装饰器（decorator）**。上面我们通过`mulSeq = n_ary(seq)`来修改函数返回新函数，这并不够简洁，python有专门用作装饰器的标记`@`：
```python
@n_ary
def seq(x, y): return ('seq', x, y)
```
上面这种方式还不完善，当我们使用`help`函数查看参数列表和文档说明，就出现问题了。
```python
>>> help(seq)
Help on function n_ary_f in module __main__:
```
当我们重新定义`seq`函数之后，函数名被自然地修改为`n_ary_f`，所以此时我们需要进一步修改函数以便能够返回原函数名以及相应说明文档。

此时我们需要导入python内置库，`functools`
```python
from functools import update_wrapper

def n_ary(f):
    def n_ary_f(x, *args):
        return x if not args else f(x, n_ary_f(*args))
    # 将原函数f的名字以及文档复制到新函数n_ary_f
    update_wrapper(n_ary_f, f)
    return n_ary_f
```
此时`help`的结果就变成了：
```python
Help on function seq in module __main__:
```
但是还有一个问题，貌似我们又违背了不重复原则，不光是对于`n_ary`装饰器我们需要在内部使用`update_wrapper`，每一个装饰器都应该有这个功能，所以我们应该考虑一个一劳永逸的办法。给装饰器函数也定义一个装饰器就可以解决这个问题：
```python
from functools import update_wrapper

def decorator(d):
    "make function d a decorator: d wraps a function fn"
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d
    
@decorator
def n_ary(f):
    """
    Given binary function f(x, y), return an n_ary function such that f(x, y, z) = f(x, f(y,z)), etc. Also allow f(x) = x.
    """
    def n_ary_f(x, *args):
        return x if not args else f(x, n_ary_f(*args))
    return n_ary_f
```
以上的`decorator`函数一次性解决了这个问题。原理如下，首先`n_ary`函数作为参数传入`decorator`函数，内部的`_d`函数就是`n_ary`函数被装饰过的新函数。在`_d`内部，`fn`就是作为参数传入`_d`的`seq`，所以在这里将`seq`复制到了`n_ary(f)`也就是`n_ary_f`。
一种更简洁的写法，使用了`lambda`函数，将`decorator`函数修改为：
```python
def decorator(d):
    return lambda fn: update_wrapper(d(fn), fn)
```

## 2 - memoization
在计算机科学中，记忆化是一种提高进程运行速度的优化技术。通过储存大计算量函数的返回值，当这个结果再次被需要时将其从缓存提取，而不用再次计算来节省计算时间。 

同样我们还是采取装饰器的方式对函数进行修改，这里以斐波那契数列作为例子，首先定义斐波那契函数:
```python
def fib(n): return 1 if n <= 1 else fib(n-1) + fib(n-2)
```
在斐波那契数列的计算过程中，普通的计算方法会将某些值计算多次，例如：
$$fib(5) = fib(4) + fib(3)$$ $$fib(4) = fib(3) + fib(2)$$ $$fib(3) = fib(2) + fib(1)$$
在上面的三次计算中 $n=2,3$ 的时候被计算了多次，如果我们设计一个缓存将前面已经计算过的结果保存下来，那么再次计算这个值的时候，我们可以直接在缓存中查询获取，这样就大大提高了计算效率。下面我们定义一个缓存装饰器，并同时定义一个调用次数计数器：
```python
@decorator
def memo(f):
	"""
	Decorator that caches the return value for each call to 
	to f(args). Then when called again with same args, we can
	just look it up. 
	"""
	cache = {}
	def _f(*args):
	    # print(args)
	    # print(cache)
		try:
			return cache[args]
		except KeyError:
		    print('*'*30)
			cache[args] = result = f(*args)
			return result
		except TypeError:
			# some element of args can't be a dict key.eg a list [1, 2, 3]
			return f(args)
	return _f
	
@decorator 
def countcalls(f):
	"decorator that makes the function count calls to it, in callcounts[f]."
	def _f(*args):
		callcounts[_f] += 1
		# print('call count')
		return f(*args)
	callcounts[_f] = 0
	return _f
```
然后再执行`fib`函数：
```python
# 不加缓存器
@countcalls
def fib(n): 
	return 1 if n <= 1 else fib(n-1) + fib(n-2)
	
callcounts = {}
print(fib(5), callcounts)
>>> 8 {<function fib at 0x7fc7468e7510>: 15}

# 添加缓存器
@countcalls
@memo
def fib(n): 
	return 1 if n <= 1 else fib(n-1) + fib(n-2)
	
callcounts = {}
print(fib(5), callcounts)
>>> 8 {<function fib at 0x7fc7468e7510>: 9}
```
显然，调用次数减少了6次，当n越大，使用缓存装饰器的优势将越明显。

对于我来说这里同时出现了两个装饰器函数还有递归，实在有些糊涂了。于是我进行了一定探索，希望搞清楚执行顺序。

通过添加print打印出相关结果，
```python
memo
countcalls
call count
(5,)
{}
******************************
call count
(4,)
{}
******************************
call count
(3,)
{}
******************************
call count
(2,)
{}
******************************
call count
(1,)
{}
******************************
call count
(0,)
{(1,): 1}
******************************
call count
(1,)
{(1,): 1, (0,): 1, (2,): 2}
call count
(2,)
{(1,): 1, (0,): 1, (2,): 2, (3,): 3}
call count
(3,)
{(1,): 1, (0,): 1, (2,): 2, (3,): 3, (4,): 5}
```
通过分析上述结果，可以知道装饰器函数调用的顺序为自下向上，然儿产生装饰作用的顺序是由上到下，首先 $n=5$ 从`countcalls`开始计数，然后达到`memo`。在这个斐波那契二叉树中红色的部分即添加缓存装饰器函数之后发生调用的9次，可以看到在这个二叉树中，直到 $n=1$，cache才不为空，当 $n=0$ 时，已经有缓存了。之后开始回溯，0之后的1，2，3直接可以从缓存中获取结果。
![此处输入图片的描述][1]


  [1]: http://wx3.sinaimg.cn/mw690/e7900ef2ly1fprqe1o82aj20l40dkwex.jpg
