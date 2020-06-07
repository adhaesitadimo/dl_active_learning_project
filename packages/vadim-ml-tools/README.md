
# Vadim's tools for ML research


```python
! pip install -e .
```

    Obtaining file:///home/vadim/Documents/code/philips-nlp/vadim-ml-tools
    Installing collected packages: vadim-ml
      Running setup.py develop for vadim-ml
    Successfully installed vadim-ml
    [33mYou are using pip version 19.0.2, however version 19.0.3 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m


## IO tools

Save and load any data with zero friction


```python
from vadim_ml.io import load_file, dump_file
```


```python
dump_file({
    'x': 0.5,
    'y': 0.1
}, 'xy.pickle')
```


```python
load_file('xy.pickle')
```




    {'x': 0.5, 'y': 0.1}




```python
dump_file({
    'x': 10,
    'y': 11
}, 'xy.json')
```


```python
load_file('xy.json')
```




    {'x': 10, 'y': 11}




```python
!cat xy.json
```

    {"x": 10, "y": 11}

It can even save and load sequences as multiple files 


```python
dump_file({
    'x': 0.5,
    'y': 0.1
}, 'xy')
```


```python
dict(load_file('xy'))
```




    {'y': 0.1, 'x': 0.5}




```python
!ls xy
```

    x.pickle  y.pickle


## Memoization

Memoize any function


```python
from vadim_ml.memoize import memoize
```


```python
from time import sleep

@memoize
def square(x):
    sleep(10)
    return x * x
```


```python
%%time

print(square(117))
```

    13689
    CPU times: user 0 ns, sys: 2.76 ms, total: 2.76 ms
    Wall time: 10 s



```python
%%time

print(square(117))
```

    13689
    CPU times: user 323 µs, sys: 114 µs, total: 437 µs
    Wall time: 422 µs


Function cache can persist if you need it to


```python
from vadim_ml.memoize import disk_memoize
```


```python
@disk_memoize('squares')
def square(x):
    return x * x
```


```python
square(15)
```




    225




```python
!ls squares
```

    15.pickle


## Text Segmentation

A class for working with annotated texts


```python
from vadim_ml.segmentation import TextSegmentation
```


```python
seg = TextSegmentation(
    'I do not like you', [
    ((0, 1), 'person'), 
    ((14, 17), 'person')
    ])
```


```python
from IPython.display import HTML

HTML(seg.html())
```




<b title="person">I</b> do not like <b title="person">you</b>




```python
seg
```




    <person>I</person> do not like <person>you</person>




```python
seg.get_label(4) # Char 4 is not within an annotation
```


```python
seg.get_label(15)
```




    'person'




```python
seg.get_segment(0)
```




    ((0, 1), 'person')




```python
seg.get_segment(7)
```




    ((1, 14), None)




```python
seg.get_segment(14)
```




    ((14, 17), 'person')


