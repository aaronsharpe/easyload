# EasyLoad
This is a helper to load data generated by [special-measure](https://github.com/yacobylab/special-measure/wiki)

# Installation
Modified from MeasureMe.

Installing python projects can be a real pain. Here is how I personally set up lab computers to use measureme, but other ways can work fine. I do not like to use tools such as Conda or pipenv, because I find them to be a real headache.

1. On Windows computers I typically use GitHub Desktop for git.
2. I install the latest Python 3 from python.org, system wide. I make sure that I can run Python from the command line. On Windows this sometimes means running py or py3 or python3. I feel like it changes every time I do it, so just try them all and see what sticks. Later versions of Windows will annoyingly pop up some app store if you don't use the right incantation. Ignore that nonsense.
3. Next, make sure pip exists with python -m ensurepip. Again, pip might have some odd path like pip3.
4. Download and install easyload. Git clone, then pip install -e.
5. Test easyload by following the basic usage section below! (Thanks Joe)

# Basic usage

Import sm_load, here we will just import the two main functions.

```python
from sm_load import load, load2d
```

To load a single file, we ignore all of the trailing stuff that you can add in special measure and simply use the identifying integer. 

```python
data = load('/path/to/data/directory', 1)
```

The returned data is in a dictionary. To quickly check all the fields in the data file:

```python
print(data.keys())
```

If we want to load many files at once (eg. a 2D measurement), there are two main ways. We can either specify a list of file indices via something like the following

```python
data = load2d('/path/to/data/directory', range(1, 10))
```

Or we can automatically load all files with the same trailing identifier as a specific file number by passing a list with -1 as the final element.

```python
data = load2d('/path/to/data/directory', [1, -1])
```

Either case will return a dictionary where each parameter is returned as a 2D matrix. In the case of an ongoing measurement, `load2d` will pad the files with zeros to match length of the first sweep. 
