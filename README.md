# Coursera Machine Learning Class
## Stanford's ML Class
This repository contains the programming assignments for Coursera's Machine Learning Class by Stanford. All assignments are completed in Matlab as of right now, but I eventually plan to reimplement some of them in python.


## Python Implementations
To view the python implementations, first create and activate an anaconda virtual environment with the following commands:


```{r, engine='bash', count_lines}
conda env create -f environment.yml -n rainbow
```

This creates an virtual enviroment with Anaconda. To activate it, type

```{r, engine='bash', count_lines}
$ source activate rainbow
```

To update project with new packages, add them to environment.yml, then type in

```{r, engine='bash', count_lines}
$ conda env update -f environment.yml -n rainbow
```

To start up Jupyter Notebooks type

```{r, engine='bash', count_lines}
$ jupyter notebook
```
