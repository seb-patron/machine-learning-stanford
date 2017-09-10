# Coursera Machine Learning Class
## Stanford's ML Class
This repository contains the programming assignments for Coursera's Machine Learning Class by Stanford. All assignments are completed in Matlab as of right now, and I have started reimplementing some of them as self containing ipython notebooks.


## Python Implementations
To view the python implementations, first create and activate an anaconda virtual environment with the following commands:


```{r, engine='bash', count_lines}
$ conda env create -f environment.yml -n ngml
```

This creates an virtual enviroment with Anaconda. To activate it, type

```{r, engine='bash', count_lines}
$ source activate ngml
```

To update project with new packages, add them to environment.yml, then type in

```{r, engine='bash', count_lines}
$ conda env update -f environment.yml -n ngml
```

To start up Jupyter Notebooks type

```{r, engine='bash', count_lines}
$ jupyter notebook
```
