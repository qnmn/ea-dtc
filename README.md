# Evolutionary algorithm for decision tree induction
## Required packages
The python packages are saved in `requirements.txt`, therefore simply run
```
$ pip install -r requirements.txt
```
Additionally, to enable visualization, one should install [GraphViz](https://graphviz.org/) on their system too.

## Execution
```
$ python3 evolve.py wine
$ python3 evolve.py glass
```

The generated tree will be saved in `Digraph.gv.pdf`.
