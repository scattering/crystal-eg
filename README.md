# crystal-eg
An implementation of the Epsilon-Greedy Algorithm for use with BLAND

## Requirements

To take advantage of this code, it is necessary to follow the steps listed at https://github.com/scattering/pycrysfml/blob/master/doc/Ubuntu_deps.txt within a linux terminal. 

### General Setup
The files inside this repository are intended to be located and run within the pycrysfml package. 

Specifically, all of these files should be placed within a folder of pathname: pycrysfml/hklgen/examples/sxtal/foldername, where one substitutes their foldername of choice for the word 'foldername'.

Provided one already has a .cfl and .int file premade, one should place those files into the folder a single tier above the code in this repository. 

For example, all of our code was inside folder pycrysfml/hklgen/examples/sxtal/epGreedy, while the .int and .cfl files were inside pycrysfml/hklgen/examples/sxtal.

### Setup within code

There are a number of changes to be made to file epGreedy.py:

1) line __: Substitute the names of your .int and .cfl files into the pathnames for variables 'observedFile' and 'infoFile'.
```shell
observedFile = os.path.join(DATAPATH,r"../<your filename here>.int")
infoFile = os.path.join(DATAPATH,r"../<your filename here>.cfl")
```

2) line __: Alter the parameters to suit your variable of interest. 

General formatting involves: <br />
m.atomListModel.atomModels[index of your atom of interest].[parameter of interest (coordinate)].value = initial guess <br />
m.atomListModel.atomModels[index of your atom of interest].[parameter of interest (coordinate)].range(lowerBoundaryOfUnitCell, upperBoundaryOfUnitCell)

For example, for our simulated MoTe2 trials, these lines were:
```shell
m.atomListModel.atomModels[0].z.value = 0.5
m.atomListModel.atomModels[0].z.range(0,1)
```
