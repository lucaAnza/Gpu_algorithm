# Experiment Cuda



## How to compile?

`nvcc -lcublas <fileCuda.cu>`


## What is it?

In this directory you can find some experiments for cuda optimization

## Usefull links

- [Thrust-algoritm sort_function()](https://nvidia.github.io/cccl/thrust/api/groups/group__algorithms.html)
- [Cudacublas min_function()](https://docs.nvidia.com/cuda/cublas/index.html#cublasi-t-amin)





### opt1_simulation.cu

#### Simulazione

Come passo intermedio, creo un programma più semplice ma molto simile al programma finale desiderato.

<img src="img/opt1.cu_1.png" width=50% alt=""> </img>

### opt1.cu


#### Obiettivo ottimizazione

<img src="img/opt1.png" width=50% alt=""> </img>

#### Steps

1.Capire quali variabili vanno trasportate su Gpu.

```mermaid
graph LR;
    Cpu --data-->B(Gpu memory type)
```


```c++
? mvKeys[]                             // Cuda Constant Memory   
vector<vector<size_t> > vRowIndices;   // Cuda Shared Memory   
float minZ                             // Cuda Constant Memory   
float minD                             // Cuda Constant Memory   
float maxD                             // Cuda Constant Memory   
ORBmatcher::TH_HIGH;                   // Cuda Constant Memory
mDescriptorsRight   //From frame.      // Cuda Constant Memory
mvInvScaleFactors   //From frame.      // Cuda Constant Memory  
mpORBextractorLeft  //From frame.      // Cuda Constant Memory  
mvScaleFactors      //From frame.      // Cuda Constant Memory
```








