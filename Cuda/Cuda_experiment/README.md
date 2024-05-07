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
std::vector<std::vector<size_t>> vRowIndices    // Cuda Global Memory
std::vector<cv::KeyPoint> mvKeys                // Cuda Global Memory
std::vector<cv::KeyPoint> mvKeysRight           // Cuda Global Memory
float minZ                                      // Cuda Constant Memory   
float minD                                      // Cuda Constant Memory   
float maxD                                      // Cuda Constant Memory   
int TH_HIGH                                     // Cuda Constant Memory   
cv::Mat mDescriptors                            // Cuda Global Memory
cv::Mat mDescriptorsRight                       // Cuda Global Memory
std::vector<float> mvInvScaleFactors            // Cuda Global Memory
std::vector<float> mvScaleFactors               // Cuda Global Memory
std::vector<size_t> size_refer                  // Cuda Global Memory
...
```

<br>

2.Creare array per navigazione della struttura dati principale <b>vRowIndices</b> (array multi-dimensionale irregolare).

<b> size_refer </b> + <b> incremental_size_refer </b>

<img src="img/size_refer.png" width=50% alt=""> </img>

- size_refer -> Rappresenta il numero di colonne per ogni riga.
- incremental_size_refer -> Rappresenta il numero di colonne fino a quella riga(riga compresa).








