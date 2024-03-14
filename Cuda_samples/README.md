# Esempi semplici di programmi CUDA e C++



### init_array 

Programma che inizializza un array di N elementi. Ogni elemento
è inizializzato a 0 e successivamente viene sommato l'id del thread che lo inizializza.
Viene instanziato solamente un **blocco**. 
⚠ Inoltre si nota che con _n > 1024_ l'inizializzazione non funziona più.
Questo poichè di default i thread massimi per blocco sono _1024_


### 2d.cu

Programma che inizializza una matrice di NxM elementi in parallelo. Ogni elemento
è inizializzato con l'id del thread che lo inizializza.
Viene instanziato solamente un **blocco** , ma vengono utilizzati **thread.x** , **thread.y** e **thread.z**. 


### grid_example.cu

Programma che mostra la creazione di una **grid** avente numerosi **blocchi** al suo interno.
Utilizzo di `gridDim.x , gridDim.y , ...` e `blockDim.x , blockDim.y , ...`.

### class_example.cpp

Programma c++ che mostra le funzioni base di una classe.





# How to compile ? 

`nvcc <fileCuda.cu>`
