[TOC]

# Terminology

---

## $W$

Sparse matrix representing connections of a graph. For example, we call the following graph *Small Graph*:

$W=\begin{bmatrix}0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0\\\end{bmatrix}$

Would represent the following graph:

![Small_Graph](C:\Users\juanj\Projects\RSCH\Documents\Notes\Small_Graph.png)

---

## $F$

Matrix representing a partition of the graph with n nodes into g groups. Has dimension $n \times g$. For small graph, an example of partition F would be

$F_{W}=\begin{bmatrix}1&0&0\\1&0&0\\1&0&0\\0&1&0\\0&1&0\\0&1&0\\0&1&0\\0&0&1\\0&0&1\\0&0&1\end{bmatrix}$

Which would represent the following partition:

![Small_Graph_Partition](C:\Users\juanj\Projects\RSCH\Documents\Notes\Small_Graph_Partition.png)

---

## $W_{Smooth} ,\Omega$

Also know as the **"Similarity Matrix of W"** is  a full matrix that contains a **measure of similarity** between some starting node $W_{i,j}$ and every other node in $W$. Can be interpreted as the probability of a random walker ending at each of the points from some starting point $W_{i,j}$.

$\Omega=Id+(\frac{\alpha}{1-\alpha}L)^{-1}$

- $\alpha$ is the probability of the random walker 
- $L=Id-WD^{-1}​$ is the graph's Laplacian Matrix

---

## $E$

---

## $\vec{p}$

---

