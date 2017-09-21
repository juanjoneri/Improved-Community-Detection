[TOC]

# Terminology

---

## $W$

Sparse matrix representing connections of a graph. For example, we call the following graph *Small Graph*:

$W=\begin{bmatrix}0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0\\\end{bmatrix}$

Would represent the following graph:
![Small_Graph](http://www.juanjoneri.com/img/RSCH/Small_Graph.png)

---
## $D$

Diagonal matrix containing the degree of edges (number of conections) for each node in a sparce matrix $W.$  

For example, Small Graph would have the following D matrix associated:

$D_{W}=\begin{bmatrix}2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 3 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 3 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 2 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 4 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 2 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 3 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 2\\\end{bmatrix}$

---

## $F$

Matrix representing a partition of the graph with n nodes into g groups. Has dimension $n \times g​$. For small graph, an example of partition F would be

$F_{W}=\begin{bmatrix}1&0&0\\1&0&0\\1&0&0\\0&1&0\\0&1&0\\0&1&0\\0&1&0\\0&0&1\\0&0&1\\0&0&1\end{bmatrix}$

Which would represent the following partition:

![Small_Graph_Partition](http://www.juanjoneri.com/img/RSCH/Small_Graph_Partition.png)

---

## $W_{Smooth} ,\Omega$

Also know as the **"Similarity Matrix of W"** is  a full matrix that contains a **measure of similarity** between some starting node $W_{i,j}$ and every other node in $W$. Can be interpreted as the probability of a random walker ending at each of the points from some starting point $W_{i,j}$. Hence,$\omega_{ij} = P_{A=\{j\}}(i)$.

$\Omega=Id+(\frac{\alpha}{1-\alpha}L)^{-1}$

- $\alpha$ is the probability of the random walker 
- $L=Id-WD^{-1}$ is the graph's Laplacian Matrix

It is important to note some things:

* $\omega_{ij} = P_{A=\{j\}}(i)$ (in this case we pick our subset $A$ to be just one node $\{j\}$.

---

## $E$

---

## $\vec{p}$

---

$\vec{p}$ is a vector representing the probability for the random walker to be at each node. Hence, $\vec{p^{k+1}}$ tells us the probability of where this random walker will be next, based on:

- were it was before ($\vec{p^k}$)
- the definition of the graph ($W$)
- the links that each node has ($D$):

$p^{k +1} = WD^{-1}p^k$

Derivate

$P(X ^{k+1}=i) = \sum_j P(X^{k}=j)P(X^k = j)$

$p^{k+1}(i) = \sum_j p^k(j) \frac{w_{ij}}{d_j}$

$p_i^{k+1} = \sum_j \frac{w_{ij}}{d_j}p_j^k = WD^{-1}p^k$

Now $p^{k+1} = \alpha W D^{-1}p^k + (1-\alpha) \frac{\mathbb{I}_A}{|A|}$ (Where A is a subset of the whole set, $\mathbb{I}_A$ are just 1 or 0) and $|A|$ size.

Say $\alpha = 0.8$, then this means 80$\%$ chance to go to a near vertex and 20 $\%$ chance to transporting to set A.

Why do we do this?

If we start with any distribution that is proportional to the degree of vertex and wait to infinity, the chances for the walker to be in a certain point is uniform across the graph.

But note now that with the modification (adding the $+(1-\alpha)$ etc) now there is a higher chance that we bring it to the set A, creating then a "bump" in this set. ( The reason for this is that a walker can only move to its neighbors so once it goes into the set A it is harder to get out of it). 

Heat bump: $p^{\Omega^{\alpha}} | _A = lim_{k \to \infty} p^k$ $\to$ Limiting distribution 

\textit{Note:} if $\alpha = 0$ this just gives us uniform distribution on set A.

Now, 
$$Ip^{\infty} = \alpha W D^{-1} p^{\infty} + (1-\alpha) \frac{\mathbb{I}_A}{|A|} $$
$$(I - \alpha W D^{-1}) p^{\infty} = (1-\alpha) \frac{\mathbb{I}_A}{|A|} $$
$$\frac{1}{1-\alpha} (I-\alpha WD^{-1}) p^{\infty} =\frac{\mathbb{I}_A}{|A|}$$
$$p^{\infty} = M_\alpha ^{-1} \frac{\mathbb{I}_A}{|A|} = p^{\Omega ^ {\alpha}}|_A$$
Where
$$ M_{\alpha} = \frac{1}{1- \alpha}(I - \alpha WD^{-1})$$
is considered to be $W_{smooth}$
