[TOC]

# Terminology

---

## $W$

Sparse matrix representing connections of a graph. For example, we call the following graph *Small Graph*:

![Small_Graph](http://www.juanjoneri.com/img/RSCH/Small_Graph.png)

Its associated $W$ matrix woudl be the following:

$W=\begin{bmatrix}0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0\\0 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0\\\end{bmatrix}$

---
## $D$

Diagonal matrix containing the degree of edges (number of conections) for each node in a sparce matrix $W.$  

For example, *Small Graph* would have the following D matrix associated:

$D_{W}=\begin{bmatrix}2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 2 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 3 & 0 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 3 & 0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 2 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 4 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 2 & 0 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 3 & 0\\0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 2\\\end{bmatrix}$

This can be thought of as the diagonal matrix of the following vector we will call lower case $d$

$d=\begin{bmatrix}2&3&2&3&3&2&4&2&3&2\end{bmatrix}^T$

$D=diag(d)$

This can be implemented in matlab using the following code

```matlab
for i=1:length(W)
    D(i) = sum(W(i,:));
end
D = diag(D');
```

Note that $W\times D^{-1}$ is column stochastic since the definition of D makes it so that each individual column will add up to 1. This means that in a matrix product, $(W\times D^{-1})\times\vec{v}$ the vector $\vec{v}$ will keep this property (of being a stochastic row vector)

For this matrix, $W\times D^{-1}$ would look like the following:

$W\times D^{-1}=\begin{bmatrix}0&0.3333&0.5000& 0& 0& 0& 0& 0& 0& 0\\0.5000& 0&0.5000&0.3333& 0& 0& 0& 0& 0& 0\\0.5000&0.3333& 0& 0& 0& 0& 0& 0& 0& 0\\0&0.3333& 0& 0&0.3333& 0&0.2500& 0& 0& 0\\0& 0& 0&0.3333& 0&0.5000&0.2500& 0& 0& 0\\0& 0& 0& 0&0.3333& 0&0.2500& 0& 0& 0\\0& 0& 0&0.3333&0.3333&0.5000& 0& 0&0.3333& 0\\0& 0& 0& 0& 0& 0& 0& 0&0.3333&0.5000\\0& 0& 0& 0& 0& 0&0.2500&0.5000& 0&0.5000\\0& 0& 0& 0& 0& 0& 0&0.5000&0.3333& 0\end{bmatrix}$

And if we multiply by the vector (which is stochastic)

$\vec{v}=\begin{bmatrix}0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\end{bmatrix}$

We get the following vector which is also stochastic

$(W\times D^{-1})\ \vec{v}=\begin{bmatrix}0.0833\\0.1333\\0.0833\\0.0917\\0.1083\\0.0583\\0.1500\\0.0833\\0.1250\\0.0833\end{bmatrix}$



---

## $F$

Matrix representing a partition of the graph with n nodes into g groups. Has dimension $F_{(n \times g)}$. For *small graph*, an example of partition F would be

$F_{W}=\begin{bmatrix}1&0&0\\1&0&0\\1&0&0\\0&1&0\\0&1&0\\0&1&0\\0&1&0\\0&0&1\\0&0&1\\0&0&1\end{bmatrix}$

Which would represent the following partition of W:

![Small_Graph_Partition](http://www.juanjoneri.com/img/RSCH/Small_Graph_Partition.png)

We will call each of the columns of F an **indicator function** of the subset. And we will designate the vector as $\mathbb{I}_A$ where $A$ is the subset of the nodes of interest. 

---

## $W_{Smooth} ,\Omega$

Is also a similarity matrix (adjacency matrix) for the graph $W$. However $\Omega$ is a full matrix obtained by looking at the steady state *(limiting probability distribution)* of the following Discrete-time Markov chain:

$u^{k+1}=\alpha\ (W\times D^{-1})\ u^k+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

With $u^{(0)}=\frac{\mathbb{I}_A}{|A|}$ meaning that the random walker is located inside subset $A$

This process represents a convex combination of two processes:

1. $ \alpha\ (W\times D^{-1})\ u^k$ the random walker visits one of its neghouring nodes with probability $\alpha$
2. $(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$ the random walker gets *teletransported* back to its original position in subset A

This means that:

- $\alpha = 0 \implies$ Walker stays confined to the original set
- $\alpha = 1 \implies$ Get a uniform distribution *(see definition of $u$)*

When we look at the steady state of this Markov chain we obtain the following relationship:

$u^{\infty}=\alpha\ (W\times D^{-1})\ u^\infty+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore u^{\infty}(\mathbb{I}-\alpha\ (W\times D^{-1}))=(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore u^{\infty}\frac{1}{(1-\alpha)}(\mathbb{I}-\alpha\ (W\times D^{-1}))=\frac{\mathbb{I}_A}{|A|}$

$\therefore u^{\infty}=\frac{\mathbb{I}_A}{|A|}\times(\frac{\mathbb{I}-\alpha\ (W\times D^{-1})}{1-\alpha})^{-1}$

We will define $\Omega$ to be the term in the right:

$\Omega=(\frac{\mathbb{I}-\alpha\ (W\times D^{-1})}{1-\alpha})^{-1}$

$\therefore \Omega = ( \frac{\mathbb{I}}{1-\alpha}-\frac{\alpha}{1-\alpha} (W \times D^{-1}))^{-1} $

$\therefore \Omega = ( \mathbb{I}\frac{1-\alpha+\alpha}{1-\alpha}-\frac{\alpha}{1-\alpha} (W \times D^{-1}))^{-1} $

$\therefore \Omega = ( \mathbb{I}(1+\frac{\alpha}{1-\alpha})-\frac{\alpha}{1-\alpha} (W \times D^{-1}))^{-1} $

$\Omega=(\mathbb{I}+\frac{\alpha}{1-\alpha}L)^{-1}$ where $L=\mathbb{I}-WD^{-1}$ is the graph's Laplacian Matrix

#### Show that $ \Omega_{ij}$ represents a measure of similarity 

$\Omega_{ij} = \Omega . [0,0,…,1,0,0,…] = [\omega_{1j},…,\omega_{nj}]$

The above represents the steady state distribution, meaning that $\omega_{ij}$ is a valid notion of similarity because it turns to be the probability of a random walk to be in a specific node. **TODO EXPAND ON THIS AND ASK FOR HELP: WHY IS THIS NOTION OF W A VALID NOTION FOR SIMILARITY?: DONE!! SEE BELLOW AND MAKE IT PRETTY** 



*Claim:* $ \Omega_{\alpha} =( \frac{1}{1- \alpha}(I - \alpha WD^{-1}))^{-1} )^{-1}= \frac{\alpha}{1-\alpha}L$ 

Proof: $ \Omega_{\alpha} = \frac{1}{1- \alpha}((1-\alpha)I +/alpha I- \alpha WD^{-1})^{-1} = = \frac{\alpha}{1-\alpha}L  $ 

#### Show $\Omega$ is positive definite:

To prove monoticity of our algorithm, we need to be using a matrix that is *positive definite* (all *eigen values* are positive). 

**Claim:** $\Omega$ is positive definite:

##### Proof:

Recall: $\Omega=Id+(\frac{\alpha}{1-\alpha}L)^{-1}$

$L $ is positive definite (look at A tutorial on spectral clustering)

*Lemma:* if $A$ is positive definite, so is $A^{-1}$

*Proof:* 

$$Ax = \lambda x$$ 

$$x=\lambda A^{-1}x$$

$$\frac{1}{\lambda} = A^{-1}x$$

 *Claim:* If $/lambda$ is the eigen value of L, then $(1+ /frac{\alpha}{1-\alpha} \lambda)$ is an eigenvalue of $\Omega$

*Proof:* $u^T \Omega u= <u, \Omega u> = (I + \frac{\alpha}{1-\alpha})$ 

**TODO: ANOTAR EL RESTO DE LAS NOTAS DE JUANJO**



---

## $E$

**TODO**

---

## $\vec{u}$ or $\vec{p}$

$\vec{u}$ is a vector representing the probability *(column stochastic)* for the random walker to be at each node. Hence, $\vec{u^{k+1}}$ tells us the probability of where this random walker will be next, based on:

- were it was before ($\vec{p^k}$)
- the definition of the graph ($W$)
- the links that each node has ($D$)

For a complete random process, the definition of the Markov chain would be the following:

$u^{k+1}=(W\times D^{-1})\ u^k$

For example, if $u^{(0)}=\begin{bmatrix}0&0&0&0&1&0&0&0&0&0\end{bmatrix}$ (Meaning that the random walker is located in node 5 of Small Graph) after one step, the position distribution would be the following:

$u^{1}=(W\times D^{-1})\ u^0=\begin{bmatrix}0&0&0&1/3&0&1/3&1/3&0&0&0\end{bmatrix}^{T}$

after $\infty$ steps *(150)*, $u$ would look like follows:

$u^{\infty}=\begin{bmatrix}0.0769&0.1154&0.0769&0.1154&0.1154&0.0769&0.1538&0.0769&0.1154&0.0769\end{bmatrix}$

This same result can be obtained running the following MATLAB code *(note that since u converges it does not matter what the original definition of u is as long as it is column stochastic)*

```matlab
// W and D as defined above
u = ones(10,1)./10;

for i = 1:150
    u = (W*D^(-1))*u;
end
u
```

Because in each step, $u$ was multiplied by $(W\times D^{-1})$ which is column sctchastic, we have $u^{\infty}$ also column stochastic, meaning that it is well defined as aprobability vector.

When defined in this way, the steady state of the process can be obtained by the following relationship on the degree vector $d$ defined in [$D$ section](## $D$)

$u^{\infty}=\frac{d}{|d|}$ where $|D|= \sum{d_i}$

In general such a process can be applied to any subset $A$ of the graph. In such case, the vector $u$ would define the indicator function A normalized to be column stochastic as follows:

$u=\frac{\mathbb{I}_A}{|A|}$

#### Derivation

$P(X ^{k+1}=i) = \sum_j P(X^{k}=j)P(X^k = j)$

$p^{k+1}(i) = \sum_j p^k(j) \frac{w_{ij}}{d_j}$

$p_i^{k+1} = \sum_j \frac{w_{ij}}{d_j}p_j^k = WD^{-1}p^k$

Now we define a *"stochastic process"* such that: $p^{k+1} = \alpha W D^{-1}p^k + (1-\alpha) \frac{\mathbb{I}_A}{|A|}$ (Where A is a subset of the whole set, $\mathbb{I}_A$ is the indicator function (meaning it has 1 on points that belong on set A and 0 on the others, and $|A|$ size)

Note: $WD^{-1}p^k$ is a probability vector and so is $\frac{\mathbb{I}_A}{|A|}$ so adding these two still give a probability vector

Say $\alpha = 0.8$, then this means 80$\%$ chance to go to a near vertex and 20 $\%$ chance to transporting to set A.

###### Why do we do this?

If we start with any distribution that is proportional to the degree of vertex and wait to infinity, the chances for the walker to be in a certain point is uniform across the graph.

But note now that with the modification (adding the $+(1-\alpha)$ etc) now there is a higher chance that we bring it to the set A, creating then a "bump" in this set. ( The reason for this is that a walker can only move to its neighbors so once it goes into the set A it is harder to get out of it). 

Now, we want to determine the steady state of this stochastic process,:

Heat bump: $p^{\Omega^{\alpha}} | _A = lim_{k \to \infty} p^k$ $\to$ Limiting distribution 

$$Ip^{\infty} = \alpha W D^{-1} p^{\infty} + (1-\alpha) \frac{\mathbb{I}_A}{|A|} $$
$$(I - \alpha W D^{-1}) p^{\infty} = (1-\alpha) \frac{\mathbb{I}_A}{|A|} $$ **(#)**
$$\frac{1}{1-\alpha} (I-\alpha WD^{-1}) p^{\infty} =\frac{\mathbb{I}_A}{|A|}$$
$$p^{\infty} = \Omega_\alpha ^{-1} \frac{\mathbb{I}_A}{|A|} = p^{\Omega ^ {\alpha}}|_A$$
Where
$$ \Omega_{\alpha} = \frac{1}{1- \alpha}(I - \alpha WD^{-1})$$
considered to be $W_{smooth}$.

Note: it is important to note that in practice we *cannot* compute this $\Omega$ since it is a full matrix. Instead, we compute $p^{\infty}$ by doing the equation shown before and iterating until convergence (maybe 30 times). Again, we do not have acces to $\Omega$ but we can have access to $\Omega.  \frac{\mathbb{I}_A}{|A|} $

###### So now, why can we think of this $\Omega_{ij}$ a measure of similarity between vertex $i$ and $j$?

Take $A = \{j\}$ and Redo reasoning from before **(#)** 

$$p^{\infty}(i) = P_\{j\}(i)$$ probability to be on $i$ if the stochastic process teletransport to $j$ .

So this gives us a similarity between the two vertex because if they are closs together then it has a high value, and if they are far away the probability is low for $j$  to get to $i$ (because it get traped going back to $j$ or near $j$ all the time).

Now that we have a measure of similarity we have the following claim:

*Claim:* $p^{\infty}_{\{j\}}(i) = \Omega_{ij} = \omega_{ij} $ 

*Proof:* $p_{i}^{\infty} = \Omega^{-1} . [0,0…1,…] $(1 in position j because in this case our set is only one point ). So then this just is the same as $\Omega_{ij}$. (multiplying a matrix by a vector with only one 1 in a row returns that row in the matrix).