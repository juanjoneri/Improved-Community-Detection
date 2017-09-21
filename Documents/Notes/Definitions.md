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

**TODO:** show what is WD-1 (becomes colom stochastic  (suma a uno en una de las directions))

---

## $F$

Matrix representing a partition of the graph with n nodes into g groups. Has dimension $n \times g​$. For small graph, an example of partition F would be

$F_{W}=\begin{bmatrix}1&0&0\\1&0&0\\1&0&0\\0&1&0\\0&1&0\\0&1&0\\0&1&0\\0&0&1\\0&0&1\\0&0&1\end{bmatrix}$

Which would represent the following partition:

![Small_Graph_Partition](http://www.juanjoneri.com/img/RSCH/Small_Graph_Partition.png)

---

## $W_{Smooth} ,\Omega$

Also know as the **"Similarity Matrix of W"** is  a full matrix that contains a **measure of similarity** between some starting node $W_{i,j}$ and every other node in $W$. Can be interpreted as the probability of a random walker ending at each of the points from some starting point $W_{i,j}$. Hence,$\omega_{ij} = P_{A=\{j\}}(i)$ (in this case we pick our subset $A$ to be just one node $\{j\}$).

$\Omega=Id+(\frac{\alpha}{1-\alpha}L)^{-1}$

- $\alpha$ is the probability of the random walker 
- $L=Id-WD^{-1}$ is the graph's Laplacian Matrix
- **TODO** add definition of L as a laplacian matrix

It is important to note some things:

* $\alpha = 0 \implies$ stays in original set
* $\alpha = 1 \implies$ get a uniform distribution (that depends on the degree of the vertex)

###### Now, what is $\Omega_{ij}$? 

$\Omega_{ij} = \Omega . [0,0,…,1,0,0,…] = [\omega_{1j},…,\omega_{nj}]$

The above represents the steady state distribution, meaning that $\omega_{ij}$ is a valid notion of similarity because it turns to be the probability of a random walk to be in a specific node. **TODO EXPAND ON THIS AND ASK FOR HELP: WHY IS THIS NOTION OF W A VALID NOTION FOR SIMILARITY?: DONE!! SEE BELLOW AND MAKE IT PRETTY** 



*Claim:* $ \Omega_{\alpha} =( \frac{1}{1- \alpha}(I - \alpha WD^{-1}))^{-1} )^{-1}= \frac{\alpha}{1-\alpha}L$ 

Proof: $ \Omega_{\alpha} = \frac{1}{1- \alpha}((1-\alpha)I +/alpha I- \alpha WD^{-1})^{-1} = = \frac{\alpha}{1-\alpha}L  $ 

###### Proving $\Omega$ is positive definite:

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

Recall we want the energy $E$ to be convex, which looks like …. ADD

---

## $\vec{p}$

$\vec{p}​$ is a vector representing the probability for the random walker to be at each node. Hence, $\vec{p^{k+1}}​$ tells us the probability of where this random walker will be next, based on:

- were it was before ($\vec{p^k}$)
- the definition of the graph ($W$)
- the links that each node has ($D$):

$p^{k +1} = WD^{-1}p^k$



### Example

For our Small Graph, p would look like the following after 500 steps of the following code:

```matlab
// W and D as defined above
p = ones(10,1)./10;

graphPlot(W);
for i = 1:150
    p = W*(Dm*p);
end
p
```

$\vec{p}^{\infty}=\begin{bmatrix}0.0769\\0.1154\\0.0769\\0.1154\\0.1154\\0.0769\\    0.1538\\0.0769\\0.1154\\0.0769\end{bmatrix}$

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