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

**TODO: look up what $D^{-1/2}WD^{-1/2}$**

Note that $W D^{-1}$ is column stochastic since the definition of D makes it so that each individual column will add up to 1. This means that in a matrix product, $(W D^{-1})\vec{v}$ the vector $\vec{v}$ will keep this property (of being a stochastic row vector)

For this matrix, $W D^{-1}$ would look like the following:

$W D^{-1}=\begin{bmatrix}0&0.3333&0.5000& 0& 0& 0& 0& 0& 0& 0\\0.5000& 0&0.5000&0.3333& 0& 0& 0& 0& 0& 0\\0.5000&0.3333& 0& 0& 0& 0& 0& 0& 0& 0\\0&0.3333& 0& 0&0.3333& 0&0.2500& 0& 0& 0\\0& 0& 0&0.3333& 0&0.5000&0.2500& 0& 0& 0\\0& 0& 0& 0&0.3333& 0&0.2500& 0& 0& 0\\0& 0& 0&0.3333&0.3333&0.5000& 0& 0&0.3333& 0\\0& 0& 0& 0& 0& 0& 0& 0&0.3333&0.5000\\0& 0& 0& 0& 0& 0&0.2500&0.5000& 0&0.5000\\0& 0& 0& 0& 0& 0& 0&0.5000&0.3333& 0\end{bmatrix}$

And if we multiply by the vector (which is stochastic)

$\vec{v}=\begin{bmatrix}0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\\0.1000\end{bmatrix}$

We get the following vector which is also stochastic

$(W D^{-1})\ \vec{v}=\begin{bmatrix}0.0833\\0.1333\\0.0833\\0.0917\\0.1083\\0.0583\\0.1500\\0.0833\\0.1250\\0.0833\end{bmatrix}$



---

## $F$

Matrix representing a partition of the graph with n nodes into g groups. Has dimension $F_{(n   g)}$. For *small graph*, an example of partition F would be

$F_{W}=\begin{bmatrix}1&0&0\\1&0&0\\1&0&0\\0&1&0\\0&1&0\\0&1&0\\0&1&0\\0&0&1\\0&0&1\\0&0&1\end{bmatrix}​$

Which would represent the following partition of W:

![Small_Graph_Partition](http://www.juanjoneri.com/img/RSCH/Small_Graph_Partition.png)

We will call each of the columns of F an **indicator function** of the subset. And we will designate the vector as $\mathbb{I}_A$ where $A$ is the subset of the nodes of interest. 

We define $f_r$ to be the column $r$ of matrix $F$ a class or as a membership function $\pi(i):\{1 ..n\}\to \{1..R\} $

---

## $W_{Smooth} ,\Omega$

Is also a similarity matrix (adjacency matrix) for the graph $W$. However $\Omega$ is a dense matrix obtained by looking at the steady state *(limiting probability distribution)* of the following Discrete-time Markov chain (really just a stochastic proces)

$u^{k+1}=\alpha\ (W D^{-1})\ u^k+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

- $u^{(0)}=\frac{\mathbb{I}_A}{|A|}$ meaning that the random walker is located inside subset $A$
- $\alpha\in[0,1]$

This process represents a convex combination of two processes:

1. $ \alpha\ (W D^{-1})\ u^k$ the random walker visits one of its neghouring nodes with probability $\alpha$ *(as shown in u or p)*
2. $(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$ the random walker gets *teletransported* back to its original position in subset A

This means that:

- $\alpha = 0 \implies$ Walker stays confined to the original set
- $\alpha = 1 \implies$ Get a uniform distribution *(see definition of $u$)*

Because both parts of the process represent probability distributions (in particular $\frac{\mathbb{I}_A}{|A|}$ is column stochastic by definition as noted in $\vec{u}$ or $\vec{p}$ section and $(W  D^{-1})\ u^k$ is also column stochastic as shown in $D$ section), its convex combination represents a probability distribution too.

When we look at the steady state of this Markov chain we obtain the following relationship:

$u^{\infty}=\alpha\ (W D^{-1})\ u^\infty+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore (\mathbb{I}-\alpha\ (W D^{-1}))u^{\infty}=(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore\frac{1}{(1-\alpha)}(\mathbb{I}-\alpha\ (W D^{-1})) u^{\infty}=\frac{\mathbb{I}_A}{|A|}$

$\therefore u^{\infty}=(\frac{\mathbb{I}-\alpha\ (W D^{-1})}{1-\alpha})^{-1}\frac{\mathbb{I}_A}{|A|}$

We will define $\Omega$ to be the term in the right:

$\Omega=(\frac{\mathbb{I}-\alpha\ (W  D^{-1})}{1-\alpha})^{-1}$

$\therefore \Omega = ( \frac{\mathbb{I}}{1-\alpha}-\frac{\alpha}{1-\alpha} (W   D^{-1}))^{-1} $

$\therefore \Omega = ( \mathbb{I}\frac{1-\alpha+\alpha}{1-\alpha}-\frac{\alpha}{1-\alpha} (W   D^{-1}))^{-1} $

$\therefore \Omega = ( \mathbb{I}(1+\frac{\alpha}{1-\alpha})-\frac{\alpha}{1-\alpha} (W   D^{-1}))^{-1} $

$\Omega=(\mathbb{I}+\frac{\alpha}{1-\alpha}L)^{-1}$ where $L=\mathbb{I}-WD^{-1}$ is the graph's **Laplacian Matrix**

#### Show that $ \Omega_{ij}$ makes sense as measure of similarity 

**Reescribir**

Take $A = \{j\}$ and Redo reasoning from before **(#)** 

$$p^{\infty}(i) = P_\{j\}(i)$$ probability to be on $i$ if the stochastic process teletransport to $j$ .

So this gives us a similarity between the two vertex because if they are closs together then it has a high value, and if they are far away the probability is low for $j$  to get to $i$ (because it get traped going back to $j$ or near $j$ all the time).

Now that we have a measure of similarity we have the following claim:

*Claim:* $p^{\infty}_{\{j\}}(i) = \Omega_{ij} = \omega_{ij} $ 

*Proof:* $p_{i}^{\infty} = \Omega^{-1} . [0,0…1,…] $(1 in position j because in this case our set is only one point ). So then this just is the same as $\Omega_{ij}$. (multiplying a matrix by a vector with only one 1 in a row returns that row in the matrix).



------

## $W_{Smooth} ,\Omega$ New definition

Recall that to prove monoticity of our algorithm, we need to be using a matrix that is *positive definite* (all $W_{Smooth} ,\Omega$ that was defined in the previous section is not symetric, and hence we cannot guarantee that it will be positive definite. Because of this reason we look for a new definition of $W_{Smooth} ,\Omega$.

The new definition is also a similarity matrix (adjacency matrix) for the graph $W$. But now lets define the following process:  

$u^{k+1}=\alpha\ (D^{-1/2}WD^{-1/2})\ u^k+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

- $\alpha\in[0,1]$

Because this process now does not represent probability distributions (in particular $\frac{\mathbb{I}_A}{|A|}​$ is column stochastic by definition as noted in $\vec{u}​$ or $\vec{p}​$ section but $(D^{-1/2}WD^{-1/2})u^k​$ is not column stochastic as shown in $D​$ section), its convex combination does not represent a probability distribution. Instead this process represents …… 

When we look at the steady state of this process we obtain the following relationship:

$u^{\infty}=\alpha\ (D^{-1/2}WD^{-1/2})\ u^\infty+(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore (\mathbb{I}-\alpha\ (D^{-1/2}WD^{-1/2}))u^{\infty}=(1-\alpha)\ (\frac{\mathbb{I}_A}{|A|})$

$\therefore\frac{1}{(1-\alpha)}(\mathbb{I}-\alpha\ (D^{-1/2}WD^{-1/2})) u^{\infty}=\frac{\mathbb{I}_A}{|A|}$

$\therefore u^{\infty}=(\frac{\mathbb{I}-\alpha\ (D^{-1/2}WD^{-1/2})}{1-\alpha})^{-1}\frac{\mathbb{I}_A}{|A|}$

We will define $\Omega$ to be the term in the right:

$\Omega=(\frac{\mathbb{I}-\alpha\ (D^{-1/2}WD^{-1/2})}{1-\alpha})^{-1}$

$\therefore \Omega = ( \frac{\mathbb{I}}{1-\alpha}-\frac{\alpha}{1-\alpha} (D^{-1/2}WD^{-1/2}))^{-1} $

$\therefore \Omega = ( \mathbb{I}\frac{1-\alpha+\alpha}{1-\alpha}-\frac{\alpha}{1-\alpha} (D^{-1/2}WD^{-1/2}))^{-1} $

$\therefore \Omega = ( \mathbb{I}(1+\frac{\alpha}{1-\alpha})-\frac{\alpha}{1-\alpha} (D^{-1/2}WD^{-1/2}))^{-1} $

$\Omega=(\mathbb{I}+\frac{\alpha}{1-\alpha}L_\text{sym})^{-1}$ where $L_{\text{sym}}=\mathbb{I}-D^{-1/2}WD^{-1/2}$ is the graph's **Symmetric Laplacian Matrix**



### Show $\Omega$ is positive definite:

#### ADD PROOF FOR POS DEF IN PAPER

**Claim:**

 $\lambda$ is an eigenvalue of $L$ $\implies$  $1+\frac{\alpha}{1-\alpha}\lambda$  is an eigenvalue of $\Omega^{-1}=Id+\frac{\alpha}{1-\alpha}L$ 

##### Proof:

Let $\lambda$ be an eigenvalue of $L$ corresponding to the eigenvector $u$

$(Id+\frac{\alpha}{1-\alpha}L)u=u+\frac{\alpha}{1-\alpha}Lu=u+\frac{\alpha}{1-\alpha}\lambda u=(1+\frac{\alpha}{1-\alpha}\lambda)u$

$\therefore (1+\frac{\alpha}{1-\alpha}\lambda) $ is an egenvalue of $\Omega^{-1}$ which is positive since $\alpha \in [0, 1)$

**Claim:** 

$A$ has eigenvalue $\lambda \implies$ $A^{-1}$ has eigenvalue $\frac{1}{\lambda}$

**Proof:**

$$Ax = \lambda x$$ 

$$x=\lambda A^{-1}x$$

$$\frac{x}{\lambda} = A^{-1}x$$

**Claim:**

$\Omega$ is positive definite

**Proof:**

We have  $\Omega^{-1}$ has positive eigenvalues $\gamma$

$\therefore \Omega$ has eigenvalues $\gamma^{-1}$ also positive

$\therefore \Omega$ positive definite

------

## $Cut$

**Lemma:**

Let $Assoc(A_r)=\sum_{i \in A} \sum_{j \in A} w_{ij}$ and $Cut(A,A^{\subset})=\sum_{i \in A}\sum_{j \in A^{\subset}}w_{ij}$ then:

$max \sum_{r=1}^{R}Assoc(Ar)$

over all partition $(A_{1},...A_{r})​$ of V is equivalent to

$min \sum_{r=1}^{R}Cut(A_{r},A^{\subset}_{r})$

**Proof:**
Let  $E(A_{1},...,A_{r})=\sum_{r=1}^{R}Assoc(Ar)$
$=\sum_{r}\sum_{i \in A_{r}} w_{ij} $
$= \sum_{r}\sum_{i \in A_{r}}(\sum_{j \in V}w_{ij} -\sum_{j \in A^{\subset}_{r}}) $
$=\sum_{r}\sum_{i \in A_{r}}\sum_{j \in V} w_{ij}-\sum_{r}\sum_{i \in A_{r}}\sum_{j \in A^{\subset}_{r}}w_{ij} $
$=(\sum_{i \in V}\sum_{j \in V} w_{ij})-\sum_{r}Cut(A_{r},A^{\subset}_{r} w_{ij}) $
$=\sum_{i \in V} d_{i} - \sum_{r} Cut(A_{r},A^{\subset}_{r}) $

---

## $E$

We define energy as: $E(f_1 , …, f_r) = \sum^R_{r=1} f_r^T W f_r$

However because $W$ is not positive definite, we implement an alternative measure of similarity $\Omega$ and get a new definition of E:

$E(f_1 , …, f_r) = \sum^R_{r=1} f_r^T \Omega f_r$ 

### Theorem: $\nabla E(f_1 , …, f_r) = \Omega f$ and $\nabla^2 E(f_1 , …, f_r) = \Omega $ 

Recall 

$E(f_1 , …, f_r) = \sum^R_{r=1} f_r^T \Omega f_r$ 

Considering the following statement:

$f(x+h) = f(x) +  \left \langle \nabla f(x) , h\right \rangle + O(df^2) $ 

Then if we consider:

$(f+df)^T \Omega (f+df)$

$= f^t \Omega f + df^T \Omega f + f^T \Omega df + df^T \Omega df$

$= f^t \Omega f + 2 \left  \langle  \Omega f, df \right \rangle  + df^T \Omega df$ 

From this we conclude that 

$$\nabla E(f_1 , …, f_r) = \Omega f$$ and $$\nabla ^2 E(f_1 , …, f_r) = \Omega $$ 

### Theorem: $E(f_1 , …, f_r) = \sum^R_{r=1} f_r^T \Omega f_r$ is convex

**Notes**

Then the function is convex $f(x)=x^T A \ x$ 

Then each term in the function, which is a sum is also convex

By the Second-order conditions [convex optimization book]:

Assuming f is twice differentiable, that is, its Hessian at each point in $\text{dom}f$, which is open. Then f is convex if and only if $\text{dom}f$ is convex and its Hessian is positive semidefinite: for all $x \in \text{dom}f$. In our case dom is R^n so we know it is convex.

The other condition follows from the fact that we know that $HE(F^k) = \Omega$

We know $E(F^k) = \sum_i \sum_j \Omega_{ij} F_iF_j$ .. **ADD CON JUANJO**

In our case, our Energy function looks like the following:

$$E(f_1,f_2,…,f_r) = \sum_r f_r^T \Omega f_r$$

So if we let  $x=f_r$  , $\phi (f) = f^T \Omega f$ ,  $F=\{f_1,…,f_r\}$ 

Then $E(F)= \sum_r \phi(f_r) $ , where each $\phi (f_r)$ is a convex function. Since sum of convex functions is convex, then we conclude that $E(F)$ is convex.

###  Theorem: $\nabla E(F^k).(F^{k+1}-F^k)>0 \implies E(F^{k+1})>E(F^k)$

 $$(F^{k+1}-F^k) . \nabla E(F^k)>0$$

By definition of convexity: 

$$E(F^{k+1}) \geq E(F^k)+ \nabla E(F^k).(F^{k+1}-F^k)$$

$\nabla E(F^k).(F^{k+1}-F^k)>0$ then $E(F^{k+1})>E(F^k)$ 

$E$ being convex implies that the value of $E$ is always above its linearized approximation. We therefore need an alforithm that can guarantee that the linearized energy of a new partition $E(F^{k+1})$ be breater than the energy of the previous partition $E(F^k)$ increase, 

---

## $H$

We define $H^k = \nabla E(F^k)$ the heat bump at step $k$

So that $H^k_r = \nabla E(f^k_r)$ is the linearized energy at step k for the class r

### Theorem: $\nabla E(F^k) =H^{k} = \Omega F^{k}$

See avobe

## Algorithm

Given a partition $F^k$ we use a greedy algorithm with complexity $O(n)$ to construct a new partition $F^{k+1}$ such that:

$$(F^{k+1}-F^k) . \nabla E(F^k)>0$$

$(F^{k+1}-F^k) . \nabla E(F^k)>0 \iff \nabla E(F^k)  F^{k+1}>\nabla E(F^k) F^k $

Given $\nabla E(F^k) =H^{k} = WF^{k}$ the above is also equivalent to

$\sum_{r}\left \langle H^k_r, f^{k+1}_r \right \rangle \geq \sum_{r}\left \langle H^k_r, f^{k}_r \right \rangle$

### Theorem: $\sum_{r}\left \langle H^k_r, f^{k+1}_r \right \rangle \geq \sum_{i}\left \langle H^k_r, f^{k}_r \right \rangle \iff  \sum_{i} H^k_{i, \pi^{k+1}(i)} \geq \sum_{i} H^k_{i, \pi^{k}(i)}$

It suffices to show that $\sum_{r}\left \langle H_r, f_r \right \rangle =  \sum_{i} H_{i, \pi(i)}$

$$\sum_r \left \langle H_r^k, f_r^{k+1} \right \rangle \geq \sum_r \left \langle H_r^k, f_r^{k} \right \rangle$$ 

$$ \sum_r \sum_i H_r^k (i) f_r^{k+1}(i) \geq \sum_r \sum_i H_r^k (i) f_r^k (i)$$

$$\sum_i \sum_r H_{i,r}^k f_r^{k+1}(i) \geq \sum_i \sum_r H_{i,r}^k f_r^k (i)$$

Now letting$\pi(i):\{1 ..n\}\to \{1..R\} $ be the  membership function, we have

$$\sum_i H^k_{i, \pi^k+1 (i)} \geq \sum_i H^k_{i,\pi^k(i)}$$  (1)

-----

Now let $\Delta \subset V$ be the vertices who want th switch class, that is:

$$\Delta = \{ i \in V : \pi^k (i) \notin \text{arg} \text{ max}_r H_{ir}^k\}$$

Note that given any nonempty subset of $\hat{\Delta} \subset \Delta $, the partition

$$ \pi^{k+1} (i) = $$

$$\text{arg} \text{ max}_r H^k_{ir}$$ if $i \in \hat{\Delta}$ 

$\pi^k(i) $ otherwhise

satisfies the previous theorem conclusion with strict inequality. The question now is to find a subset $\hat{\Delta} \subset \Delta $ such that the new partition still satisfies the size constraints. We also want this subset $\hat{\Delta}$ to be as big as possible, otherwise the algorithm is going to stagnate.

In short, $\Delta$ is the subset of vertices which would change clss under plain thresholding. The problem is that if all vertices change classes typically the balance cosntrain can be broken. The key, however is taht not all the vertices in $\Delta$ need to change class: any subsett of $\Delta$, wven if it contains only one vertex, whill still lead to a decrease in the cut value. The goal is then to look for the subset $\hat{\Delta} \in \Delta$ that leads to the biggest gain in eq (1) whole not breaking the size constraing. So this lead to the optimization problem from before.

#### Steps

The algorithm is short, at each step we do:

* Compute $H^k = \Omega F^k$ where $F^k$ is the 0-1 indicator function of the current partition.
* To get $F^{k+1}$ we threshold the matrix $H^k$, but before to change the class of the vertices who are asking to change, we pause and solve a small linear program. This linear program determines which vertex will be allowed to change class. We will accept only the most beneficial change and we will guarantee that the size constraints are satisfied.

## $\vec{u}$ or $\vec{p}$

$\vec{u}$ is a vector representing the probability *(column stochastic)* for the random walker to be at each node. Hence, $\vec{u^{k+1}}$ tells us the probability of where this random walker will be next, based on:

- were it was before ($\vec{p^k}$)
- the definition of the graph ($W$)
- the links that each node has ($D$)

**Proof**

$P(X ^{k+1}=i) = \sum_j P(X^{k}=j)P(X^k = j)$

$p^{k+1}(i) = \sum_j p^k(j) \frac{w_{ij}}{d_j}$

$p_i^{k+1} = \sum_j \frac{w_{ij}}{d_j}p_j^k = WD^{-1}p^k$

Therefore, for a complete random process, the definition of the Markov chain would be the following:

$u^{k+1}=(W  D^{-1})\ u^k$

For example, if $u^{(0)}=\begin{bmatrix}0&0&0&0&1&0&0&0&0&0\end{bmatrix}$ (Meaning that the random walker is located in node 5 of Small Graph) after one step, the position distribution would be the following:

$u^{1}=(W  D^{-1})\ u^0=\begin{bmatrix}0&0&0&1/3&0&1/3&1/3&0&0&0\end{bmatrix}^{T}$

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

Because in each step, $u$ was multiplied by $(W  D^{-1})$ which is column sctchastic, we have $u^{\infty}$ also column stochastic, meaning that it is well defined as aprobability vector.

When defined in this way, the steady state of the process can be obtained by the following relationship on the degree vector $d$ defined in [$D$ section](## $D$)

$u^{\infty}=\frac{d}{|d|}$ where $|d|= \sum{d_i}$

In general such a process can be applied to any subset $A$ of the graph. In such case, the vector $u$ would define the indicator function A normalized to be column stochastic as follows:

$u=\frac{\mathbb{I}_A}{|A|}$





**BELLOW NOT REALLY NEEDED:**

By the spectral theorem, since $\Omega$ is symmetric, we can do the following:

Let $x$ be a vector, then by spectral theorem,

U is the eigenvector, $\Lambda$ a matrix with the eigenvalues $\lambda$ in the diagonal (in each of the directions of the eigenvectors, the function behaves like a parabola)

$x^T \Omega x = x^T U \Lambda U ^T x = (U^Tx)^T \Lambda U x)$ (Change of basis)

$=\left \langle U^T x, \Lambda U^T \right \rangle $

Now, see that this is of similar form to the definition of E.

Let $x=f_r$  , $\phi (f) = f^T \Omega f$ ,  $F=\{f_1,…,f_r\}$ and $\Phi (f)$ 

Then $E(F)= \sum_r \phi(f_r) $ 

