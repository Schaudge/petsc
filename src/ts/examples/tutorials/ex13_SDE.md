

# Stochastic Heat Equation in 2D

## I. Governing equation

We solve a stochastic heat equation in 2D:
$$
\begin{align}
u_{t}&= \Delta u+\sigma \zeta(t,\mathbf{x}), \\
u|_{t=0} &= b e^{c |\mathbf{x}|^{3}} 1_{\{|\mathbf{x}| \leq r\}}. \\
\end{align}
$$

* $\zeta(t,\mathbf{x})=\dot{W}^{Q}(t,\mathbf{x}) =dW^{Q}(t, \mathbf{x}) /dt.$

* Therefore, we may also write: 
  $$
  \begin{equation}
  du=(u_{xx}+u_{yy}) dt+\sigma dW^Q. \label{spde}
\end{equation}
  $$
  
* $\zeta(t,\mathbf{x})$ is a *space-time colored noise* (white in time, colored in space):

  * $\displaystyle \dot{W}^{Q}(t, \mathbf{x})= \sum_{j\ge1}\sqrt{q_j} \phi_j(\mathbf{x})\dot{w_j}(t).$

    * $q_j>0$ and $\phi_j(\mathbf{x})$ are eigenpairs of $Q$.

    * $Q$ is the covariance operator defined by 

      $\displaystyle (Q\phi)(\mathbf{x})=\int_{D}q(\mathbf{x},\mathbf{x}^{\prime})\phi(\mathbf{x}^{\prime})d\mathbf{x}^{\prime}, \mathbf{x}\in D,$

      with the covariance function  $q(\mathbf{x},\mathbf{x}^{\prime})$ as its kernel.

    * $\dot{w_j}(t) \text{ is the Gaussian white noise.}$

  * $\mathbb{E}\left[\zeta(t,\mathbf{x}) \zeta(s,\mathbf{x}^{\prime})\right]=\mathbb{E}\left[\dot{W}^{Q}(t, \mathbf{x}) \dot{W}^{Q}(s,\mathbf{x}^{\prime})\right]=\delta(t-s) q(\mathbf{x},\mathbf{x}^{\prime}).$

  * $q(\mathbf{x},\mathbf{x}^{\prime})$ is the covariance function for its spatial correlation structure; i.e., $q(\mathbf{x},\mathbf{x}^{\prime})=Cov[\zeta(t,\mathbf{x}),\zeta(t,\mathbf{x}^{\prime})]=\mathbb{E}\left[\zeta(t,\mathbf{x}) \zeta(t,\mathbf{x}^{\prime})\right], \forall t \in [0,\infty)$.

* Since the Gaussian white noise $\dot{w_j}(t)$ can be thought of as the derivative of a Brownian motion/Wiener process $w_j(t)$, we may write:
  $$
  \displaystyle W^{Q}(t, \mathbf{x})= \sum_{j\ge1}\sqrt{q_j} \phi_j(\mathbf{x})w_j(t). \label{Qwiener}
  $$

  - $ w_j(t)$ are i.i.d. Wiener process (Brownian motion). 
  - It is in the form of *Karhunen-Loéve (KL) expansion*.

* Here we consider:

  * $\mathbf{x}\in D=[0,1]^2, t\in[0,T]$. 
  * $q(\mathbf{x},\mathbf{y})$ is chosen from the covariance functions below:
    * $\exp\left(-\frac{|x_1-x^{\prime}_1|}{L_1}\right) \exp\left(-\frac{|x_2-x^{\prime}_2|}{L_2}\right)$ : Separable exponential
    * $\exp \left(\frac{-\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|}{L_{c}}\right)$ : Exponential
    * $\exp \left(\frac{-\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|^{2}}{2 L_{c}^{2}}\right)$ : Gaussian

## II. Discretization

To discretize $\eqref{spde}$, we apply Crank-Nicolson scheme:
$$
U^{n+1}_{i,j}-U^n_{i,j}=\frac{\Delta t}{2}\left(\frac{1}{\Delta x^2}(U^{n+1}_{i+1,j}-2U^{n+1}_{i,j}+U^{n+1}_{i-1,j})\\+\frac{1}{\Delta y^2}(U^{n+1}_{i,j+1}-2U^{n+1}_{i,j}+U^{n+1}_{i,j-1})+
\frac{1}{\Delta x^2}(U^{n}_{i+1,j}-2U^{n}_{i,j}+U^{n}_{i-1,j})+\\\frac{1}{\Delta y^2}(U^{n}_{i,j+1}-2U^{n}_{i,j}+U^{n}_{i,j-1})\right)+\sigma \left((W^Q)^{n+1}-(W^Q)^{n}\right).
$$
Since $w_j(t_{n+1})-w_j(t_n)\sim N(0, \Delta t)$, rewrite the stochastic part as:
$$
(W^Q)^{n+1}-(W^Q)^{n}=\sqrt{\Delta t}\sum_{j\ge1}\sqrt{q_j}\phi_j(\mathbf{x})\xi^n_{j}.
$$
, where $\xi^n_j:=(w_j(t_{n+1})-w_j(t_n))/\sqrt{\Delta t}\sim N(0,1)$ i.i.d. are easily sampled.

## III. Computation of KL expansion

Given $q(\mathbf{x},\mathbf{x}^\prime)$, we need to find eigenpairs {$q_{j},\phi_j$} of $Q$ by solving:
$$
\int_D q(\mathbf{x},\mathbf{x}^\prime)\phi_j(\mathbf{x^\prime})d\mathbf{x}^\prime=q_j\phi_j(\mathbf{x}).
$$
It is called Fredholm integral equation of the second kind.

### A. Approximation via collocation and quadrature

#### 	1. Collocation method

​	Let $\mathbf{x}_1,\mathbf{x}_2,…,\mathbf{x}_{P}$ be points in $
D$, define
$$
R_j:=\int_D q(\mathbf{x},\mathbf{x}^\prime)\phi_j(\mathbf{x}^\prime)d\mathbf{x}^\prime-q_j\phi_j(\mathbf{x}).
$$
​	If $R_j(x_k)=0$ for $k=1,…,P$,  {$q_j,\phi_j$} is called a collocation approximation.

#### 	2. Quadrature

​	Suppose the $\mathbf{x}_k$ are quadrature points with weights $\varpi_k$ chosen such that:
$$
\int_D q(\mathbf{x}_k,\mathbf{x^\prime})\phi(\mathbf{x^\prime})d\mathbf{x^\prime}\approx \sum^{P}_{i=1}\varpi_iq(\mathbf{x}_k,\mathbf{x_i})\phi(\mathbf{x_i}).
$$

###  B. Eigenvalue problem

Thus, we obtain the collocation approximation by solving:
$$
\sum^{P}_{i=1}\varpi_i q(\mathbf{x_k},\mathbf{x_i})\phi_j(\mathbf{x_i})=q_j\phi_j(\mathbf{x_k}), \ k=1,...,P.
$$
In matrix notation: 
$$
CW\psi_j=q_j\psi_j
$$
, where $C\in \mathbb{R}^{P\times P}$ is the covariance matrix for the quadrature points, $W\in \mathbb{R}^{P\times P}$ is the diagonal matrix of weights and $\psi_j$ contains point evaluations of $\phi_j$.

This is an eigenvalue problem for matrix $CW$, which may not be symmetric.  We may alternatively consider the symmetric matrix $K=W^{1/2}CW^{1/2}$, since then 
$$
Kz_j=q_jz_j \label{symeig}
$$
 with $z_j:=W^{1/2}\psi_j$.

### C. SVD decomposition

We may apply SVD decomposition to solve $\eqref{symeig}$:
$$
K=USV'
$$
, since $K$ is a symmetric positive definite. (Recall $q_j>0$ is assumed $\forall j\in \mathbb{N}$.)

Here $S$ contains the eigenvalues $q_j$ in non-increasing order, and $U=V$ contains all the corresponding eigenvectors $z_j$ of $K$. The point evaluations $\psi_j$ of the eigenvectors $\phi_j$ can be recovered by $\psi_j=W^{-1/2}z_j$ .

### D. Vertex-based quadrature

Partition $D=[0,1]^2$ into $(N_x-1)\times (N_y-1)$ rectangles with edge with side length $h_x=1/(N_x-1)$, $h_y=1/(N_y-1)$ and let $\mathbf{x_k}, k=1,2,…,P=N_xN_y$, be the vertices. We apply the quadrature (trapezoidal) rule with weights:
$$
w_{i}=\left\{\begin{array}{ll}{h_xh_y/ 4,} & {\text { if } x_{i} \text { is a corner of } D,} \\ {h_xh_y / 2,} & {\text { if } x_{i} \text { lies on an edge of } D,} \\ {h_xh_y,} & {\text { if } x_{i} \text { lies in the interior of } D.}\end{array}\right.
$$

### E. Truncated KL expansion

Consider the finite sum approximation of $W^Q$ in $\eqref{Qwiener}$:
$$
W_J=\sum^{N_{KL}}_{j=1}\sqrt{q_j}\phi_j(\mathbf{x})w_{j}(t).
$$
Here we keep all terms in KL expansion $N_{KL}=N_xN_y$.

## IV. Test

