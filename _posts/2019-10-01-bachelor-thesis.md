---
title: From Prior Knowledge to Gaussian Prosses Hyperparameters in Regression
author: 
layout: post
---


Gaussian Process Regression (GPR) is a non-parametric approach to probabilistic modeling that is gaining increased traction within the machine learning community. Its effectiveness in predicting stochastic systems and dynamical control models is widely recognized. However, the method of integrating specific prior knowledge into the selection of hyperparameters remains an area of exploration.

<div class="math-text" markdown="1">

<h2 number="1."> The Squared Exponential Kernel Function</h2>


The squared exponential (SE) kernel is a cornerstone of Gaussian Processes (GPs), known for its  robust performance across a broad spectrum of applications. This kernel is characterized by two main parameters:
- **Signal Variance ($\sigma_{SE}^2$)**: This parameter controls the overall variability of the predictions. A higher signal variance allows for a broader range of function values, increasing prediction variability.
- **Length Scale ($l_{SE}$)**: The length scale dictates the smoothness of the function. A larger length scale smoothens the curve, suggesting that input points are highly correlated even when far apart.

The kernel function is defined as:
<div id="1.1" class="equation">
$$
k_{SE}(\textbf{x}, \textbf{x'}) = \sigma_{SE}^2 \exp\left(-\frac{1}{2} \sum_{i=1}^{d}\left(\frac{x_i - x'_i}{l_{SE}}\right)^2\right)
$$
</div>

The following plots helps the reader to gain an understanding of the effect of the hyperparameters on the Gaussian process and its squared exponenetial kernel function for the simplified case of $d=1$.
<div style="width=100%">
  <iframe src="{{ 'assets/interactive_plots/ba-gaussian-process/index.html' | relative_url }}"  width="90%" height="800pt" frameborder="0"></iframe>
</div>

</div>
<div style="margin-top: 30px;"></div>
<div class="math-text" markdown="1">

<h2 number="2."> From Kernel to Probabilistic Lipschitz Constant</h2>
In GPs, the kernel function's structure is directly linked to the behavior of the unknown function being modeled. Specifically, the probabilistic Lipschitz constant of the kernel $L_k$ provides insights into the continuity, as well as the differential behavior of the unknown function. Given a set $$\mathbb{X}$$ of dimensions $d$ the Lipschitz constant is related to the GP through its kernel:
<div id="2.1" class="equation">
  $$
  L_k = \max_{x, x' \in X} \left\| \begin{bmatrix} \frac{\partial k(x,x')}{\partial x_1}  \dots  \frac{\partial k(x,x')}{\partial x_d} \end{bmatrix}^T \right\|  
  $$
</div>

Furthermore, the following Lemma relates the GP to the supremum of the unknown function through its kenel. 

Lemma B.2 <span class="citation" ref="lederer-stability"></span>:
> Consider a centered Gaussian process with a continuously differentiable covariance function $$ k(\cdot, \cdot) $$ and let $$ L_k $$ denote its Lipschitz constant on the set $$ X $$ with maximum extension $$ r = \max_{\textbf{x}, \textbf{x}' \in \mathbb{X}} \|\textbf{x} - \textbf{x}'\| $$. Then, with probability of at least $$ 1 - \delta $$, the supremum of a sample function $$ f(\textbf{x}) $$ of this Gaussian process is bounded by
><div id="2.2" class="equation">
> $$
> \sup_{x \in X} f(\textbf{x}) \leq \sqrt{2 \log \left( \frac{1}{\delta} \right)} \max_{x \in X} \sqrt{k(\textbf{x}, \textbf{x})} + 12 \sqrt{6d}\left\{ \max_{x \in X} \sqrt{k(\textbf{x}, \textbf{x})} , \sqrt{rL_k} \right\}.
> $$
></div>

Lemma B.2 is derived by applying the metric entropy criterion, also known as Dudleys Theorem <span class="citation" ref="dudley-entropy"></span>, to bound the expected supremum.
Next, the $\epsilon$-covering number is derived and its integral over the supremum of the variance on $\mathbb{X}$, allowing for the deduction of an analytic upper bound. Through the transitive property of inequality, this upper bound holds true for the $\sup_{x \in X} f(\textbf{x})$. 
However, this step utilizes the metric entropy criterion as outlined in <span class="citation" ref="grunwalder-regret"></span>, building upon insights from <span class="citation" ref="massarat-concentration"></span>. Though incorporating a factor of $12$ is mathematically valid, the subsequent derivation of bounds adopts a more general perspective, using a constant $C$ similar to the original methodology in <span class="citation" ref="dudley-entropy"></span>. The constant $C$ remains deliberately unspecified, allowing readers to consider the definition of this value in relation to their own interest. This change has been made in retrospect and is not part of the original work.

Theorem 3.2 form <span class="citation" ref="lederer-stability"></span>:

> Consider a zero mean Gaussian process defined through the covariance kernel $k(\cdot, \cdot)$ with continuous partial derivatives up to the fourth order and partial derivative kernels
> <div id="2.3" class="equation">
> $$
> k^{\partial_i}(\textbf{x}, \textbf{x}') = \frac{\partial^2 }{\partial x_i \partial x'_i} k(\textbf{x}, \textbf{x}') \quad \text{for } i = 1, \ldots, d.
> $$
> </div>
> Let $$L^{\partial_i}_k$$ denote the Lipschitz constants of the partial derivative kernels $k^{\partial_i}(\cdot, \cdot)$ on the set $\mathbb{X}$ with maximal extension $ r = \max_{\textbf{x}, \textbf{x}' \in \mathbb{X}} \|\textbf{x} - \textbf{x}'\| $. Then, a sample function $f(\cdot)$ of the Gaussian process is almost surely continuous on $X$ and with probability of at least $1 - \delta_i$ it holds that
> <div id="2.4" class="equation">
> $$
>L_f = \left\| \begin{bmatrix}
\sqrt{2 \log \left( \frac{2d}{\delta_L} \right)} \max_{x \in X} \sqrt{k^{\partial_1}(\textbf{x}, \textbf{x})} + C\sqrt{6d} \max_{x \in X} \left\{ \sqrt{k^{\partial_1}(\textbf{x}, \textbf{x})} \sqrt{rL^{\partial_1}_k} \right\} \\
\vdots \\
\sqrt{2 \log \left( \frac{2d}{\delta_L} \right)} \max_{x \in X} \sqrt{k^{\partial_d}(\textbf{x}, \textbf{x})} + C\sqrt{6d} \max_{x \in X} \left\{ \sqrt{k^{\partial_d}(\textbf{x}, \textbf{x})} \sqrt{rL^{\partial_d}_k} \right\}
\end{bmatrix} \right\|
> $$
> </div>
> is a Lipschitz constant of $$f(\cdot)$$ on $$X$$.



Theorem 3.2 and Lemma B.2, adapted from <span class="citation" ref="lederer-stability"></span>, are introduced to link the kernel's properties to the continuity and probabilistic bounds of the unknown function's behavior.

</div>
<div style="margin-top: 30px;"></div>
<div class="math-text" markdown="1">

<h2 number="3."> From Probabilistic Lipschitz Constant to the Hyperparameters</h2>

This section establishes the relation between the expected porperties of the unknown function and the hyperparameters $\vartheta \in \mathbb{R}^2$ of the SE kernal in a GP. The relationship is founded on equation <a href="#2.2" class="cite-equation">Equation 2.2</a> and <a href="#2.3" class="cite-equation">Equation 2.3</a>. Due to the discontinuous maximum term, present in both equations, a case distinction is applied for the deduction of the hyperparameter bounds. The subsequent sections first establish the case destinction, under which the hyperparameter bounds can be solved, and the solution itself, respectively. This work will be limited to the univariate case, however, given a trivially computed analytical solution for $L_k$ and $L_k^{\partial_i}$, the results can easily be extended.

#### 3.1 Establishing the Case Distinction

In the following, the SE subscript applied to the kernel expression is omitted for notational simplicity, implying an SE kernel.
The computation of the Lipschitz constant for the partial derivative of the kernel function, denoted as $L_k^\partial$, is derived by applying the SE kernel (<a href="#1.1" class="cite-equation">Equation 1.1</a>) to the partial derivative (<a href="#2.3" class="cite-equation">Equation 2.3</a>) , introduced into <a href="#2.1" class="cite-equation">Equation 2.1</a>. This yields the following expression:

<div id="3.1" class="equation">
  $$
  L_k^{\partial} = \max_{x, x' \in \mathbb{X}}
  \left|
  \frac{\partial k^{\partial}(x, x')}{\partial x}
  \right| = 
  \max_{x, x' \in \mathbb{X}}
  \left|
  -\frac{(x-x')(3l^2-(x-x')^2)}{l^6}
  \right|k(x,x').
  $$
</div>

Identifying the optimal points of this equation involves finding the critical points of $$k^{\partial}$$,  where $$\partial_x \partial_{x'} k^{\partial}(x,x') = 0$$ and $$\partial_x^2 k^{\partial}(x,x') = 0$$. Once determined, substituting these critical points back into <a href="#3.1" class="cite-equation">Equation 3.1</a> allows us to compute the Lipschitz constant of the partial derivative as follows:

<div id="3.2" class="equation">
$$
L^\partial_k 
 = \sqrt{6(3-\sqrt{6})}\,
 \exp\left(\sqrt{\frac{3}{2}}-\frac{3}{2}\right)
 \frac{\sigma^2}{l^3}
 = \omega \frac{\sigma^2}{l^3},
$$
</div>
where $\omega$ is a constant.

Next, in order to calculate the Lipschitz constant for the SE kernel $L_k$ itself, we replicate this procedure for $k(x,x')$, yielding the optimization problem:

<div id="3.3" class="equation">
$$
L_k = \max_{x, x' \in \mathbb{X}}
\left|
\frac{\partial k(x, x')}{\partial x}
\right| = 
\max_{x, x' \in \mathbb{X}} \left| \frac{(x-x')}{l^2}  \right| k(x,x').
$$
</div>


Here, we introduce the critical points where $\partial_x \partial_{x'} k(x,x') = 0$ and $\partial_x^2 k(x,x') = 0$ into the equation to deduce the Lipschitz constant and obtain:
<div id="3.4" class="equation">
$$
L_k = \frac{\sigma^2}{l\sqrt{e}}.
$$
</div>

Having define the Lipschitz constants, we can substitute the expressions back into <a href="#2.3" class="cite-equation">Equation 2.3</a> and <a href="#2.4" class="cite-equation">Equation 2.4</a>, along side the trivially computed auto-covariances of $\sqrt{k^{\partial}(x, x)} = \frac{\sigma^2}{l^2}$ and $ \sqrt{k(x, x)} = \sigma^2$. This yields a dual equation system:

<div id="3.5" class="equation">
$$
L_f = 
\sqrt{2\log\left(\frac{2}{\delta_L}\right)} \frac{\sigma}{l}  + C \sqrt{6} \max 
\left\{
  \frac{\sigma}{l}, \sqrt{\frac{r\omega}{l}} \frac{\sigma}{l}
\right\},
$$
</div>

<div id="3.6" class="equation">
$$
\bar{f}(\cdot) = 
\sqrt{2\log\left(\frac{2}{\delta_L}\right)} \sigma + C \sqrt{6} \max 
\begin{Bmatrix}
\sigma, \sqrt{\frac{r}{l \sqrt{e}}}\sigma
\end{Bmatrix},
$$
</div>


From here, we can derive the hyperparameters $\vartheta$ by applying case distinctions, addressing the maximum term. Knowing that $\sqrt{e} > \omega > 0$, we consider three scenarios:

- **Case 1:** For $l > r\omega$, where $\sqrt{\frac{r\omega}{l}} < 1$ and $\sqrt{\frac{r}{l \sqrt{e}}} < 1$.
- **Case 2:** For $r\omega \geq l > \frac{r}{\sqrt{e}}$, where $\sqrt{\frac{r\omega}{l}} > 1$ and $\sqrt{\frac{r}{l \sqrt{e}}} < 1$.
- **Case 3:** For $\frac{r}{\sqrt{e}} > l$, where both $\sqrt{\frac{r\omega}{l}} > 1$ and $\sqrt{\frac{r}{l \sqrt{e}}} > 1$.

This structured approach facilitates the computation of the Lipschitz constants and the derivation of hyperparameters under different conditions.

#### 3.2 Solving for the Length Scale, Case by Case

This section introduces the equations used to compute the bounds on the hpyerparameters for the SE kernel on a case-by-case basis. 

<large> Case 1: $l > r\omega$ </large>

By addressing <a href="#3.5" class="cite-equation">Equations 3.5</a>, we can directly deduce the formula for the standard deviation $\sigma$. Given a maximum admissible suppremum $M_{sup} \geq \bar{f}(\cdot)$, we obtain:

<div id="3.7" class="equation">
$$
    M_{\text{sup}} \geq \sqrt{2\log\left(\frac{1}{\delta_L}\right)} \sigma + C \sqrt{6} \sigma
$$
</div>
Next, by solving for $\sigma$ we obtain
<div id="3.8" class="equation">
$$
    \sigma \leq \bar{\sigma}_1 =  \frac{M_{\text{sup}}}{\sqrt{2\log\left(\frac{1}{\delta_L}\right)} + C \sqrt{6}}
$$
</div>
The notation $$\bar{\sigma}_1$$ refers to the upper bound of the SE standard deviation for the first case. The notation is maintained for the following cases. 
In order to obtain a bound for the length scale, we solve <a href="#3.6" class="cite-equation">Equations 3.6</a> for $l$. Given a maximum admissible Lipschitz constant $L_{max} \geq L_f$, we obtain the lower bound:
<div id="3.9" class="equation">
$$
    l \geq  \frac{\sigma}{L_{\text{max}}} \left( \sqrt{2\log\left(\frac{2}{\delta_L}\right)} + C \sqrt{6} \right)
$$
</div>
By introducing $\sigma$ into <a href="#3.9" class="cite-equation">Equations 3.9</a>, the bound on $l$ becomes:
<div id="3.10" class="equation">
$$
    l \geq \underline{l}_1 = \frac{ M_{\text{sup}} \cdot \sqrt{2\log\left(\frac{2}{\delta_L}\right)} + C \sqrt{6}}{L_{\text{max}}\cdot\sqrt{2\log\left(\frac{1}{\delta_L}\right)} + C \sqrt{6}}
$$
</div>
Analogous to the bound for the standard deviation, the notation $\underline{l}_1$ denotes the lower bound for the length scale, given the first case. 


<large>  Case 2: $r\omega \geq l > \frac{r}{\sqrt{e}}$</large> 

While the bound on $\sigma$ obtained in the previous case (<a href="#3.7" class="cite-equation">Equations 3.7</a>) is maintained, and therefore $\bar{\sigma}_1 = \bar{\sigma}_2$, the new inequality, stemming from <a href="#3.5" class="cite-equation">Equations 3.5</a>, needs to be consisdered: 
<div id="3.11" class="equation">
$$
    L_{\text{max}} \geq \sqrt{2\log\left(\frac{2}{\delta_L}\right)} \frac{\sigma}{l} + C \sqrt{6r\omega} \frac{\sigma}{l^{3/2}}
$$
</div>
By introducing <a href="#3.7" class="cite-equation">Equations 3.7</a> into the equation, we obtain:
<div id="3.12" class="equation">
$$
    L_{\text{max}} \geq \left(\sqrt{2\log\left(\frac{2}{\delta_L}\right)} l^{-1} + C \sqrt{6r\omega}l^{-3/2}\right)\frac{M_{\text{sup}}}{\sqrt{2\log\left(\frac{1}{\delta_L}\right)} + C \sqrt{6}}
$$
</div>

In order to find the value of $\underline{l}_2$, we substitute $l = I_2^2$ and rearrange the equation into a cubic polynomial form: 

<div id="3.13" class="equation">
$$
L_{max} \left(\sqrt{2\log\left(\frac{1}{\delta_L}\right)} + C \sqrt{6}\right)I_2^3 - M_{sup} \sqrt{2\log\left(\frac{2}{\delta_L}\right)}I_2 - M_{supp} C \sqrt{6r\omega} \geq 0,
$$
</div>

Given equality in <a href="#3.13" class="cite-equation">Equations 3.13</a>, a solutions for $I_2$ can be determined through numerical methods or analytical approaches. An investigation into the analytical solution of $I_2$ reveals that a real solution exists only under the condition of $\delta_L > \exp(-3888)$. Depending on the sign of $I_2$, the bound $\underline{l}_2 = I_2$ is either an upper or lower bound. More precisely, if $sign(I) = 1$, then $l\leq\underline{l}_2$, and  $l\geq\underline{l}_2$ otherwise.
 
<large> Case 3: $\frac{r}{\sqrt{e}} > l$ </large> 

For the third and final case, the inequality for $\sigma$ is altered to the following form:
<div id="3.14" class="equation">
$$
\sigma  \leq  \frac{M_{sup}}{\sqrt{2\log\left(\frac{1}{\delta_L}\right)} + C \sqrt{\frac{6r}{l\cdot\sqrt{e}  }}}.
$$
</div>

The inequality now depends on $l$, necessitating a solution for the length scale before $\sigma$ can be obtained.
To find a solution for $\sigma$, we proceed analoguously to the previous case, leading to an equation that expresses $l$ in terms of $I_3^2$. The resulting third-degree polynomial equation is given by:

<div id="3.15" class="equation">
$$
  L_{max}\cdot\sqrt{2\log\left(\frac{1}{\delta_L}\right)} \cdot I_3^3 + L_{max}\cdot C \sqrt{\frac{6r}{\sqrt{e}}}\cdot I_3^2 - M_{supp}\sqrt{2\log\left(\frac{2}{\delta_L}\right)}\cdot I_3 - M_{supp}C \sqrt{6r\omega} \geq 0.
$$
</div>

Upon analytical examination, we discover that this equation does not admit a real solution, which respecting the condition imposed by the case distinction. Therefore, no single solution for $$\bar{\sigma}_3$$ and $$\underline{l}_3$$ can be found. Instead, the releationship between the two hyperparameters is governed by <a href="#3.14" class="cite-equation">Equations 3.14</a> and <a href="#3.5" class="cite-equation">Equations 3.5</a>. Therefore, the bound $$\bar{\sigma}_3$$ can be determined, only after selecting a length scale, and is conditioned to fulfill requirements of $$\sigma$$ for $M_{supp}$ and $L_{max}$:

<div id="3.16" class="equation">
$$
\sigma  \leq \bar{\sigma}_3 = \min\left(\frac{M_{supp}}{\sqrt{2\log\left(\frac{2}{\delta_L}\right)} + C \sqrt{\frac{6r}{l\cdot\sqrt{e}  }}}, \frac{L_{max}}{\sqrt{2\log\left(\frac{2}{\delta_L}\right)} l^{-1} + C \sqrt{6r\omega}l^{-3/2}}\right).
$$
</div>

</div>
<div style="margin-top: 30px;"></div>
<div class="math-text" markdown="1">

<h2 number="4."> Selecting Hyperparameters</h2>

Based on the results of the previous section, we can now impose bounds on the hyperparameters, ensuring the a priori knowledge expressed by <a href="#3.5" class="cite-equation">Equations 3.5</a> and <a href="#3.6" class="cite-equation">Equations 3.6</a> is respected. 
For the bound on the variance, we obtain the following case distinction:
<div id="3.17" class="equation">
$$
  \sigma 
  \begin{cases}
    \leq \bar{\sigma}_1  & \text{for } l \geq \frac{r}{\sqrt{e}} \\
    \leq \bar{\sigma}_3 & \text{for } l < \frac{r}{\sqrt{e}},
  \end{cases},
$$
</div>
where $$\bar{\sigma}_1$$ and $$\bar{\sigma}_3$$ are defined in <a href="#3.8" class="cite-equation">Equations 3.8</a> and <a href="#3.16" class="cite-equation">Equations 3.16</a> respecctively.
For the bound on the length scale, we obtain the following case distinction:
<div id="3.18" class="equation">
$$
  l  \begin{cases}
      \geq \underline{l}_1  & \text{for } l \geq r\omega \\
      \geq \underline{l}_2  & \text{for } r\omega \geq l > \frac{r}{\sqrt{e}}  \text{ and } sign(I_2) = 1 \\
      \leq \underline{l}_2  & \text{for } r\omega \geq l > \frac{r}{\sqrt{e}} \text{ and } sign(I_2) = -1 \\
      & \text{unbounded otherwise. }
\end{cases}
$$
</div>
where $$\underline{l}_1$$ is defined in <a href="#3.10" class="cite-equation">Equations 3.10</a> and $$\underline{l}_2$$ is the square of the solution $$I_2$$ to <a href="#3.13" class="cite-equation">Equations 3.13</a>.
The following plot depicts the resulting surface for the lower bound of $l$ in depending on $L_{max}$ and $M_{supp}$. The slider adjusts the supremum of the variance on the set $\mathbb{X}$, r. 

<div style="width: 100%; text-align: center; height: 620pt; display: flex; justify-content: center; align-items: center;">
  <iframe src="{{ 'assets/interactive_plots/ba-hyperparameter.html' | relative_url }}"  width="90%" height="620pt" frameborder="0"></iframe>
</div>

Considering the case-by-case solution, as well as the bounds defined by <a href="#3.18" class="cite-equation">Equations 3.13</a>, the behavior of $l$ can be interpreted. For small values of $L_{max}$, that is a tightly bounded rate of change in the unknwon function, the length scale increases sharply, apporach infinity for $L_{max}$ approaching zero. This behaviour is expected as an increased length scale results in extremely smooth sample functions, with minimal rates of change.  

The relationship with $M_{supp}$ is more complicated. An increase in $M_{supp}$ translates into an increased floor for $l$. Given that the extent of the bounded region increases with $M_{supp}$, one would expect that the length scales decreases, allowing for a larger rate of change and therefore increased fluctuations in the unknwon function, however an increase in $M_{supp}$ also results in an increase in the upper bound of $\sigma$ (<a href="#3.13" class="cite-equation">Equations 3.13</a>). This increase in $\sigma$ alows for a larger variation of the unknwon function in general, necessitating the decrease in the length scale. 

In areas where $M_{\text{supp}}$ is small and $L_{\text{max}}$ is large, there is no absolute minimum value set for the length scale. In these cases, the length scale is constrained based on its relationship with the chosen standard deviation. This scenario suggests that the constricted bound represented by $M_{\text{supp}}$ prescribes the permissible fluctuations before any violation occurs, depending on the potential variability of the unknown function. Lower variability allows for more significant fluctuation and vice versa. The interpretation is supported by the findings in <a href="#3.16" class="cite-equation">Equations 3.16</a>.


</div>
<div style="margin-top: 30px;"></div>
<div class="math-text" markdown="1">

<h2 number="4."> Evaluting the Hyperparmeter Bounds</h2>

The effect of the bounds can be determined empirically. The following interactive plots give the reader an intuition for the effect of the bounds on the unknown function of GPs, with a combination of $L_{max}$ of either 1 or 100 and $M_{supp}$ of either 1 or 100. The variance of $x$ on $\mathbb{X}$ is $r=1$ and $\delta_L=0.1$. The constant $C$ is chosen as $1$, which reduces the strictness of the bounds by a factor of 12. However, it can be observed that the bounds remain excessively strict, given $\delta_L=0.1$. 

<div style="display: flex; width: 100%;">
  <div style="flex: 1;">
    <iframe src="{{ 'assets/interactive_plots/ba-a-priori-val/index.html' | relative_url }}?L_f=100&f_bar=100" width="100%" height="750" frameborder="0"></iframe>
  </div>
  <div style="flex: 1;">
    <iframe src="{{ 'assets/interactive_plots/ba-a-priori-val/index.html' | relative_url }}?L_f=100&f_bar=1" width="100%" height="750" frameborder="0"></iframe>
  </div>
</div>
---
<div style="display: flex; width: 100%;">
<div style="flex: 1;">
  <iframe src="{{ 'assets/interactive_plots/ba-a-priori-val/index.html' | relative_url }}?L_f=1&f_bar=100"  width="100%" height="750" frameborder="0"></iframe>
</div>
<div style="flex: 1;">
  <iframe src="{{ 'assets/interactive_plots/ba-a-priori-val/index.html' | relative_url }}?L_f=1&f_bar=1"  width="100%" height="750" frameborder="0"></iframe>
</div>
</div>


</div>
<div style="margin-top: 30px;"></div>
<div class="math-text" markdown="1">
<h2 number="5."> Conclusion and Future Work</h2>

For further evaluation, the results can now be applied to a real-world system, to provide empirical results
for the application of the bound to a control problem. Most systems of relevance, necessitate the GP to perform over an input vector, calling for the extension of the bound to the case of $d>1$.

</div>


<div style="margin-top: 30px;"></div>
<div class="references-text" markdown="1">

<div class=reference id="lederer-stability" shorthand="LUK" publisher="Advances in Neural Information Processing Systems 32 (2019)." >
Lederer, Armin, Jonas Umlauft, and Sandra Hirche. "Uniform error bounds for gaussian process regression with application to safe control."
</div>

<div class=reference id="dudley-entropy" shorthand="DUD" publisher="Journal of Functional Analysis 1.3 (1967): 290-330." >
Dudley, Richard M. "The sizes of compact subsets of Hilbert space and continuity of Gaussian processes." 
</div>

<div class=reference id="grunwalder-regret" shorthand="GRU" publisher="Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010." >
Grünewälder, Steffen, et al. "Regret bounds for Gaussian process bandit problems."
</div>

<div class=reference id="massarat-concentration" shorthand="MAS" publisher=" XXXIII-2003. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007." >
Massart, Pascal. "Concentration inequalities and model selection: Ecole d'Eté de Probabilités de Saint-Flour"</div>
