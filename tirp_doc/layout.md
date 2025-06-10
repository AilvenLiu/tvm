# Layout Systems in Tensor Compilers

## Axe (Ours)

### Axes

Axes $A = \{a_0, a_1, \dots, a_{n_A}\}$ is a special set of objects. Each element is called an axis.

We define $\mathbb{Z}A$ as the set of $n_A + 1$ rank tuples,
$$
\mathbb{Z}A = \{ (z_0@a_0, z_1@a_1, \dots, z_{n_A}@a_{n_A}) \mid z_i \in \mathbb{Z} \}
$$
where we write $z@a$ for $(z, a) \in \mathbb{Z} \times A$.

When the context is clear, we write $\sum_{i=0}^{n_A} z_i@a_i$ for $(z_0@a_0, z_1@a_1, \dots, z_i@a_i, \dots, z_{n_A}@a_{n_A})$.

We define scalar multiplication, tuple addition and tuple multiplication as 

$$
\begin{align*}
&\times : \mathbb{Z} \times \mathbb{Z}A \to \mathbb{Z}A \\
s &\times (z_0@a_0, z_1@a_1, \dots, z_{n_A}@a_{n_A}) = (sz_0@a_0, sz_1@a_1, \dots, sz_{n_A}@a_{n_A})
\end{align*}
$$

$$
\begin{align*}
& +: \mathbb{Z}A \times \mathbb{Z}A \to \mathbb{Z}A \\
(z_0@a_0, z_1@a_1, \dots, z_{n_A}@a_{n_A}) &+ (z_0'@a_0, z_1'@a_1, \dots, z_{n_A}'@a_{n_A}) = ((z_0 + z_0')@a_0, (z_1 + z_1')@a_1, \dots, (z_{n_A} + z_{n_A}')@a_{n_A}).
\end{align*}
$$

Note that this effectively defines a vector space over $\mathbb{Z}$ with basis $A$.

We also define a element-wise tuple product as 

$$
\begin{align*}
&\odot: \mathbb{Z}A \times \mathbb{Z}A \to \mathbb{Z}A \\
(z_0@a_0, z_1@a_1, \dots, z_{n_A}@a_{n_A}) &\odot (z_0'@a_0, z_1'@a_1, \dots, z_{n_A}'@a_{n_A}) = ((z_0 \times z_0')@a_0, (z_1 \times z_1')@a_1, \dots, (z_{n_A} \times z_{n_A}')@a_{n_A})
\end{align*}
$$


### Iter

A iter $I$ is a triple $(e, s, a)$, where

- $e$ is a strictly positive integer,which is the extent $[0, e)$ of the iter
- $s$ is an integer, which is the stride along the axis $a$ of the iter
- $a$ is the axis of the iter where $a \in A$

We write $e_I, s_I, a_I$ for the extent, stride, and axis of iter $I$.

**Associated Function:** 

A iter $I$ can be associated with a function $f_I: \mathbb{Z} \to \mathbb{Z}A$, where $f_I(x)=(x*s)@a$.

### Layout

A layout $L$ is a triple $(L_D, L_R, L_O)$ where

- $L_D$ is a tuple of $n_D$ iters $(I_0, I_1, \dots, I_{n_D})$, $n_D \geq 1$.
- $L_R$ is an unordered tuple of $n_R$ iters $(J_0, J_1, \dots, J_{n_R})$, $n_R \geq 0$.
- $L_O$ is an unordered tuple of $n_O$ iters $(K_0, K_1, \dots, K_{n_O})$ along with a tuple of $n_O$ integer offsets $o=(o_0, o_1, \dots, o_{n_O})$, $n_O \geq 0$.

Note that $L_R$ and $L_O$ can be empty, but $L_D$ cannot be empty.

**Associated Function** 

We start from $L_D$ only, and then add $L_R$ and $L_O$ to it.

Let $E_D = e_{I_0} \times e_{I_1} \times \dots \times e_{I_{n_D}}$ be the size of $L_D$ (we also call it the size of $L$, written as $E_L$). Then we have an isomorphism

$$
\iota : [0, E) \cong [0, e_{I_0}) \times [0, e_{I_1}) \times \dots \times [0, e_{I_{n_D}})
$$

where $\iota(x) = (\left\lfloor \frac{x}{e_{I_1} \times \dots \times e_{I_{n_D}}} \right\rfloor, \left\lfloor \frac{x}{e_{I_2} \times \dots \times e_{I_{n_D}}} \right\rfloor \mod e_{I_1}, \dots, \left\lfloor \frac{x}{e_{I_{n_D}}} \right\rfloor \mod e_{I_{n_D-1}}, x \mod e_{I_{n_D}})$.

Then, for a given layout $L = L_D$, we have an associated function $f_{L_D}: \mathbb{Z} \to \mathbb{Z}A$ defined as

$$
f_L(x) = f_{L_D}(x) = \sum_{i=0}^{n_D} f_{I_i}(\iota(x)_i) = \sum_{i=0}^{n_D} (x_i * s_{I_i})@a_{I_i}.
$$

With $L_O$ introduced, we offset the result of $f_{L_D}(x)$ by a constant w.r.t. $x$.

$$
f_{L_D}(x) + f_{L_O}(o) = f_{L_D}(x) + \sum_{i=0}^{n_O} f_{K_i}(o_i) = \sum_{i=0}^{n_D} (x_i * s_{I_i})@a_{I_i} + \sum_{i=0}^{n_O} (o_i * s_{K_i})@a_{K_i}
$$

With $L_R$ introduced, however, we map to a set of points in $\mathbb{Z}A$. We have

$$
f_{L_R}(r) = \sum_{i=0}^{n_R} f_{J_i}(r_i) = \sum_{i=0}^{n_R} (r_i * s_{J_i})@a_{J_i},
$$
where $r \in [0, E_R), E_R = e_{J_0} \times e_{J_1} \times \dots \times e_{J_{n_R}}$. We enumerate all possible $r$ and get a set of points in $\mathbb{Z}A$. Then finally we have $f_L: [0, E_D) \to 2^{\mathbb{Z}A}$ defined as

$$
\begin{align*}  
f_L(x) &= \{f_{L_D}(x) + f_{L_R}(r) + f_{L_O}(o) \mid r \in [0, E_R)\} \\
&= \{\sum_{i=0}^{n_D} (x_i * s_{I_i})@a_{I_i} + \sum_{i=0}^{n_R} (r_i * s_{J_i})@a_{J_i} + \sum_{i=0}^{n_O} (o_i * s_{K_i})@a_{K_i} \mid r \in [0, E_R)\}
\end{align*}
$$

**Compatible Domains of $f_L$**

Suppose there's an integer factorization of $E_L = \prod_{i=0}^{\alpha} e'_i$, then we can make $f_L$ accept an coordinate in space $[0, e'_0) \times [0, e'_1) \times \dots \times [0, e'_\alpha)$ by mapping it to $[0, E_L)$ using the isomorphism $\iota$ defined above. Such a coordinate space is called a compatible domain of $f_L$.

Hence, when we write $f_L(x)$ for some $x$ in a compatible coordinate space, we mean $f_L(\iota(x))$.

**Codomain size of $f_L$**

For a given axis $a \in A$, enumerating $x$ along $[0, E_L)$ and get it into $f_L$ spans a set of values $\text{Vals}_{f_L, a}$ in $\mathbb{Z}$ over that axis. 

We define the codomain size of $f_L$ over that axis the the size of $max(\text{Vals}_{f_L, a}) - min(\text{Vals}_{f_L, a}) + 1$.

If $L$ doesn't touch axis $a$, then $\text{Vals}_{f_L, a} = \emptyset$, and the codomain size of $f_L$ over that axis is defined as 0.

The codomain size of $f_L$ is over all axes $a \in A$ can be written as a element in $\mathbb{Z}A$ as 

$$
\text{cosize} (f_L) = \sum_{a \in A} \text{size}(\text{Vals}_{f_L, a})@a
$$


### Normalize

From the definition of layout associated function, we can see that there are many possible layouts that can be associated with the same function. It's important to answer if two layouts $L_A$ and $L_B$ are effectively the same (i.e. $f_{L_A}(x) = f_{L_B}(x)$ for all $x \in [0, E_A)$ and $E_A = E_B$). We answer this by normalizing $L_A$ and $L_B$, then checking if the normalized layouts are structurally the same.

A normalized layout $A^*$ of some layout $A$ is a layout $A*$ that satisfies the following conditions:

- $E_{A^*} = E_{A}$
- $f_{A^*}(x) = f_{A}(x)$ for all $x \in [0, E_A)$
- $A^*$ achieves the minimal rank of all layouts that have the same associated function as $A$. Specifically, for any layout $A'$ that satisfies the above two conditions, we have $n_D(A^*) \leq n_D(A'), n_R(A^*) \leq n_R(A'), n_O(A^*) \leq n_O(A')$.

**Algorithm**

We derive the normalized layout of a given layout $L$ is as follows:

- Remove iters with extent 1.
- If $L_D$ has two adjacent iters $I_i$ and $I_{i+1}$ with the same axis $a$, we merge them into a single iter $I = (e_{I_i}e_{I_{i+1}}, s_{I_{i+1}}, a)$ if $s_{I_i} = s_{I_{i+1}}$.
- Do the same for $L_R$ and $L_O$, except that the two iters don't have to be adjacent since they are unordered.

TODO (need to be proved, or disproved):
- The existence and uniqueness of the normalized layout.
  -  Maybe there isn't a minimal rank layout for some layouts
- The algorithm does guarantee the minimal rank of the normalized layout.

### Group by Shape

Given a layout $L$ and a shape (which is a tuple of integers) $S$, we derive a new layout $L||S$ such that it satisfies the following conditions:

- $E_{L||S} = E_L$
- $f_{L||S}(x) = f_L(x) \times S$ for all $x \in [0, E_L)$
- There exists a strictly increasing integer sequence $P = 0, P_1, P_2, \dots, (n_D)_{L||S}$ with rank(S) + 1 elements such that 
$$
\prod_{p=P_i}^{P_{i+1}-1} e_{({I_{L||S}})_{p}} = S_i \text{ for all } i \in [0, rank(S))
$$

Intuitively, we rearrange and group L_D iters into a new layout $L||S$ such that the product of the extents of the iters in each group equals to the corresponding element in $S$, while keeping the same associated function.

Note that there may not exist a such $L||S$ for some $L$ and $S$, so when we write $L||S$, we mean by default that $L$ can be grouped by shape $S$.

Note that we $L_R$ and $L_O$ are not affected by the grouping.

We call space $[0, S_0) \times [0, S_1) \times \dots \times [0, S_{rank(S)-1})$ the induced space of $S$, written as $\overline{S}$. Note that if $L||S$ exists,  $\overline{S}$ is a compatible domain of $f_{L||S}$ (and also $f_L$).

### Tile

**Preliminary**

Given $S_x$ and $S_y$ such that $rank(S_x) = rank(S_y) = r$, we define tiled space $\overline{S_x} || \overline{S_y}$ as

$$
\overline{S_x} || \overline{S_y} = [0, S_x[0]) \times [0, S_y[0]) \times [0, S_x[1]) \times [0, S_y[1]) \times \dots \times [0, S_x[r]) \times [0, S_y[r])
$$

Coorepondingly, a coordinate $x || y \in \overline{S_x} || \overline{S_y}$ is a tuple of $2r$ integers $(x_0, y_0, x_1, y_1, \dots, x_r, y_r)$ where $x \in \overline{S_x}$ and $y \in \overline{S_y}$.


**Definition**

Given Layout $A$, $B$ and shape $S_A$ and $S_B$ such that rank($S_A$) = rank($S_B$) = $r$, we define the tile layout of $A||S_A$ and $B||S_B$ as a layout $T$ that satisfies the following conditions:

- $E_T = E_A \times E_B$, and hence $E_T = E_{A||S_A} \times E_{B||S_B}$
- For any $x \in \overline{S_A}, y \in \overline{S_B}$, $f_T(x || y) = f_{A||S_A}(x) \odot \text{cosize}(f_{B||S_B}) + f_{B||S_B}(y)$

We write $T$ as $(A||S_A) \otimes (B||S_B)$.


## Linear Layouts (Triton)

### Definition of Linear Layouts

A **Linear Layout** is defined as a linear map between labeled vector spaces over the field of two elements, $\mathbb{F}_{2}$. In this field, addition corresponds to a logical XOR operation, and multiplication corresponds to a logical AND operation.

These layouts model the mapping between physical hardware resources (like registers, threads, and warps) and a logical tensor. The hardware resources and the tensor dimensions are treated as vector spaces over $\mathbb{F}_{2}$, and the layout itself is the linear function—represented by a matrix—that connects them.

For instance, a layout `L` can be defined as a map from the hardware resources to the dimensions of a logical tensor:
$L: Reg \times Thr \times Wrp \rightarrow \mathbb{F}_{2}^{n} \times \mathbb{F}_{2}^{m}$ 

Here:
* **Reg, Thr, Wrp**: Represent the vector spaces for registers, threads, and warps.
* $\mathbb{F}_{2}^{n} \times \mathbb{F}_{2}^{m}$: Represents the vector space of a 2D logical tensor.

This approach allows complex mappings, such as data swizzling and broadcasting, to be expressed uniformly through binary arithmetic on the bit-vectors of the inputs and outputs.

**Example**
$$
A = 
\begin{array}{c|cc|ccccc|c} & \text{Reg} & & \text{Thr} & & & & & \text{Wrp} \\ 
\hline j_0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 
\\ j_1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 
\\ j_2 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 
\\ j_3 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 
\\ i_0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 
\\ i_1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 
\\ i_2 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 
\\ i_3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 
\end{array}
$$

The columns and rows are annotated as follows:

Columns (Input from Hardware): The 8 columns of the matrix correspond to the 8 bits of the input vector $v$, which represents the hardware resources in the space $Reg \times Thr \times Wrp$.
  - Columns 0-1 correspond to the 2 bits for the **Register (Reg)**.
  - Columns 2-6 correspond to the 5 bits for the **Thread (Thr)**.
  - Column 7 corresponds to the single bit for the **Warp (Wrp)**.

Rows (Output to Tensor): The 8 rows of the matrix produce the 8 bits of the output vector $w$, which represents the logical tensor coordinates (i, j).
  - Rows 0-3 correspond to the 4 bits for the **`j` coordinate** ($w_{0:3}$).
  - Rows 4-7 correspond to the 4 bits for the **`i` coordinate** ($w_{4:7}$).

### Operators of Linear Layouts

Linear layouts can be combined and manipulated using several fundamental operators derived from linear algebra over $\mathbb{F}_{2}$.

**Composition**

The **composition** operator is used to combine layouts sequentially. Given two linear layouts, $L_1: U \rightarrow V$ and $L_2: V \rightarrow W$, their composition is defined as $L_2 \circ L_1$, which maps an element $u \in U$ to $L_2(L_1(u))$. If $M_1$ and $M_2$ are the matrix representations of $L_1$ and $L_2$, their composition is represented by the matrix product $M_2M_1$ over $\mathbb{F}_{2}$. This is useful for tasks like extracting a slice from a parent layout.

**Product**

The **product** operator is used to build a complex layout from simpler ones incrementally, such as progressing from registers to threads to warps. Given two linear layouts, $L_1: U_1 \rightarrow V_1$ and $L_2: U_2 \rightarrow V_2$, their product is defined as:
$L_1 \times L_2: U_1 \times U_2 \rightarrow V_1 \times V_2$
$(u_1, u_2) \mapsto (L_1(u_1), L_2(u_2))$.
The resulting matrix is a block-diagonal matrix formed from the individual layout matrices $M_1$ and $M_2$. This operation is also referred to as the direct sum of maps.

**Left Division**

Left division is the inverse of the product operation. A matrix $M$ is divisible on the left by a matrix $M_1$ if $M$ has a block-diagonal structure of the form:
$M= \begin{bmatrix} M_1 & 0 \\ 0 & M_2 \end{bmatrix}$.****
The result of the division is $M_2$, denoted as $M/_l M_1$. This operation is particularly useful for determining if a layout can be broken down into smaller layouts that are compatible with efficient hardware primitives.

**Right Inverse**

A surjective (or "onto") linear layout $L: U \rightarrow V$ has a **right inverse**. If $L$ is represented by a matrix $M$, its right inverse $M^{-1}$ is the least squares solution to the equation $MX = I$, where $I$ is the identity matrix. This can be computed using methods like Gaussian elimination over $\mathbb{F}_{2}$. The right inverse is essential for recovering the original hardware indices from the coordinates within the logical tensor.

### Restrictions

**Power-of-Two Shapes**

Linear layouts are fundamentally designed for tensor dimensions and hardware resource counts (like threads per block) that are powers of two.  This is because the underlying mathematical model uses binary arithmetic on the bits of resource and tensor indices, which aligns naturally with power-of-two structures common in GPU programming. 

**Mitigation Strategy**

The paper notes that this limitation can be addressed by using a larger, power-of-two-sized tensor and then masking the elements that fall outside the boundaries of the actual, non-power-of-two shape.  This workaround allows the framework to handle arbitrary shapes at the cost of some additional logic.

## CuTe

### Definition of a CuTe Layout

A **layout** $L$ is fundamentally a pair of positive integer tuples of matching dimensions, known as the **shape $S$** and **stride $D$**. This is represented as:

$$L = S:D$$

For example, a layout $A$ could be defined as $A = (6, 2):(1, 7)$.

### Core Terminology

* **Shape $S$**: A tuple of positive integers, $(M_0, M_1, ..., M_\alpha)$.
* **Stride $D$**: A tuple of positive integers with the same dimension as the shape, $(d_0, d_1, ..., d_\alpha)$.
* **Size**: The product of the shape's elements, $M = M_0 * M_1 * ... * M_\alpha$.
* **Cosize**: The size of the codomain of the layout function, often computed as $L(\text{size}(L) - 1) + 1$
* **Length**: The number of elements in the shape and stride tuples, which is $\alpha + 1$.
  * **Mode**: A single pair of corresponding shape and stride elements, $(M_k):(d_k)$, which can be viewed as a layout of length 1.
* **Layout Function $f_L$**: Every layout has an associated function, $f_L$, that maps a logical index from $[0, M)$ to a value in the natural numbers ($\mathbb{N}$), determined by the shape and stride.

### Layout Operators

The algebra of CuTe layouts includes several key operations for manipulating these data structures.

**Concatenation**

Concatenation combines two distinct layouts into a single, larger layout.

* **Definition**: Given two layouts, $L = S:D$ and $L' = S':D'$, their concatenation, denoted $(L, L')$, is formed by concatenating their respective shape and stride tuples.
* **Result**: The resulting layout is $S'':D''$, where $S''$ is the flattened tuple $(S, S')$ and $D''$ is the flattened tuple $(D, D')$.

**Complementation**

Complementation defines a layout's complement with respect to a given integer $M$.

* **Admissibility**: For the operation to be valid, the pair $\{A, M\}$ must be "admissible for complementation". For a sorted layout $A = (N₀, ..., Nₐ) : (d₀, ..., dₐ)$, this requires that:
    1.  The product $N_{i-1}d_{i-1}$ must divide $d_i$ for all $1 ≤ i ≤ \alpha$.
    2.  The product $N_\alpha d_\alpha$ must divide $M$.
* **Definition**: If the pair is admissible, the complement of $A$ with respect to $M$ is defined as:
    $\text{complement}(A, M) = (d_0, \frac{d_1}{N_0d_0}, \frac{d_2}{N_1d_1}, ..., \frac{M}{N_{\alpha}d_{\alpha}}) : (1, N_0d_0, N_1d_1, ..., N_{\alpha}d_{\alpha})$

The resulting layout $B = complement(A, M)$ satisfies several properties, including that its layout function $f_B$ is strictly increasing and that $size(B) = M / size(A)$.

- What complement actually does?

Let $\{A=(N_{0},...,N_{\alpha}):(d_{0},...,d_{\alpha}),M\}$ be an admissible pair and let $B = \text{complement}(A, M)$. If $C$ is the concatenated layout $(A, B)$, then the size of $C$ is $M$. Furthermore, the layout function $f_C$ restricts to a bijection on the interval $[0,M)$, meaning it creates a one-to-one mapping from $[0, M)$ to itself.

In simpler terms, combining a layout with its properly defined complement results in a new layout that perfectly and uniquely represents every element in a memory space of size $M$. The layout function essentially shuffles, or permutes, the indices from 0 to $M-1$.

**Composition**

Composition creates a new layout, $A ○ B$, whose function $f_{A ○ B}$ is equivalent to the composition of the functions of the original layouts, $f_A ○ f_B$.

* **Admissibility**: The operation is well-defined if the pair $\{S, B\}$ (where $S$ is the shape of $A$) is "admissible for composition". This has two main conditions when $B$ is treated as a concatenation of its modes ($B_0, ..., B_\alpha$):
    1.  For every mode $B_k$, the pair $\{S, B_k\}$ must be admissible for composition, which involves specific "left divisibility" rules.
    2.  The "intervals of definition" for each of these pairs must be disjoint to avoid collisions.
* **Definition**: If admissible, the composition $A ○ B$ is defined as the concatenation of the individual compositions of $A$ with each mode of $B$
   
**Logical Division**

Logical division is defined using the operations of composition and complementation.

* **Admissibility**: For the logical division $A / B$ to be defined, where $M$ is the size of $A$, two conditions must be met:
    1.  The pair $\{B, M\}$ must be admissible for **complementation**.
    2.  The pair $\{S, B\}$, where $S$ is the shape of $A$, must be admissible for **composition**.
* **Definition**: If these conditions are satisfied, the logical division is defined as the composition of $A$ with the concatenation of $B$ and its complement:
    $A / B := A ○ (B, \text{complement}(B, M))$

**Logicla Product**

Logical product $A \otimes B$ is defined as $(A, \text{complement}(A, \text{size}(A) * \text{cosize}(B)) \circ B)$.
