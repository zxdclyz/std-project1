# std-project1

视听信息系统导论第一次大作业

## 方法参考

主要需要阅读论文的 2.1, 2.2 节

论文中给出的方法总结如下：

1. Find the average of the training images and use it as an initial estimate of the albedo, $\alpha(x,y)$.
2. Without doing any row or column permutations, sift out all of the full rows (with no missing measurements) of matrix $X$ to form a full submatrix $\tilde X$. The number of rows in $\tilde X$ is almost always larger than its number of columns, $k$.
3. Perform SVD on $\tilde X$ to find an initial estimate of matrix $S \in \mathbb{R}^{3\times k}$ which best spans the row space of $\tilde X$.
4. Find the vectors $b_j^{*}$(the rows of $B^*$) by performing the minimization in (7) and by using the elements of matrix X for the values of $x_{ij}$ and the columns of matrix $S$ for the values of $s_i$. The $S$ matrix is fixed to its current estimate.
5. Estimate a possibly nonintegrable set of partial derivatives $z_x^*(x,y)$ and $z_y^*(x,y)$ by using the rows of $B^*$ for the values of $\bold{b}(x,y)$ in (1). The albedo,$\alpha (x,y)$, is fixed to its current estimate.
6. Estimate (as functions of $\bar{c}(\bold{w})$) a set of integrable partial derivatives $\bar{z}_x(x,y)$ and $\bar{z}_y(x,y)$ by minimizing the cost functional in (13). (For more details on how to perform this minimization, see [16].)
7. Update the albedo $\alpha(x,y)$ by least-squares minimization using the previously estimated matrix S and the partial derivatives $\bar{z}_x(x,y)$ and $\bar{z}_y(x,y)$.
8. Construct $\bar{B}$ by using the newly calculated albedo $\alpha(x,y)$ and the partial derivatives $\bar{z}_x(x,y)$ and $\bar{z}_y(x,y)$ in (1).
9. Update each of the light source directions and strengths si independently using least-squares minimization and the newly constructed $\bar{B}$.
10. Repeat Steps 4 through 9 until the estimates converge.
11. Perform inverse DCT on the coefficients $\bar{c}(\bold{w})$ to get the GBR surface $\bar{z}(x,y)$.

## 维护者

[@zxdclyz](https://github.com/zxdclyz)

[@duskmoon314](https://github.com/duskmoon314)

[@BobAnkh](https://github.com/BobAnkh)

## 关联项目

[视听导第一次大作业](https://github.com/zxdclyz/std-project1)

[视听导第二次大作业](https://github.com/duskmoon314/std-project2)

[视听导第三次大作业](https://github.com/BobAnkh/std-project3)

