# Enforcing Integrability

我们有$z_x^*$和$z_y^*$，欲结合 DCT 求$z$

$$
z(x, y) = \sum C(\omega)\varPhi(x, y, \omega)\\
z_x(x, y) = \sum C(\omega)\varPhi_x(x, y, \omega)\\
z_y(x, y) = \sum C(\omega)\varPhi_y(x, y, \omega)
$$

如果取基函数为傅里叶变换基，即$\varPhi(x, y, \omega) = e^{j\omega^T (x, y)}$，对于$N\times N$的图片，可取$\omega \in \{(2\pi n, 2\pi m)|n,m \in \{0,1,\cdots, N-1\}\}$

我们做同样的分解

$$
z_x^*(x, y) = \sum C_1(\omega) \varPhi_x(x, y, \omega)\\
z_y^*(x, y) = \sum C_2(\omega) \varPhi_y(x, y, \omega)
$$

欲得到$z_x$和$z_y$的估计值，最小化距离

$$
d\{(z_x^*, z_y^*), (\bar{z_x}, \bar{z_y})\} = \sum_x \sum_y (\bar{z_x} - z_x^*)^2 + (\bar{z_y} - z_y^*)^2
$$

带入上面的展开式，得到

$$
d\{(z_x^*, z_y^*), (\bar{z_x}, \bar{z_y})\} =\\ \sum_x \sum_y \\(\sum C(\omega)\varPhi_x(x, y, \omega) - \sum C_1(\omega) \varPhi_x(x, y, \omega))^2\\ +\\ (\sum C(\omega)\varPhi_y(x, y, \omega) - \sum C_2(\omega) \varPhi_y(x, y, \omega))^2\\
=\sum_x \sum_y [\sum_{\omega}(C - C_1)^2 \varPhi_x^2 + \sum_{\omega}(C - C_2)^2 \varPhi_y^2]
$$

由于$C, C_1, C_2$与对 xy 求和无关，换序

$$
d\{(z_x^*, z_y^*), (\bar{z_x}, \bar{z_y})\} = \sum_{\omega} [(C - C_1)^2 \sum_x \sum_y \varPhi_x^2 + (C - C_2)^2 \sum_x \sum_y \varPhi_y^2]\\
= \sum_{\omega} [(C - C_1)^2 P_x(\omega) + (C - C_2)^2 P_y(\omega)]
$$

对此进行最小化，变量为$C(\omega)$，使用类似最小二乘的方法

现考虑矩阵化上式，定义矩阵：

$$
\Phi_x(x, y) = [\varPhi_x(x, y, \omega_1), \cdots, \varPhi_x(x, y, \omega_n), \cdots]^T\\
\Phi_y(x, y) = [\varPhi_y(x, y, \omega_1), \cdots, \varPhi_y(x, y, \omega_n), \cdots]^T\\
C = [C(\omega_1), \cdots, C(\omega_n), \cdots]^T\\
C_1 = [C_1(\omega_1), \cdots, C_1(\omega_n), \cdots]^T\\
C_2 = [C_2(\omega_1), \cdots, C_2(\omega_n), \cdots]^T
$$

所以

$$
\begin{bmatrix}
    z_x^*\\z_y^*
\end{bmatrix}
=
\begin{bmatrix}
    \Phi_x^T & 0 \\
    0 & \Phi_y^T
\end{bmatrix}_{2 \times 2|\Omega|}
\begin{bmatrix}
    C_1 \\ C_2
\end{bmatrix}_{2|\Omega| \times 1}\\
\begin{bmatrix}
    \bar{z_x}\\\bar{z_y}
\end{bmatrix}
=
\begin{bmatrix}
    \Phi_x^T\\
    \Phi_y^T
\end{bmatrix}_{2 \times |\Omega|}
C
$$

考虑 N 点 DCT，一维形式

$$
X[k] = f(k) \sum_{n = 0}^{N - 1} x[n]\cos{\frac{\pi k (2 n + 1)}{2N}}\\
f(k) = \left\{
\begin{aligned}
& \sqrt{\frac{1}{N}} & k=0\\
& \sqrt{\frac{2}{N}} & k \ne 0
\end{aligned}
\right.
$$

矩阵形式

$$
X =
\begin{bmatrix}
    f(0)\cos{0} & \cdots & f(0)\cos{0}\\
    \vdots & &\vdots\\
    f(N)\cos{\frac{\pi N (2 \cdot 0 + 1)}{2N}} & \cdots & f(N)\cos{\frac{\pi N (2 (N - 1) + 1)}{2N}}
\end{bmatrix}
\begin{bmatrix}
    x[0]\\\vdots\\x[N - 1]
\end{bmatrix}
$$

二维只需要做“两次”，$Y = DXD^T$，易得逆变换$X = D^T Y D$

先考虑一维逆变换

$$
x[n] = \sum_{k=0}^{N-1}X[k]f(k)\cos{(\frac{\pi k (2 n + 1)}{2N})}\\
f(k) = \left\{
\begin{aligned}
& \sqrt{\frac{1}{N}} & k=0\\
& \sqrt{\frac{2}{N}} & k \ne 0
\end{aligned}
\right.
$$

基函数 $\varPhi(n; k) = f(k)\cos{(\frac{\pi k (2 n + 1)}{2N})}$ ，系数为 $X[k]$。对 n 求导，$\varPhi_n(n; k) = -f(k)\sin{(\frac{\pi k (2 n + 1)}{2N})}\frac{\pi k}{N}$

考虑二维逆变换

$$
z(x, y) = \sum_{u=0}^{N-1}\sum_{v=0}^{N-1}C(u, v) f(u)f(v) \cos{(\frac{\pi u (2 x + 1)}{2N})}\cos{(\frac{\pi v (2 y + 1)}{2N})}
$$

基函数 $\varPhi(x, y; u, v) = f(u)f(v) \cos{(\frac{\pi u (2 x + 1)}{2N})}\cos{(\frac{\pi v (2 y + 1)}{2N})}$

偏导

$$
\varPhi_x(x, y; u, v) = -f(u)f(v) \sin{(\frac{\pi u (2 x + 1)}{2N})}\cos{(\frac{\pi v (2 y + 1)}{2N})} \frac{\pi u}{N}\\
\varPhi_y(x, y; u, v) = -f(u)f(v) \cos{(\frac{\pi u (2 x + 1)}{2N})}\sin{(\frac{\pi v (2 y + 1)}{2N})} \frac{\pi v}{N}
$$

考虑一维情况偏导的变换

$$
X[k] = f(k) \sum_{n = 0}^{N - 1} x[n]\cos{\frac{\pi k (2 n + 1)}{2N}}\\
x[n] = \sum_{k=0}^{N-1}X[k]f(k)\cos{(\frac{\pi k (2 n + 1)}{2N})}\\
x_n[n]=\sum_{k=0}^{N-1} X[k] (-1)f(k)\sin{(\frac{\pi k (2 n + 1)}{2N})}\frac{\pi k}{N}\\
\begin{bmatrix}
    x_n[0]\\\vdots\\x_n[N-1]
\end{bmatrix}
=
\begin{bmatrix}
    \varPhi_x(0; 0) & \cdots & \varPhi_x(0; N - 1)\\
    \vdots & &\vdots\\
    \varPhi_x(N-1; 0) & \cdots & \varPhi_x(N-1; N -1)
\end{bmatrix}
\begin{bmatrix}
    X[0]\\\vdots\\X[N-1]
\end{bmatrix}
$$
