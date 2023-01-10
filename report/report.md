戴傲初 2022310778 InfoNeRF-jittor

Reference: [mjmjeong/InfoNeRF (github.com)](https://github.com/mjmjeong/InfoNeRF)

## 原理说明

**神经网络搭建**

神经网络共10层，输入位置与方向，输出RBG信息和神经辐射场。

第一层为输入维度到256维的线性层，之后8层256维的线性层。其中第四层重新添加输入信息以强化记忆；第八层额外添加特征层以学习神经辐射场参数。

后续由线性层降维到3，即RGB信息

损失函数共有三部分

$$\mathcal{L}_{RGB} = \frac{1}{|R|}\sum_{r\in R}\Vert C(r)-\hat{C}(r)\Vert_2^2$$

$$\mathcal{L}_{entropy} = \frac{1}{|\mathcal{R}_s|+|\mathcal{R}_u|}\sum_{r\in \mathcal{R}_s \cup\mathcal{R}_u}M(r)\odot H(r)$$

其中$R_s$为训练集采样光线、$R_u$为额外采样光线；
$$
M(r) = 
\begin{cases}
1\ \ Q(r) > \epsilon\\
0\ \ \text{otherwise}
\end{cases}\\
Q(r)=\sum_{i=1}^{N}1-exp(-\sigma_i \delta_i)\\
H(r) = -\sum_{i=1}^{N}p(r_i)\log p(r_i)\\
p(r_i) = \frac{\alpha_i}{\sum_j \alpha_i} = \frac{1-exp(-\sigma_i \delta_i)}{\sum_j 1-exp(-\sigma_i \delta_i)}
$$
在代码中，`network.NeRF`为网络主体、`run_infonerf.create_nerf`生成一个NeRF网络；`EntropyLoss`和`SmoothingLoss`为两个损失函数类。



**体渲染**

体渲染公式为
$$
\hat{C}(x, d) = \sum_{i=1}^{N}T_i(1-exp(-\sigma_i \delta_i))c_i\\
\delta_i = t_{i+1}-t_i\\
T_i = exp(-\sum_{j=1}^{i-1}\sigma_j \delta_i)
$$
在代码中，`run_infonerf.render`为体渲染计算函数

**训练过程**

**评价指标**



## 训练环境与参数

Ubuntu 22.04, Geforce RTX 3090, jittor

初始学习率为 5e-4

共50000次迭代

详细参数见configs/lego.txt



## 训练结果

测试集训练数据

| Iter  | Loss     | PSNR    | redefine_PSNR |
| ----- | -------- | ------- | ------------- |
| 20000 | 0.012849 | 18.9114 | 19.3429       |
| 25000 | 0.013371 | 18.7382 | 19.1881       |
| 50000 | 0.014128 | 18.4991 | 18.9636       |



<video src="D:\Courses\2022Autumn\grap\Infonerf-jittor\report\lego_spiral_050000_rgb.mp4"></video>

## 结果分析

