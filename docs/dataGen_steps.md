# MRPNN 多特征 Descriptor 构造文档

基于论文 *Deep Real-time Volumetric Rendering Using Multi-feature Fusion*（SIGGRAPH 2023）整理。([ACM Digital Library][1])

## 1. 目标

为当前着色点 $u$ 构造网络输入 descriptor：

$$
z=\{\Sigma_1,\Sigma_2,\dots,\Sigma_K,G,\varsigma^\alpha,\gamma\}
$$

其中：

* $K=12$ 为 stencil 总层数
* $\Sigma_i$ 为第 $i$ 层的局部特征集合
* $G$ 为 HG phase 参数
* $\varsigma^\alpha$ 为 albedo 项
* $\gamma=\cos^{-1}(\omega\cdot l)$ 为视线与光照夹角 

---

## 2. 输入

构造单个 descriptor 需要以下输入：

* 当前着色点 $u$
* 当前视线方向 $\omega$
* 当前光照方向 $l$
* 原始体密度场 $\mu_0$
* phase 参数 $G$
* albedo $\varsigma$
* 预定义的 12 层 frequency-sensitive stencil
* 每层 stencil 对应的 mip-level $m_i$  

---

## 3. 预处理阶段

### 3.1 构造 density mipmaps

职责：为多尺度 stencil 采样提供不同分辨率的密度表示。

流程：

1. 从原始密度场 $\mu_0$ 出发
2. 递归下采样生成多层 density mipmaps $\mu_i$
3. 论文中使用 9 层，从 $256^3$ 到 $1^3$ 

---

### 3.2 构造 transmittance fields

职责：为方向无关散射部分提供辅助特征提示。

流程：

1. 对每个 mip-level $i$，使用缩放后的密度 $\beta^{(i+1)}\mu_i$
2. 沿光照方向做透射率积分
3. 得到第 $i$ 层 transmittance field：

$$
\tilde S_i=\exp\left(-\int \beta^{(i+1)}\mu_i(u),du\right)
$$

4. 正文经验设置 $\beta=0.578$  

---

### 3.3 准备 phase 的累计表示

职责：为每个 stencil 点提供 phase 特征所需的累计 phase 函数 $p'$。

流程：

1. 使用 volume-averaged phase function
2. 文中默认使用 HG，相参数为 $G$
3. supplementary 中定义 cumulative phase function：

$$
p'(\omega,\theta,c)=\int_{\Omega} p(\omega,\phi),d\phi,\quad
\Omega=\{\phi\mid (\theta\cdot\phi)>\cos(c/2)\}
$$

其中 $c$ 表示当前 stencil 点看到体元时对应的立体角张角。 

---

## 4. 构造 stencil

### 4.1 定义 stencil 总体结构

职责：确定 descriptor 在空间上的采样支撑。

流程：

1. 使用 frequency-sensitive stencil
2. 总层数 $K=12$
3. 前 $M=8$ 层为低频部分
4. 后 4 层为高频部分
5. 每层对应一个 mip-level $m_i$ 

---

### 4.2 构造低频 stencil

职责：采集 diffusive / low-frequency scattering 信息。

流程：

1. 低频部分由 spherical 与 intra-spherical 两类点组成
2. 共 8 层，总计 160 个点
3. 第 1 层含 8 个点，其中 1 个在中心，其余分布在球面
4. 后续层按 supplementary 的配置分配点数与 mip-level
5. 点分布通过迭代松弛生成，使其尽量均匀 

---

### 4.3 构造高频 stencil

职责：采集沿光照方向的 shadow-boundary / high-frequency 信息。

流程：

1. 高频部分沿光照方向放置
2. 共 4 层，索引 9–12
3. 每层 8 个点，总计 32 个点
4. 随着离中心距离增加，mip-level 增大，形成类似 cone tracing 的锥形分布 

---

## 5. 针对单个着色点构造 descriptor

### 5.1 将 stencil 放到当前点 $u$

职责：得到每个局部特征的实际采样位置。

流程：

1. 对第 $i$ 层第 $j$ 个 stencil 偏移 $\mathbf q_{i,j}$
2. 计算实际采样点：

$$
v_{i,j}=u+\mathbf q_{i,j}
$$

3. 后续所有局部特征均在 $v_{i,j}$ 处采样 

---

### 5.2 采样 density 特征

职责：描述当前点周围的多尺度密度分布。

流程：

1. 在 $v_{i,j}$ 处，从对应 mip-level $m_i$ 的 density mipmap 采样
2. 得到：

$$
F^\mu_{i,j}
$$

3. 对其做数值压缩：

$$
F^\mu_{i,j}\leftarrow \log(F^\mu_{i,j}+1)
$$

 

---

### 5.3 采样 scaled-transmittance 特征

职责：提供与方向无关散射结构相关的辅助信息。

流程：

1. 在 $v_{i,j}$ 处，从对应 mip-level 的 transmittance field 采样
2. 得到：

$$
F^S_{i,j}
$$

---

### 5.4 采样 phase 特征

职责：描述从光照方向到 stencil 点、再到视线方向的角向散射关系。

流程：

1. 对第 $i$ 个 stencil 点采用 phase 特征：

$$
P_i=p'(\omega,v_i-u)\cdot p'(v_i-u,l)
$$

2. 将其作为该点的 phase 特征：

$$
F^P_{i,j}
$$

3. 对其做数值压缩：

$$
F^P_{i,j}\leftarrow \log(F^P_{i,j}+1)
$$

4. 中心点 $v_0=u$ 的 phase 特征退化为 $p'(\omega,l)$   

---

## 6. 边界处理

### 6.1 density 边界处理

职责：保证 stencil 落到体外时的密度定义一致。

规则：

* 体边界外 density 视为 0 

---

### 6.2 transmittance 边界处理

职责：避免体外采样点因硬件 clamping 导致错误 transmittance 值。

规则：

1. 若 stencil 点落到体外
2. 不直接使用 clamping
3. 沿光照方向将该点投影到体边界交点
4. 在投影点处采样 transmittance  

---

## 7. 组装单层 descriptor

职责：把同一层 stencil 上的三类局部特征组织成单层输入。

流程：

1. 对每个 stencil 点构造三元组：

$$
F_{i,j}=\{F^\mu_{i,j},F^S_{i,j},F^P_{i,j}\}
$$

2. 将第 $i$ 层所有点组成单层 descriptor：

$$
\Sigma_i=\{F_{i,1},F_{i,2},\dots,F_{i,N_i}\}
$$

---

## 8. 组装最终 descriptor

职责：形成网络的完整输入。

流程：

1. 按层次顺序收集全部单层 descriptor：

$$
\Sigma_1,\Sigma_2,\dots,\Sigma_K
$$

2. 计算全局参数：

   * $G$
   * $\varsigma^\alpha$
   * $\gamma=\cos^{-1}(\omega\cdot l)$

3. 组合成最终 descriptor：

$$
z=\{\Sigma_1,\Sigma_2,\dots,\Sigma_K,G,\varsigma^\alpha,\gamma\}
$$

---

## 9. 最终结果

最终得到的 descriptor 由两部分组成：

### 9.1 局部多尺度特征

* density
* scaled-transmittance
* phase

它们在 12 层 frequency-sensitive stencil 上采样获得。 

### 9.2 全局着色参数

* $G$
* $\varsigma^\alpha$
* $\gamma$

它们直接附加到 descriptor 末尾。

[1]: https://dl.acm.org/doi/10.1145/3588432.3591493?utm_source=chatgpt.com "Deep Real-time Volumetric Rendering Using Multi-feature ..."