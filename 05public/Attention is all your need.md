---
title: Attention Is All You Need 精读（原文对照）
description: 逐段精读 Transformer 原始论文,每段配英文原文、中文译文和我的感悟,从为什么抛弃 RNN 循环,讲到多头注意力、位置编码和训练细节
pubDate: 2026-10-07
tags:
  - LLM
  - NLP
  - Transformer
  - 论文精读
---

## 写在前面

这篇是我精读 2017 年那篇《Attention Is All You Need》的笔记,论文作者是 Google Brain / Google Research 的 Vaswani 等人。它提出的 **Transformer** 是后来 BERT、GPT 乃至今天所有大语言模型的共同骨架.

读法很简单:我把论文正文按自然段拆开,每一段都给三样东西——

1. **原文**:论文的英文原句,用引用块呈现;
2. **译文**:我自己翻的中文,追求意思准确、说人话,不逐词硬译;
3. **我的感悟**:这一段在讲什么、为什么重要、背后的工程直觉是什么.


## Abstract(摘要)

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

**译文**

当前主流的序列转换(sequence transduction)模型,都建立在包含编码器和解码器的复杂循环神经网络或卷积神经网络之上.表现最好的模型还会通过注意力机制(attention mechanism)把编码器和解码器连接起来.我们提出一种全新的、简单的网络架构——Transformer,它完全基于注意力机制,彻底抛弃了循环(recurrence)和卷积(convolution).在两个机器翻译任务上的实验表明,这类模型质量更优,同时并行性更好,训练所需时间也大幅缩短.我们的模型在 WMT 2014 英德翻译任务上取得了 28.4 的 BLEU 值,比此前最好的结果(包括集成模型)还高出 2 个 BLEU 以上.在 WMT 2014 英法翻译任务上,模型在 8 块 GPU 上训练 3.5 天后,创下了 41.8 这一全新的单模型 BLEU 纪录,而训练成本只占文献中最佳模型的一小部分.我们还把 Transformer 成功应用于英语成分句法分析(English constituency parsing),无论训练数据充足还是有限都表现良好,证明它能很好地泛化到其他任务.

**我的感悟**

摘要是整篇论文的压缩包,我读下来它其实只喊了三句话:

- **架构简单到激进**:`based solely on attention`(只用注意力),`dispensing with recurrence and convolutions entirely`(把循环和卷积整个扔掉).要知道在 2017 年,RNN 系就是序列建模的"政治正确",这句话等于当着全村人的面说旧房子不用要了.
- **又快又好**:并行度高、训练时间短,同时 BLEU 还更高.通常论文只能在"效果"和"成本"里占一头,它两头都占,这才是最难的部分.
- **不只是翻译专用**:最后特意补了句法分析任务,是为了证明 Transformer 是个通用架构,而不是为翻译任务调出来的偏方.

另外留意两个数字的口径:摘要里的 **3.5 天 / 8 GPU 是 big 模型**;正文里还会出现一个 **12 小时 / 8 GPU,那是 base 模型**,别搞混.

## 1 Introduction(引言)

### 第 1 段:RNN 家族是当时的既定主流

> Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

**译文**

循环神经网络,尤其是长短期记忆网络(LSTM)[13] 和门控循环神经网络(GRU)[7],已经在语言建模、机器翻译等序列建模与序列转换问题上被牢固地确立为最先进(state of the art)的方法 [35, 2, 5].此后仍有大量工作在持续拓展循环语言模型和编码器-解码器架构的边界 [38, 24, 15].

**我的感悟**

这是典型的"先立靶子":在提出新架构之前,先承认旧王朝的统治地位.注意作者点了三个名字——普通 RNN、**LSTM**、**GRU**,意思是"我说的是整个循环家族,不是某一个具体模型".这样后面说"循环不行了"的时候,就没有人能用"那你怎么不试试 LSTM?"来反驳.

写论文这是很聪明的笔法:把对手的范围划到最大,再一次性超越.

### 第 2 段:循环模型的原罪——顺序计算

> Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

**译文**

循环模型通常沿着输入和输出序列的符号位置来组织计算.它把序列位置和计算时间上的步骤一一对应,生成一系列隐藏状态 $h_t$,而 $h_t$ 是前一个隐藏状态 $h_{t-1}$ 和位置 $t$ 处输入的函数.这种**内在的顺序特性**使得训练样本内部无法并行;当序列变长时问题尤为严重,因为内存限制又制约了跨样本的批处理.近期的工作通过分解技巧(factorization tricks)[21] 和条件计算(conditional computation)[32] 在计算效率上取得了显著提升,后者还顺带改善了模型性能.然而,**顺序计算这一根本性约束依然存在**.

**我的感悟**

这一段是整篇论文的"问题陈述",值得逐句嚼:

- RNN 的核心递推是 $h_t = f(h_{t-1}, x_t)$.第 $t$ 步必须等第 $t-1$ 步算完,这不是代码写得烂,而是**写在数学定义里的依赖关系**.GPU 最擅长的"一次铺开几千个核心同时算"在这里完全用不上.
- 作者还区分了两个层面的并行:样本**内部**无法并行(一句话里的词必须按顺序算),样本**之间**的批处理又受显存限制(长句子太吃显存,batch 开不大).两头被堵死.
- 最狠的是最后一句:别人做了那么多工程优化(factorization、条件计算),`the fundamental constraint ... remains`——**根子上的问题没动**.这句话等于宣判:在 RNN 框架内做优化,天花板已经看见了.

我自己的体会是,判断一个架构缺陷是"工程问题"还是"基因问题"特别关键.能靠加机器、改实现解决的是工程问题;定义里自带的、怎么优化都绕不开的,是基因问题,只能换架构.这篇论文选的就是后者.

### 第 3 段:注意力机制已经很好,但一直被绑在 RNN 上

> Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

**译文**

在各类任务中,注意力机制已经成为有竞争力的序列建模与转换模型不可或缺的一部分,它允许模型对依赖关系进行建模,而无需顾忌这些依赖在输入或输出序列中相隔多远 [2, 19].然而,除了少数例外 [27],这类注意力机制都是与循环网络结合使用的.

**我的感悟**

这段澄清了一个常见误解:**注意力不是这篇论文发明的**.Bahdanau 等人 2014 年就把 attention 用在机器翻译上了 [2],它解决的正是 RNN 的老毛病——不管两个词隔多远,都能直接"看"过去.

但当时的用法是"RNN 打底,attention 打补丁":编码器还是一堆 LSTM,只是在解码器读编码器输出时加一层注意力.作者在这里敏锐地指出了一个不合逻辑的地方:既然 attention 自己就能无视距离地建立依赖,**那为什么还要背着 RNN 这个顺序计算的包袱?**

这就是创新里常见的一种模式:不是发明新零件,而是发现旧零件之间的主从关系搞反了——把"配角"扶正,把"主角"请下台.

### 第 4 段:正式提出 Transformer

> In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

**译文**

在本工作中,我们提出 Transformer:这是一种避开循环、完全依靠注意力机制来刻画输入与输出之间全局依赖关系的模型架构.Transformer 支持高得多的并行度,只需在 8 块 P100 GPU 上训练短短 12 小时,就能在翻译质量上达到新的最先进水平.

**我的感悟**

引言收尾,亮明答案.这里给了三个关键词:

- `eschewing recurrence`(避开循环):明确说不做什么;
- `relying entirely on attention`(完全依靠注意力):明确说靠什么;
- `global dependencies`(全局依赖):注意力一步就能连接任意两个位置,这是后面反复强调的优势.

注意这里的 **12 小时是 base 模型**的数字,和摘要里 big 模型的 3.5 天是两个配置.论文这种"用最短训练时间刷出 SOTA"的写法,在当时是很有冲击力的——别人动辄训练几周,它半天就超过去了.

我觉得引言这四段的推进节奏很值得学:旧主流很强 → 但它有基因缺陷 → 补丁(attention)其实比本体更能打 → 那干脆让补丁当主角.环环相扣,读完你已经被说服一半了.

## 2 Background(背景)

### 第 1 段:用 CNN 取代循环的尝试,以及它们的路径长度问题

> The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

**译文**

减少顺序计算这一目标,同样构成了 Extended Neural GPU [16]、ByteNet [18] 和 ConvS2S [9] 的基础.它们都用卷积神经网络作为基本构件,为所有输入和输出位置并行计算隐藏表示.但在这些模型中,要把任意两个输入或输出位置的信号关联起来,所需的操作数会随位置之间的距离而增长——ConvS2S 是线性增长,ByteNet 是对数增长.这使得学习远距离位置之间的依赖变得更加困难 [12].而在 Transformer 中,这一开销被降低到**常数级操作数**;不过代价是,对注意力加权位置做平均会降低有效分辨率,我们用第 3.2 节描述的**多头注意力(Multi-Head Attention)**来抵消这一影响.

**我的感悟**

这一段埋了全文最重要的一条暗线:**信号在两个位置之间要走多少步(路径长度)**.

- CNN 为什么并行?因为一个卷积核在所有位置上可以同时扫,没有时间步依赖.
- 但卷积是**局部**操作:一个核宽为 $k$ 的卷积层,每个位置只能看到左右 $k$ 个邻居.想让句首和句尾"见上面",就得堆很多层——ConvS2S 这种普通卷积要堆 $O(n)$ 层(线性),ByteNet 用了膨胀卷积(dilated convolution),感受野指数扩张,只要 $O(\log n)$ 层.
- Transformer 自注意力更狠:**一层之内,任意两个位置直接算关联**,路径长度是 $O(1)$.

但作者诚实地交代了代价:注意力的输出是对所有位置做**加权平均**,单次平均会把信息"抹平",分辨率下降.这是一个真实的 tradeoff,而解药就是**多头注意力**——让多组注意力各自关注不同的东西,再拼回来.这里先抛名词,3.2 节再展开,前后是扣着的.

### 第 2 段:什么是自注意力

> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

**译文**

自注意力(self-attention),有时也叫内部注意力(intra-attention),是一种把**单个序列内部不同位置关联起来**,从而计算该序列表示的注意力机制.自注意力已经成功应用于多种任务,包括阅读理解、抽象摘要、文本蕴含,以及学习与具体任务无关的句子表示 [4, 27, 28, 22].

**我的感悟**

这里要分清两种注意力:

- **普通注意力(encoder-decoder attention)**:query 来自一个序列(解码器),key/value 来自另一个序列(编码器),是**两个序列之间**的媒婆;
- **自注意力(self-attention)**:query、key、value **全部来自同一个序列**,是序列照镜子,让句子里的每个词和同句的其他词挨个发生关系.

为什么自注意力有用?举个最直观的例子,句子 "The animal didn't cross the street because it was too tired" 里,模型想理解 `it` 指的是 animal 而不是 street,就必须让 `it` 这个位置"回头看"句子里的其他词.自注意力干的就是这件事,而且一层就能干成.作者列的阅读理解、指代消解、摘要这些任务,本质上都需要这种句内回看.

### 第 3 段:记忆网络的另一条路线

> End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

**译文**

端到端记忆网络(End-to-end memory networks)建立在一种**循环式的注意力机制**之上,而不是沿序列对齐的循环,并已被证明在简单语言问答和语言建模任务上表现良好 [34].

**我的感悟**

这是在做 related work 的"站位":作者要说明"去掉序列对齐的循环"这个思路并非凭空出现,记忆网络 [34] 已经用多轮注意力(在记忆上反复 hop)替代了 RNN 式递推.但记忆网络当时主要在问答这种"查记忆"的场景里有效,没有成为通用的序列转换架构.

论文里提这么一句的作用是:既承认思想源头(不显得自己横空出世、目中无人),又划清界限(它们和我做的不是一回事).这是 related work 的标准分寸感.

### 第 4 段:Transformer 的"第一"

> To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

**译文**

然而据我们所知,Transformer 是**第一个完全依靠自注意力**来计算输入和输出表示、而不使用沿序列对齐的 RNN 或卷积的序列转换模型.在接下来的几节中,我们将介绍 Transformer,说明自注意力的动机,并讨论它相对于 [17, 18]、[9] 等模型的优势.

**我的感悟**

`to the best of our knowledge`(据我们所知)是论文里 claim 优先权的标准措辞——严谨地把"第一个"限定在作者检索能力所及的范围内.这个"第一"的完整表述值得记住:

- **完全依靠**自注意力(不是当补丁,而是当唯一的依赖建模手段);
- 输入和输出表示都靠它;
- **既不用 RNN,也不用卷积**.

三个限定词叠在一起,才圈出真正的新意所在.后面三节就是按"怎么搭(第 3 节)→ 为什么好(第 4 节)→ 怎么训(第 5 节)"来兑现这句话的.

## 3 Model Architecture(模型架构)

### 第 1 段:编码器-解码器的总体范式

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

**译文**

大多数有竞争力的神经序列转换模型都采用编码器-解码器结构 [5, 2, 35].其中,编码器把一个由符号表示组成的输入序列 $(x_1, \dots, x_n)$ 映射为一个连续表示序列 $\mathbf{z} = (z_1, \dots, z_n)$.给定 $\mathbf{z}$,解码器再逐个元素地生成输出符号序列 $(y_1, \dots, y_m)$.在每一步,模型都是**自回归(auto-regressive)**的 [10]:生成下一个符号时,会把此前已经生成的符号当作额外输入一起消费.

**我的感悟**

这一段是整个架构的世界观,先把符号约定立起来:

| 符号                               | 含义                                 |
| -------------------------------- | ---------------------------------- |
| $(x_1, \dots, x_n)$              | 输入序列,$n$ 是输入长度                     |
| $\mathbf{z} = (z_1, \dots, z_n)$ | 编码器输出的连续表示序列                       |
| $(y_1, \dots, y_m)$              | 输出序列,$m$ 是输出长度(翻译时 $m$ 和 $n$ 不必相等) |

两个关键概念:

- **编码器负责"理解"**:把离散符号变成一串连续向量;**解码器负责"生成"**:拿着理解结果往外吐目标序列.
- **自回归(auto-regressive)**:生成第 $t$ 个词时,要把前面 $t-1$ 个已生成的词也喂回去当输入.像挤牙膏,一次一个,不能倒着来.这正是 GPT 类模型的生成方式,也是为什么解码器必须有 mask(马上讲到).

### 第 2 段:Transformer 遵循该范式,但换了内部构件

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

**译文**

Transformer 遵循这一总体架构,但在编码器和解码器中都改用**堆叠的自注意力层和逐点(point-wise)全连接层**,分别如图 1 的左半部分和右半部分所示.

**我的感悟**

这句话点明:Transformer 革命的不是"编码器-解码器"这个顶层范式(它保留了),而是**内部的层**——把 RNN/CNN 换成了自注意力 + 全连接.

看论文的 Figure 1 时,我建议按这个顺序读:左下角 Inputs → Embedding → 加 Positional Encoding → 进左侧 $N\times$ 堆叠的编码器 → 编码器输出送到右侧解码器每层的中间那个注意力块 → 解码器往上走 → Linear → Softmax → Output Probabilities.左编码、右解码,中间靠注意力桥接,整张图就是这篇论文的全部.

### 3.1 Encoder and Decoder Stacks(编码器与解码器堆栈)

#### Encoder(编码器)

> The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$, where $\text{Sublayer}(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}} = 512$.

**译文**

编码器由 $N = 6$ 个相同层堆叠而成.每层有两个子层:第一个是**多头自注意力机制**,第二个是一个简单的、逐位置(position-wise)的全连接前馈网络.我们在每个子层外面都包了一个**残差连接(residual connection)**[11],再接**层归一化(layer normalization)**[1].也就是说,每个子层的输出是:

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

其中 $\text{Sublayer}(x)$ 是子层自身实现的函数.为了让这些残差连接能够相加,模型中所有子层(包括嵌入层)的输出维度都统一为 $d_{\text{model}} = 512$.

**我的感悟**

编码器单层的结构可以画成:

```text
输入 x
  │
  ├──────────────┐
  ▼              │
Multi-Head       │  (残差捷径)
Self-Attention   │
  └──► + ◄───────┘
       │
    LayerNorm
       │
  ├──────────────┐
  ▼              │
Feed-Forward     │
  └──► + ◄───────┘
       │
    LayerNorm
       ▼
输出(维度仍是 512)
```

几个设计直觉:

- **为什么用残差?** 残差连接让信号(以及梯度)有一条"高速公路"可以直接绕过子层.网络堆到 6 层、几十层时,没有残差很容易梯度消失、训不动.这是从 ResNet [11] 借来的成熟经验.
- **为什么所有子层都强行统一成 512 维?** 因为残差要做 $x + \text{Sublayer}(x)$ 逐元素相加,维度不一致就加不了.统一维度是残差结构的硬性前提,记住 $d_{\text{model}} = 512$ 这个数字,后面处处用它.
- **LayerNorm 放在残差相加之后**,对每个位置的 512 维特征做归一化,稳住每层输出的尺度.

#### Decoder(解码器)

> The decoder is also composed of a stack of $N = 6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

**译文**

解码器同样由 $N = 6$ 个相同层堆叠而成.除了编码器每层中都有的那两个子层外,解码器**插入了第三个子层**,它对编码器堆栈的输出执行多头注意力.与编码器类似,我们在每个子层外都使用残差连接,并接层归一化.我们还修改了解码器堆栈中的自注意力子层,**防止某个位置注意到它之后的位置**.这种掩码(masking),再加上输出嵌入整体偏移一个位置的事实,保证了对位置 $i$ 的预测只能依赖于位置 $i$ 之前的已知输出.

**我的感悟**

解码器每层比编码器多一个子层,一共三个,顺序是:

```text
① Masked Multi-Head Self-Attention  (看自己已生成的部分,且不许偷看未来)
② Multi-Head Encoder-Decoder Attention  (Q 来自①,K/V 来自编码器输出 z)
③ Position-wise Feed-Forward
每个子层外都包 残差 + LayerNorm
```

两个必须吃透的点:

- **第②个子层是编码器和解码器的唯一接口**.query 来自解码器上一层,key 和 value 来自编码器的最终输出 $\mathbf{z}$.这样解码器每个位置都能"回看"输入句子的所有位置——这就是传统意义上的 attention.
- **第①个子层为什么要 mask?** 训练时为了效率,目标句子是整个喂进去的(teacher forcing),如果不挡,位置 $i$ 的自注意力一眼就能看到"标准答案"里 $i+1, i+2$ 的词,等于考试提前看到答案,模型什么都学不到.所以要把"未来位置"全部遮掉,强制位置 $i$ 只能看 $< i$ 的位置,这叫**因果掩码(causal mask)**.

具体怎么实现,3.2.3 节会说:在 softmax 之前,把非法连接对应的分数设为 $-\infty$,softmax 之后这些位置的权重就变成 0,相当于完全看不见.

注:训练时整句并行 + mask,推理时却只能一个词一个词地自回归生成——这也是为什么 Transformer "训练并行、推理仍串行",今天各种推理加速(vLLM、KV cache 等)很大程度上都在补这个短板.

### 3.2 Attention(注意力)

#### 引导段:注意力的统一表述——Query、Key、Value

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

**译文**

注意力函数可以描述为:把一个**查询(query)**和一组**键-值(key-value)对**映射为一个**输出(output)**;query、key、value 和输出都是向量.输出被计算为所有 value 的**加权和**,而分配给每个 value 的权重,由 query 与对应 key 之间的**相容性函数(compatibility function)**计算得到.

**我的感悟**

这是全文最该背下来的一段话,它把所有注意力机制统一成了**数据库检索**的语言:

- **Query(Q)**:我当前想找什么(带着问题来);
- **Key(K)**:每个候选条目门上贴的标签;
- **Value(V)**:每个候选条目真正的内容;
- 流程:拿 Q 去和每个 K 比相似度 → 相似度经 softmax 归一化成权重 → 用这些权重对 V 加权求和 → 得到输出.

一句话:**用 Q 和 K 的匹配度,决定从 V 里各取多少**.这个 QKV 框架不是 Transformer 发明的,但它把这套语言固定了下来,今天读任何 attention 的变种代码,先认 Q、K、V 从哪来,就不会乱.

#### 3.2.1 Scaled Dot-Product Attention(缩放点积注意力)

##### 第 1 段:算法定义

> We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

**译文**

我们把自己这种特定的注意力称为"**缩放点积注意力(Scaled Dot-Product Attention)**"(图 2).输入由维度为 $d_k$ 的 query 和 key、以及维度为 $d_v$ 的 value 组成.我们计算 query 与所有 key 的点积,把每个点积除以 $\sqrt{d_k}$,再施加 softmax 函数,得到作用在 value 上的权重.

**我的感悟**

三步操作,对应 Figure 2 左图从下往上的流程:MatMul(Q 和 K 做点积)→ Scale(除以 $\sqrt{d_k}$)→ Mask(可选)→ SoftMax(得到权重)→ 再和 V 做一次 MatMul(加权求和).

"相容性函数"在这里的具体选择就是**点积**:两个向量方向越一致,点积越大,认为越相关.所谓"缩放",就是点积结果先除以 $\sqrt{d_k}$ 再进 softmax——这个除法是整个小节的题眼,第 4 段会解释为什么非除不可.

##### 第 2 段:矩阵形式与核心公式

> In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:
>
> $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{1}$

**译文**

实践中,我们把一组 query 打包成矩阵 $Q$,同时对整组 query 计算注意力;key 和 value 也分别打包成矩阵 $K$ 和 $V$.输出矩阵按下式计算:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{1}
$$

**我的感悟**

这是 Transformer 的第一号公式,我把形状摊开(以自注意力为例,序列长度 $n$):

| 张量 | 形状 | 含义 |
|---|---|---|
| $Q$ | $n \times d_k$ | $n$ 个 query,每个 $d_k$ 维 |
| $K$ | $n \times d_k$ | $n$ 个 key |
| $V$ | $n \times d_v$ | $n$ 个 value |
| $QK^T$ | $n \times n$ | 两两位置的点积,即"相关度矩阵" |
| $\text{softmax}(QK^T/\sqrt{d_k})$ | $n \times n$ | 每行是一个 query 对所有位置的注意力权重,行和为 1 |
| 输出 | $n \times d_v$ | 加权求和后的新表示 |

关键直觉:**那个 $n \times n$ 矩阵里,第 $i$ 行第 $j$ 列就是"第 $i$ 个词对第 $j$ 个词的关注程度"**.所有位置两两交互,一次矩阵乘就全部算完,没有任何时间步依赖——这就是它能完全并行、路径长度为 $O(1)$ 的根本原因.而整个公式全是矩阵乘,正好喂给 GPU / TPU 上高度优化的 GEMM,这是它快的工程根源.

注:自注意力里 $Q、K、V$ 来自同一序列;encoder-decoder attention 里 $Q$ 来自解码器、$K、V$ 来自编码器,公式完全一样,只是输入来源不同.

##### 第 3 段:点积注意力 vs 加性注意力

> The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

**译文**

最常用的两种注意力函数是**加性注意力(additive attention)**[2] 和**点积(乘性)注意力(dot-product attention)**.点积注意力和我们的算法完全相同,只是少了 $\frac{1}{\sqrt{d_k}}$ 这个缩放因子.加性注意力则用一个带单隐藏层的前馈网络来计算相容性函数.虽然两者理论复杂度相近,但点积注意力在实践中要快得多、也省空间得多,因为它可以用高度优化的矩阵乘法代码实现.

**我的感悟**

两种主流打分方式的对比:

- **加性注意力(Bahdanau 风格)**:相似度靠一个小神经网络算出来,$\text{score} = v^T \tanh(W_1 q + W_2 k)$,表达能力强,但没法整体变成矩阵乘,慢;
- **点积注意力**:相似度直接 $q \cdot k$,就是矩阵乘,GPU 上快一个量级.

作者选点积,本质是**向硬件低头、向工程要速度**的决定:理论上两者差不多,但在真实 GPU 上,矩阵乘有 cuBLAS 这类底层库和专用张量核心加持,差距是数量级的.做深度学习架构,理论复杂度之外,"能不能映射成矩阵乘"往往才是生死线,这个思想贯穿全文.

##### 第 4 段:为什么必须除以 √d_k

> While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$ [3]. We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$, has mean $0$ and variance $d_k$. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

**译文**

当 $d_k$ 较小时,两种机制表现相近;但当 $d_k$ 较大时,**不做缩放的点积注意力会差于加性注意力** [3].我们推测,当 $d_k$ 很大时,点积的数值幅度会变得很大,把 softmax 推进梯度极小的区域.为说明点积为什么会变大,假设 $q$ 和 $k$ 的各分量是相互独立的随机变量,均值为 $0$、方差为 $1$.那么它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 $0$、方差为 $d_k$.为抵消这一效应,我们把点积乘以 $\frac{1}{\sqrt{d_k}}$ 进行缩放.

**我的感悟**

这是论文里我最喜欢的一段,因为它把一个看似拍脑袋的常数 $\sqrt{d_k}$,用一行概率论严格推导出来了:

- 每个 $q_i k_i$ 均值 0、方差 1;$d_k$ 项相加,方差累加为 $d_k$,所以点积 $q \cdot k$ 的**标准差是 $\sqrt{d_k}$**.
- $d_k = 512$ 时,点积的典型幅度会到 $\sqrt{512} \approx 22.6$.这么大的数喂进 softmax 会发生什么?$\text{softmax}$ 对最大的那个 logit 会给出接近 1 的概率,其余全接近 0——**分布变得 one-hot、极度尖锐**.
- 此时 softmax 进入饱和区,反向传播的梯度趋近于 0,模型要么学不动,要么直接"winner-take-all",只盯着一个词,丧失了注意力本来该有的软性分配能力.

除以 $\sqrt{d_k}$ 之后,点积的方差重新被拉回 1,无论 $d_k$ 多大,输入 softmax 的数值尺度都保持稳定.这就是**缩放(Scaled)**二字的全部道理.

注:这也是一个通用工程经验——凡是要进 softmax / sigmoid 的分数,都要留意它的尺度会不会随维度、随层数膨胀.今天大模型里常见的 RMSNorm、QK-Norm(对 Q、K 做归一化)等改进,本质上都在继续解决同一个"注意力分数尺度失控"的问题.

#### 3.2.2 Multi-Head Attention(多头注意力)

##### 第 1 段:把注意力拆成多个"头"并行

> Instead of performing a single attention function with $d_{\text{model}}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

**译文**

我们发现,与其用 $d_{\text{model}}$ 维的 key、value、query 只做一次注意力,不如让 query、key、value 分别经过 $h$ 次**不同的、可学习的线性投影**,投到 $d_k$、$d_k$、$d_v$ 维;然后在每一组投影后的 query、key、value 上**并行**执行注意力函数,得到 $d_v$ 维的输出;最后把这些输出拼接(concatenate)起来,再做一次线性投影,得到最终结果,如图 2 所示.

**我的感悟**

单头注意力是"一个人一次性看全句";多头是"派 $h$ 个专家各戴一副不同的眼镜去看,再把看到的东西汇总".流程是:

```text
          Q          K          V
          │          │          │
     h 组线性投影 W_i^Q, W_i^K, W_i^V  (每组参数不同)
          │          │          │
   head_1 ... head_h  各自并行做 Scaled Dot-Product Attention
          └────┬─────┘
            Concat(拼接)
               │
            W^O 线性投影
               ▼
            最终输出
```

注意每一头的投影矩阵是**各自学出来的、不共享**,这保证了不同头真的能学成不同的样子,而不是 $h$ 个一模一样的副本.

##### 第 2 段:多头的意义——不同的表示子空间

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

**译文**

多头注意力允许模型在不同位置**联合关注来自不同表示子空间(representation subspaces)的信息**.而只用单个注意力头时,加权平均会抑制这种能力.

**我的感悟**

这句话回答了"为什么不能一次算个大的,非要拆成 8 个小的":

- 单头:一个 512 维的加权平均,所有类型的关系(语法关系、指代关系、词义搭配……)都被压进同一组权重里互相争夺,最后平均成一个"四不像";
- 多头:每个头只在一个 64 维的**子空间**里运作,可以一个头专门盯句法主谓、一个头专门消解指代 `it`、一个头专门找相邻词搭配,互不干扰,最后拼起来.

论文附录的可视化(Figure 3-5)后来确实发现:不同头学到了明显不同的行为,有的头专攻长距离依赖,有的头做指代消解,有的头注意力模式直接对应句子的句法结构.这不是事后附会,是多头设计被实证支持.

##### 第 3 段:多头的公式与参数

> Where the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

**译文**

其中投影操作的参数矩阵为 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$,以及 $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

**我的感悟**

把第 1 段的文字翻译成公式,就是第二号公式组:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

形状走一遍:输入 $Q,K,V$ 都是 $n \times 512$;$Q W_i^Q$ 变成 $n \times d_k$;每头输出 $n \times d_v$;$h$ 个头沿特征维拼接得到 $n \times (h d_v)$;乘 $W^O\;(h d_v \times d_{\text{model}})$ 回到 $n \times d_{\text{model}}$.**进是 512 维,出还是 512 维**,残差连接又能成立了.

##### 第 4 段:头数与维度的配置

> In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{\text{model}}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

**译文**

本工作使用 $h = 8$ 个并行的注意力层(即头).每个头取 $d_k = d_v = d_{\text{model}}/h = 64$.由于每个头的维度降低了,**总的计算开销与单头、满维度的注意力大致相当**.

**我的感悟**

这个配置体现了一个极漂亮的"免费午餐":

- 单头满维:投影 + 一次 $d = 512$ 的注意力;
- 8 头:每头只在 $64$ 维上算,8 个头的点积计算量是 $8 \times 64^2$,和单头 $512^2$ 相比——因为点积开销随维度平方、随头数线性,所以 $h \times (d/h)^2 = d^2/h$,实际上**多头还略便宜**;
- 代价只是多存几组投影矩阵,换来的却是 8 个独立子空间的表达能力.

记住这组经典超参:**$h = 8$,$d_k = d_v = 64$,$8 \times 64 = 512 = d_{\text{model}}$**.后来的模型头数越用越多(big 模型 16 头、$d_{\text{model}} = 1024$,仍是每头 64 维),可以发现"**每头 64 维左右**"是个长期稳定的经验值.

#### 3.2.3 Applications of Attention in our Model(注意力在模型中的三种用法)

> The Transformer uses multi-head attention in three different ways:
>
> - In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].
> - The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
> - Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

**译文**

Transformer 在三个不同的地方使用多头注意力:

- **编码器-解码器注意力层**:query 来自解码器的上一层,而充当记忆的 key 和 value 来自编码器的输出.这让解码器的每个位置都能注意到输入序列的**所有位置**,复刻了 [38, 2, 9] 等序列到序列模型中典型的编码器-解码器注意力机制.
- **编码器中的自注意力层**:所有 key、value、query 都来自同一个地方,即编码器上一层的输出.编码器的每个位置都能注意到编码器上一层的所有位置.
- **解码器中的自注意力层**:类似地,它允许解码器的每个位置注意到解码器中**截止到该位置(含自身)**的所有位置.为保持自回归特性,我们必须阻止解码器中信息"向左"(向未来)流动.实现方式是在缩放点积注意力内部,把 softmax 输入中所有对应非法连接的值**掩码掉(设为 $-\infty$)**,见图 2.

**我的感悟**

这一段是对全模型注意力的总账,我把它压成一张表,以后看架构图对照即可:

| 位置 | Q 来自 | K、V 来自 | 能否看未来 |
|---|---|---|---|
| 编码器自注意力 | 编码器上一层 | 编码器上一层(同源) | 全可见,无 mask |
| 解码器自注意力 | 解码器上一层 | 解码器上一层(同源) | **因果 mask,只能看 ≤ 当前位置** |
| 编码器-解码器注意力 | 解码器上一层 | **编码器最终输出** | 编码器输出全可见 |

关于 mask 的实现细节再补一刀:在 softmax 之前把非法位置的分数设为 $-\infty$,因为 $e^{-\infty} = 0$,softmax 归一化后这些位置的注意力权重严格为 0,梯度也传不过去,信息泄漏被彻底封死.这就是代码里常见的一个下三角可见矩阵(下三角为 1、上三角为 0).

我自己的记忆口诀是:**编码器双向随便看,解码器只能回头看,解码中间还要看编码器**.后来 BERT 只用了编码器(双向语言模型),GPT 只用了解码器(单向自回归),T5 用完整的编码器-解码器——追根溯源,全是这张表里三种注意力的不同组合.

### 3.3 Position-wise Feed-Forward Networks(逐位置前馈网络)

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
>
> $\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2 \tag{2}$
>
> While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{\text{model}} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$.

**译文**

除了注意力子层,编码器和解码器的每一层还包含一个全连接前馈网络,它**分别地、且以相同方式作用于每一个位置**.它由两个线性变换组成,中间夹一个 ReLU 激活:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2 \tag{2}
$$

同一层内,不同位置用的是**同一套**线性变换;但不同层之间参数各不相同.另一种等价描述是:它就是两个核大小为 1 的卷积(1×1 convolution).输入和输出的维度都是 $d_{\text{model}} = 512$,中间隐藏层维度为 $d_{ff} = 2048$.

**我的感悟**

这一层要和注意力层对照着理解,两者分工明确:

- **注意力层:跨位置**做信息混合,回答"我该从句中哪些词那里取信息";
- **FFN 层:逐位置**做非线性变换,不与其他位置交互,回答"取完信息后,我该怎么加工我自己".

形状上它是 $512 \to 2048 \to 512$:先把维度扩到 **4 倍**,在高维空间里过 ReLU 做非线性筛选,再压回 512 维.这个"先扩 4 倍、再压回"的瓶颈结构,和 1×1 卷积数学上完全等价(对每个位置独立地做两次全连接),也和后来所有 Transformer 变体的 FFN 几乎一模一样,4 倍扩展率是沿用至今的默认值.

为什么需要它?我自己的理解是:注意力本质是加权平均,是**线性**操作(权重由 QK 决定,但对 V 的组合是线性的),光靠它表达能力不够;FFN 提供了必需的**逐元素非线性**,让每个位置能对注意力送来的信息做真正复杂的变换.可以说 **attention 负责"通信",FFN 负责"计算"**,二者交替,一层都不能少.

注:今天大模型里常见的 SwiGLU、GeGLU 等 FFN 变体,主要是把 ReLU 换成门控型激活(如 SiLU/SwiGLU),扩容再压缩的骨架和 4 倍左右的扩展思想并没有变.

### 3.4 Embeddings and Softmax(嵌入与 Softmax)

> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\text{model}}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.

**译文**

与其他序列转换模型类似,我们用学习得到的嵌入(embeddings)把输入 token 和输出 token 转换成维度为 $d_{\text{model}}$ 的向量.我们也用常见的、可学习的线性变换加 softmax 函数,把解码器输出转换成对下一个 token 的预测概率.在我们的模型中,**两个嵌入层以及 softmax 前的线性变换共享同一个权重矩阵**,这一点与 [30] 类似.在嵌入层中,我们把这些权重乘以 $\sqrt{d_{\text{model}}}$.

**我的感悟**

两个工程细节:

- **权重共享(weight tying)**:正常来说,输入词嵌入矩阵、输出词嵌入矩阵、输出端投影到词表的 Linear 矩阵是三个矩阵;这里把它们绑成同一个.直觉上,"一个词作为输入时的含义"和"作为输出被预测时的含义"本就高度相关,共享既合理又能省下大量参数(词表 3 万多 × 512,是上千万量级的参数).
- **embedding 乘 $\sqrt{d_{\text{model}}}$**:词嵌入通常初始化得很小(量级接近 1),而紧接着要和位置编码相加.位置编码正弦函数的量级约在 [-1, 1],若不放大 embedding,位置编码会"喧宾夺主",语义信息被位置信息淹没.乘以 $\sqrt{512} \approx 22.6$ 把 embedding 拉到和位置编码相加时合理的相对尺度.

注:这个 $\sqrt{d_{\text{model}}}$ 缩放和 3.2.1 里除以 $\sqrt{d_k}$ 是两个不同地方的两个相反方向操作,别记串——一个是为了稳住 softmax 前的分数,一个是为了平衡 embedding 与位置编码相加时的尺度.

### 3.5 Positional Encoding(位置编码)

#### 第 1 段:为什么必须显式注入位置

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{\text{model}}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

**译文**

由于我们的模型既没有循环也没有卷积,为了让模型利用序列的顺序,**必须主动注入**一些关于 token 在序列中相对位置或绝对位置的信息.为此,我们在编码器和解码器堆栈底部的输入嵌入上加上"位置编码(positional encodings)".位置编码与嵌入维度相同,都是 $d_{\text{model}}$,这样两者才能相加.位置编码有许多种选择,包括学习式的和固定的 [9].

**我的感悟**

这是抛弃循环/卷积后必须付的代价,也是最容易被初学者忽略的一点:**纯自注意力对位置是完全盲的**.

把一句话的词顺序彻底打乱再喂进去,自注意力算出的输出只是跟着位置重排而已——它知道哪些词彼此相关,却**不知道谁在前、谁在后**."狗咬人"和"人咬狗"在它眼里词袋相同.因为注意力是集合上的操作,对输入排列等变(permutation equivariant),没有任何顺序概念.

解决办法:给每个位置的嵌入向量加上一个"只跟位置有关"的向量,而且维度相同、直接**相加**(不是拼接,拼接会改变维度、破坏后续统一结构).从 Figure 1 底部那个 $\oplus$ 符号开始,位置信息就一路跟着语义信息往上走了.

#### 第 2 段:正弦位置编码

> In this work, we use sine and cosine functions of different frequencies:
>
> $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$
>
> $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$
>
> where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

**译文**

本工作使用不同频率的正弦、余弦函数:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{\,2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{\,10000^{\,2i/d_{\text{model}}}}\right)
$$

其中 $pos$ 是位置,$i$ 是维度编号.也就是说,位置编码的每一维都对应一条正弦曲线,波长构成一个从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列.我们选择这个函数,是因为假设它能让模型轻松学会**按相对位置**去注意:对于任意固定偏移 $k$,$PE_{pos+k}$ 都可以表示成 $PE_{pos}$ 的线性函数.

**我的感悟**

这个编码设计得很精巧,我分三层说:

- **每一维是一条不同频率的波**:低维(小 $i$)对应高频短波,在相邻位置间剧烈变化,负责刻画**近处**的细微位置差;高维(大 $i$)对应低频长波,缓慢变化,负责刻画**远处**的粗粒度位置.一组从 $2\pi$ 到 $10000 \cdot 2\pi$ 的波长,等于同时给了模型精度不同的一排"尺子".
- **为什么有利于相对位置?** 用三角恒等式展开:

$$
\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)
$$

也就是说,$PE_{pos+k}$ 可以由 $PE_{pos}$ 的正弦、余弦分量**线性组合**得到(系数只跟 $k$ 有关).模型只要学出这些线性关系,就能知道"我和某个词隔了多远",而不必死记绝对位置.
- 奇偶维分别用 sin 和 cos 配对,正是为了让上述线性变换在每一对维度内闭合.

#### 第 3 段:固定编码 vs 学习式编码

> We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

**译文**

我们也试验过改用**学习式的位置嵌入** [9],发现两种版本产生了几乎完全相同的结果(见表 3 的 (E) 行).我们最终选择正弦版本,是因为它或许能让模型**外推(extrapolate)到比训练时见过的更长的序列**.

**我的感悟**

两种路线的取舍:

- **学习式**:给每个位置一个可训练向量,简单直接,但训练时只见过位置 0 到 $L$,遇到第 $L+1$ 个位置就没有对应的向量了,**无法外推**;
- **正弦式**:位置是公式算出来的,任意 $pos$(哪怕训练时没见过)都能现算,理论上可以泛化到更长序列.

实验结果两者几乎打平(Table 3 row E),说明位置信息的注入"有没有"比"具体哪种形式"更重要.作者选正弦,图的就是那个外推的可能性.

注:位置编码后来成了 Transformer 改进最活跃的方向之一——学习式绝对位置(GPT 系)、相对位置编码(T5)、再到今天大模型几乎标配的 **RoPE(旋转位置编码)**,通过对 Q、K 做旋转把相对位置直接编进注意力点积,可以看作"正弦相对位置"思想的高阶进化.但无论哪种,要解决的根本问题都是这一段点出的:**让对顺序盲目的注意力,知道词的位置**.

## 4 Why Self-Attention(为什么用自注意力)

### 第 1 段:比较的设定与三个评价维度

> In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $(x_1, ..., x_n)$ to another sequence of equal length $(z_1, ..., z_n)$, with $x_i, z_i \in \mathbb{R}^d$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

**译文**

本节中,我们把自注意力层与常用的循环层、卷积层在多个方面进行比较.这些层都用于把一个变长符号表示序列 $(x_1, \dots, x_n)$ 映射为另一个等长序列 $(z_1, \dots, z_n)$,其中 $x_i, z_i \in \mathbb{R}^d$,就像典型序列转换模型编码器或解码器中的一个隐藏层.为论证自注意力的合理性,我们考察**三个评价指标(desiderata)**.

**我的感悟**

比较之前先把问题严格框定:大家都是"等长序列到等长序列"的映射层,输入输出接口一致,这样才是公平对比.然后作者宣布要用三把尺子量,这三把尺子就是接下来两段:

### 第 2 段:维度一——计算复杂度;维度二——并行度

> One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

**译文**

第一把尺子是**每层的总计算复杂度**;第二把尺子是**可并行的计算量**,用完成该层所需的**最少顺序操作数**来衡量(顺序操作数越少,可并行部分越多).

**我的感悟**

这两个指标必须分开看,它们不是一回事:

- **计算复杂度**:总共要做多少算术操作,决定总成本、总耗电;
- **最少顺序操作数**:这些操作里有多少是"必须排队、一个等一个"的,决定**能不能靠加 GPU 核心把时间摊薄**.

一个层可以总操作数不多,但因为全是顺序依赖而很慢(RNN 正是如此);也可以总操作数略多,但全部能一次性并行铺开(attention、CNN 正是如此).GPU 的算力红利只奖励后者.

### 第 3 段:维度三——长程依赖的路径长度

> The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

**译文**

第三把尺子是网络中长程依赖之间的**路径长度**.学习长程依赖是许多序列转换任务的关键挑战.影响这种能力的一个关键因素,是前向信号和反向信号在网络中必须穿越的路径长度.输入、输出序列中任意位置组合之间的路径越短,就越容易学到长程依赖 [12].因此我们还比较了由不同层类型构成的网络中,任意两个输入/输出位置之间的**最大路径长度**.

**我的感悟**

路径长度为什么同时影响前向和反向:

- **前向**:从句首到句尾的信息,要经过多少层、多少步才能"见面";
- **反向(更关键)**:梯度从远处位置回传到源头位置,路径越长,梯度连乘越多次,越容易消失或爆炸,长程依赖就越学不动.这正是 Hochreiter 等人 [12] 论证 RNN 难学长程依赖的核心机理.

所以"最大路径长度"本质上是在量**梯度走的距离**.路径短 = 梯度高速公路 = 远距离关系可学.

### Table 1:三种层的正面交锋

> Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.

**译文(表题)**

表 1:不同层类型的最大路径长度、每层复杂度和最少顺序操作数.$n$ 为序列长度,$d$ 为表示维度,$k$ 为卷积核大小,$r$ 为受限自注意力中的邻域大小.

| 层类型 | 每层复杂度 | 最少顺序操作数 | 最大路径长度 |
|---|---|---|---|
| 自注意力 | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| 循环(RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积(CNN) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| 受限自注意力 | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

**我的感悟**

这张表是整篇论文论证的"腰",读透它就理解了架构之争:

- **自注意力**:顺序操作 $O(1)$、路径长度 $O(1)$——并行度和长程依赖双满分,唯一的"软肋"是复杂度里的 $n^2$;
- **RNN**:顺序操作和路径长度都是 $O(n)$——两项全输,这就是基因病的量化;
- **CNN**:顺序操作 $O(1)$(并行没问题),但单层路径长,靠堆叠/膨胀卷积做到 $O(\log_k n)$,介于两者之间;
- **受限自注意力**:作者预留的折中方案,只看半径 $r$ 的邻域,把 $n^2$ 降成 $r \cdot n$,代价是路径长度回到 $O(n/r)$.

### 第 4 段:什么时候自注意力反而更便宜

> As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O(n/r)$. We plan to investigate this approach further in future work.

**译文**

如表 1 所示,自注意力层用常数个顺序操作就连接了所有位置,而循环层需要 $O(n)$ 个顺序操作.在计算复杂度上,**当序列长度 $n$ 小于表示维度 $d$ 时,自注意力层比循环层更快**;机器翻译里最先进模型所用的句子表示(如 word-piece [38]、byte-pair [31] 表示)通常正是这种情况.为了在超长序列任务上提升计算性能,可以让自注意力只考虑以相应输出位置为中心、大小为 $r$ 的邻域,这会把最大路径长度增加到 $O(n/r)$.我们计划在未来工作中进一步研究这一思路.

**我的感悟**

这里给出了自注意力 $O(n^2 d)$ 与 RNN $O(n d^2)$ 的盈亏平衡点:**比较 $n$ 和 $d$**.

- $n < d$ 时,$n^2 d < n d^2$,自注意力又快又能并行;翻译句子 $n$ 通常几十,而 $d = 512$,所以稳赢.
- $n \gg d$ 时(长文档、长代码、图像展平成的 token 序列),$n^2$ 项爆炸,自注意力反而更贵.这正是后来长文本建模的核心痛点,也是今天 FlashAttention(省显存 IO)、稀疏/局部注意力(Longformer 等)、滑动窗口注意力(Mistral)等一大堆工作的出发点——**作者在 2017 年就把这个坑和药方(restricted attention)写好了**.

### 第 5 段:和卷积的进一步算账

> A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n/k)$ convolutional layers in the case of contiguous kernels, or $O(\log_k(n))$ in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$. Separable convolutions [6], however, decrease the complexity considerably, to $O(k \cdot n \cdot d + n \cdot d^2)$. Even with $k = n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

**译文**

核宽 $k < n$ 的单层卷积,无法连接所有输入/输出位置对.要做到全连接,连续核卷积需要堆叠 $O(n/k)$ 层,膨胀卷积 [18] 需要 $O(\log_k n)$ 层,这都会拉长网络中任意两位置间的最长路径.卷积层通常比循环层还贵 $k$ 倍.不过**可分离卷积(separable convolution)**[6] 能把复杂度大幅降到 $O(k \cdot n \cdot d + n \cdot d^2)$.然而即便取 $k = n$,可分离卷积的复杂度也只等于"一个自注意力层 + 一个逐点前馈层"的组合——而这正是我们模型采用的方案.

**我的感悟**

这段论证非常"杀人诛心":作者没有拿普通 CNN 欺负人,而是假设对手用上最省的**深度可分离卷积**,再把核宽开到极限 $k = n$(一层就能全局连接),结果其复杂度 $O(n^2 d + n d^2)$ **恰好等于"自注意力 + FFN"**.

换句话说:卷积路线为了获得全局感受野,在最理想情况下的复杂度下限,正好就是 Transformer 一层的构造;而 Transformer 还白赚了 $O(1)$ 的路径长度和更直接的可解释性.这等于把"那我用更强的 CNN 行不行?"这条退路也堵死了.

### 第 6 段:附带好处——可解释性

> As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

**译文**

作为附带好处,自注意力能产生**更可解释(interpretable)**的模型.我们检查了模型的注意力分布,并在附录中展示和讨论了若干例子:各个注意力头不仅明显学会了执行不同的任务,其中许多头还表现出与句子句法、语义结构相关的行为.

**我的感悟**

RNN 把信息压在一个不断被覆写的隐藏状态里,想看它"在想什么"非常困难;而注意力显式产出一个 $n \times n$ 的权重矩阵,**模型每一步在看哪些词,是明牌摆在那里的**,直接可视化即可.

附录 Figure 3-5 的例子后来很有名:有的头自动追踪 "making ... more difficult" 这种隔了好几个词的动词搭配,有的头专门做 "its" 的指代消解.单个头的注意力图清晰对应句法结构.虽然今天我们对"注意力 = 可解释性"会更谨慎(注意力权重高不等于因果重要),但在 2017 年,这种"打开黑盒能看见东西"的特性确实是实打实的加分项.

## 5 Training(训练)

> This section describes the training regime for our models.

**译文**

本节描述模型的训练方案.

### 5.1 Training Data and Batching(训练数据与批处理)

> We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

**译文**

我们在标准 WMT 2014 英德数据集上训练,约含 450 万句对.句子用**字节对编码(BPE,Byte-Pair Encoding)**[3] 编码,源语言和目标语言共享一个约 37000 token 的词表.英法任务则使用大得多的 WMT 2014 英法数据集,含 3600 万句,token 被切分为 32000 的 word-piece 词表 [38].句对按大致长度分批,每个训练 batch 包含约 25000 个源 token 和 25000 个目标 token.

**我的感悟**

两个工程点值得记住:

- **为什么用 BPE / word-piece 这类子词(subword)单元?** 翻译时总会遇到生词、变形词、专有名词,纯词级词表要么 OOV(未登录词),要么词表爆炸.子词把罕见词拆成片段,比如 `playing` → `play` + `ing`,用两三万的固定小词表就能覆盖几乎所有文本,还能在源/目标语言间共享词表.这是今天所有大模型 tokenizer 的鼻祖(SentencePiece、tiktoken 都是这套).
- **按长度分批**:把长度相近的句子放一个 batch,减少 padding(填充)浪费;每批按 token 总数(约 2.5 万)而不是句数来定,让每个 batch 的算力负载均匀.这是处理变长序列的标准技巧.

### 5.2 Hardware and Schedule(硬件与训练日程)

> We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, (described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

**译文**

我们在一台装有 8 块 NVIDIA P100 GPU 的机器上训练模型.使用全文所述超参数的 base 模型,每步约 0.4 秒,共训练 10 万步,约 12 小时.big 模型(见表 3 最后一行)每步 1.0 秒,共训练 30 万步,约 3.5 天.

**我的感悟**

这就是论文里反复出现的两个训练成本数字的出处,我把两个配置对齐:

| 配置 | 每步耗时 | 总步数 | 总时长 |
|---|---|---|---|
| Transformer(base) | 0.4 秒 | 100,000 | 约 12 小时 |
| Transformer(big) | 1.0 秒 | 300,000 | 约 3.5 天 |

以今天的眼光看,8 块 P100、训练半天到几天,成本低得惊人,却刷下了别人训练几周的模型——这才是论文最有说服力的地方:**不是靠堆算力赢,是靠架构效率赢**.

### 5.3 Optimizer(优化器)

> We used the Adam optimizer [20] with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$. We varied the learning rate over the course of training, according to the formula:
>
> $\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step\_num}^{-0.5},\; \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}) \tag{3}$
>
> This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

**译文**

我们使用 Adam 优化器 [20],取 $\beta_1 = 0.9$、$\beta_2 = 0.98$、$\epsilon = 10^{-9}$.训练过程中学习率按下式变化:

$$
\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5}\right) \tag{3}
$$

这对应于:前 warmup_steps 步学习率**线性上升**,之后按步数平方根的倒数(即 $\text{step}^{-0.5}$)**衰减**.我们取 warmup_steps = 4000.

**我的感悟**

这个学习率调度(常被称为 **Noam schedule / warmup-decay**)是 Transformer 训练的标志性配方:

- **为什么要 warmup(热身)?** 训练刚开始时权重是随机的,Q、K、V 的投影都还没成型,直接上大学习率容易把模型带飞、梯度爆炸.所以前 4000 步把学习率从 0 线性拉到峰值,让各层先稳定下来.
- **为什么之后衰减?** 训练后期需要小步微调,在最优点附近收敛,而不是一直大步横跳.
- 公式里那个 $d_{\text{model}}^{-0.5}$ 表示**模型越宽,峰值学习率越小**——维度大、参数多,更新要更保守.

注:这套"线性 warmup + 余弦/平方根衰减"的调度至今仍是训练大模型的默认姿势,只是衰减部分今天更多换成余弦退火;$\beta_2 = 0.98$(比 Adam 默认的 0.999 小)这种让二阶矩估计更新更快的设置也被长期沿用.

### 5.4 Regularization(正则化)

> We employ three types of regularization during training:

（注:论文正文实际展开了残差 Dropout 与标签平滑两种,配合数据/训练层面的处理。）

#### Residual Dropout(残差 Dropout)

> We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{drop} = 0.1$.

**译文**

我们在每个子层的输出上、在它与子层输入相加并归一化**之前**,施加 dropout [33].此外,编码器和解码器堆栈中,嵌入与位置编码相加之后也施加 dropout.base 模型取丢弃率 $P_{drop} = 0.1$.

**我的感悟**

记住 dropout 的两个落点:

- **每个子层的输出**(attention 的输出、FFN 的输出),在进入残差相加前随机置零;
- **embedding + 位置编码的和**,在进入堆栈前随机置零.

dropout 强迫网络不依赖任何单个位置、单条通路,提升鲁棒性、抑制过拟合.它必须放在 LayerNorm 之前,因为归一化后再随机挖洞会破坏刚稳定下来的尺度.

#### Label Smoothing(标签平滑)

> During training, we employed label smoothing of value $\epsilon_{ls} = 0.1$ [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

**译文**

训练期间,我们使用 $\epsilon_{ls} = 0.1$ 的标签平滑 [36].它会**损害困惑度(perplexity)**,因为模型学着变得不那么确定;但能提升准确率和 BLEU 分数.

**我的感悟**

标签平滑改的是训练目标:正常情况下,正确词的目标概率是 1、其他词全是 0,模型会拼命把正确词的 logit 推向无穷大、把自己训练得极度自信;平滑后,正确词目标概率变成 $1 - \epsilon = 0.9$,那 0.1 的概率质量均摊给其他所有词.

好处是防止模型过度自信、留出容错空间,改善预测的校准(calibration),BLEU 和准确率反而更高.

但有个反直觉的点:**它让困惑度 PPL 变差**.因为 PPL 是 $e$ 的交叉熵次方,它只奖励"给正确词多高的概率",你把 0.1 分给别人,正确词概率从 0.99 降到 0.9,PPL 数字必然难看.这是一个绝佳的例子说明**优化指标 ≠ 最终业务指标**——PPL 只是代理指标,真正要的 BLEU/准确率涨了就行.读 Table 3 时看到 big 模型 PPL 更低却未必各方面都更好,口径要心里有数.

## 6 Results(结果)

### 6.1 Machine Translation(机器翻译)

> Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

**译文(表题)**

表 2:在英德、英法 newstest2014 测试上,Transformer 以一小部分训练成本取得了比此前最先进模型更好的 BLEU 分数.

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本(FLOPs) |
|---|---|---|---|
| ByteNet [18] | 23.75 | — | — |
| Deep-Att + PosUnk [39] | — | 39.2 | $1.0 \times 10^{20}$ |
| GNMT + RL [38] | 24.6 | 39.92 | $2.3 \times 10^{19}$ / $1.4 \times 10^{20}$ |
| ConvS2S [9] | 25.16 | 40.46 | $9.6 \times 10^{18}$ / $1.5 \times 10^{20}$ |
| MoE [32] | 26.03 | 40.56 | $2.0 \times 10^{19}$ / $1.2 \times 10^{20}$ |
| GNMT + RL 集成 [38] | 26.30 | 41.16 | $1.8 \times 10^{20}$ / $1.1 \times 10^{21}$ |
| ConvS2S 集成 [9] | 26.36 | 41.29 | $7.7 \times 10^{19}$ / $1.2 \times 10^{21}$ |
| **Transformer(base)** | 27.3 | 38.1 | $3.3 \times 10^{18}$ |
| **Transformer(big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

#### 第 1 段:英德任务——单模型击败所有集成

> On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

**译文**

在 WMT 2014 英德任务上,big 模型(表 2 中 Transformer(big))比此前报告的最佳模型(**包括集成模型**)还高出 2.0 BLEU 以上,创下 28.4 的新纪录.模型配置列在表 3 最后一行,在 8 块 P100 上训练 3.5 天.**就连 base 模型也超过了此前所有已发表的模型和集成**,而训练成本只是任何竞争模型的零头.

**我的感悟**

最值得强调的是"**单模型击败集成(ensemble)**"这件事.在深度学习里,集成(把多个模型的预测平均)几乎是免费的涨点手段,代价是几倍训练和推理成本;此前 SOTA 基本都靠集成堆出来.Transformer 一个模型就把别人一堆模型的联合体挑落,这是架构层面的代差,不是调参能弥补的.

再看训练成本那一列更直观:Transformer big 是 $2.3 \times 10^{19}$ FLOPs,而 ConvS2S、GNMT 的集成动辄 $10^{20} \sim 10^{21}$,**差出一个数量级**.效果更好、成本更低,这就是论文标题的底气.

#### 第 2 段:英法任务

> On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{drop} = 0.1$, instead of 0.3.

**译文**

在 WMT 2014 英法任务上,我们的 big 模型取得 41.0 的 BLEU,超过此前所有已发表的单模型,而训练成本不到此前最先进模型的 1/4.英法 big 模型使用的 dropout 率为 $P_{drop} = 0.1$,而不是 0.3.

**我的感悟**

注:这里正文写的是 **41.0**,但摘要和表 2 里写的是 **41.8**,这是论文原文自身的一处数字不一致(社区版本长期如此).引用时以表格和摘要的 **41.8** 为准,读论文遇到这种对不上的地方不用慌,知道是原文的小笔误即可.

另一个细节:big 模型在英德任务用 dropout 0.3,英法因为数据量大得多(3600 万句对)、过拟合压力小,反而降回 0.1.这说明**正则化强度要跟着数据规模走**,没有万能的固定值.

#### 第 3 段:推理与解码细节

> For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty $\alpha = 0.6$ [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

**译文**

base 模型我们把最后 5 个 checkpoint(每 10 分钟存一个)的权重平均成一个模型;big 模型平均最后 20 个.解码使用 **beam search(束搜索)**,束宽为 4,长度惩罚系数 $\alpha = 0.6$ [38],这些超参在开发集上实验后选定.推理时最大输出长度设为输入长度 + 50,但条件允许时提前终止.

**我的感悟**

两个实用技巧:

- **Checkpoint 平均(weight averaging)**:不直接用最后一步的权重,而是把训练末期若干个 checkpoint 平均,等价于一种"参数空间的集成",几乎零成本地让模型更平滑、更稳,常常白捡零点几个 BLEU.这是后来 SWA/EMA(权重滑动平均)进入大模型训练的雏形.
- **Beam search + 长度惩罚**:自回归生成时,beam=4 意味着每步保留 4 条候选序列再择优,比贪心解码好;长度惩罚 $\alpha$ 用来纠正模型"偏爱短句子"的倾向(NLP 里分数对长度敏感).最大长度设输入 + 50 并允许提前停(EOS),兼顾安全与效率.

#### 第 4段:训练成本怎么算

> Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

**译文**

表 2 总结了结果,并把翻译质量和训练成本与文献中的其他架构做了比较.我们估算训练一个模型所用浮点操作数的方法是:**训练时长 × 使用的 GPU 数量 × 每块 GPU 的单精度持续浮点算力估计值**.

**我的感悟**

论文脚注给了当时各 GPU 的持续单精度算力估计:K80 为 2.8、K40 为 3.7、M40 为 6.0、P100 为 9.5 TFLOPS.

注意它用的是"**持续(sustained)**算力"而不是厂商标称的峰值——真实训练中算力利用率远到不了峰值,用可持续的数字估算才公平.这个方法论今天依然适用:比较训练成本时,口径(峰值还是实测、是否含通信/空闲时间)必须统一,否则 FLOPs 数字没有可比性.

### 6.2 Model Variations(模型变体消融)

> To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

**译文**

为评估 Transformer 各组件的重要性,我们以不同方式改动 base 模型,在英德翻译开发集 newstest2013 上测量性能变化.解码用上一节所述 beam search,但不做 checkpoint 平均,结果见表 3.

**我的感悟**

这就是**消融实验(ablation study)**:每次只动一个旋钮,看性能怎么变,从而反推每个设计到底有没有用、有多重要.这是验证架构设计合理性最硬的证据.

Table 3 内容很多,我抽出最关键的几行(PPL 为开发集困惑度,BLEU 为开发集分数):

| 改动 | $N$ | $d_{\text{model}}$ | $h$ | $d_k/d_v$ | PPL↓ | BLEU↑ |
|---|---|---|---|---|---|---|
| **base(基准)** | 6 | 512 | 8 | 64 | 4.92 | 25.8 |
| (A) 单头 | 6 | 512 | 1 | 512 | 5.29 | 24.9 |
| (A) 4 头 | 6 | 512 | 4 | 128 | 5.00 | 25.5 |
| (A) 16 头 | 6 | 512 | 16 | 32 | 4.91 | 25.8 |
| (A) 32 头 | 6 | 512 | 32 | 16 | 5.01 | 25.4 |
| (C) 2 层 | 2 | 512 | 8 | 64 | 6.11 | 23.7 |
| (C) $d_{\text{model}}=256$ | 6 | 256 | 8 | 32 | 5.75 | 24.5 |
| (C) $d_{\text{model}}=1024$ | 6 | 1024 | 16 | 128 | 4.66 | 26.0 |
| (D) 无 dropout | 6 | 512 | 8 | 64 | 5.77 | 24.6 |
| (E) 学习式位置编码 | 6 | 512 | 8 | 64 | 4.92 | 25.7 |
| **big** | 6 | 1024 | 16 | 64 | 4.33 | 26.4 |

#### 第 2 段:头数的消融

> In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

**译文**

表 3 的 (A) 行中,我们改变注意力头数以及 key/value 维度,同时如 3.2.2 节所述**保持总计算量恒定**.单头注意力比最佳设置差 0.9 BLEU,而头数过多时质量同样会下降.

**我的感悟**

头数是个"两头都不好"的超参:

- **太少(单头)**:没有多子空间,差 0.9 BLEU,直接证明多头设计有效;
- **太多(32 头,每头仅 16 维)**:每个头的维度被压得太窄,装不下足够的信息来刻画关系,质量也回落.

8 头 / 每头 64 维是这张表里的甜点.这印证了一个朴素道理:**并行的专家数量和每个专家的能力(维度)要平衡**,光拆得细没用.

#### 第 3 段:key 维度、模型规模、dropout、位置编码

> In Table 3 rows (B), we observe that reducing the attention key size $d_k$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

**译文**

表 3 的 (B) 行中,我们观察到**缩小 key 的维度 $d_k$ 会损害质量**.这说明判定相容性并不简单,也许比点积更复杂的相容性函数会有帮助.在 (C)、(D) 行我们进一步看到(正如预期):**模型越大越好,dropout 对避免过拟合非常有帮助**.(E) 行把正弦位置编码换成学习式位置嵌入 [9],结果与 base 几乎相同.

**我的感悟**

四条结论各有深意:

- **$d_k$ 不能太小**:点积打分需要足够维度才能判断两个词是否匹配,维度不够就"看不清".作者还顺势点了一句——也许点积并非最优相容性函数,这为后来各种 attention 打分改进留了口子.
- **大模型更好**(C):加层、加宽都稳定涨点,这其实是"scaling 有效"的早期信号,为日后大力出奇迹埋下伏笔.
- **dropout 很关键**(D):关掉 dropout,BLEU 从 25.8 掉到 24.6,过拟合立现.
- **位置编码形式无所谓**(E):学习式和正弦打平,再次说明位置信息"有没有"是关键.

### 6.3 English Constituency Parsing(英语成分句法分析)

> Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ).

**译文(表题)**

表 4:Transformer 很好地泛化到英语成分句法分析(结果在 WSJ 第 23 节上测得).

| 解析器 | 训练方式 | WSJ 23 F1 |
|---|---|---|
| Petrov et al. (2006) [29] | 仅 WSJ,判别式 | 90.4 |
| Dyer et al. (2016) [8] | 仅 WSJ,判别式 | 91.7 |
| **Transformer(4 层)** | **仅 WSJ,判别式** | **91.3** |
| Zhu et al. (2013) [40] | 半监督 | 91.3 |
| McClosky et al. (2006) [26] | 半监督 | 92.1 |
| **Transformer(4 层)** | **半监督** | **92.7** |
| Dyer et al. (2016) [8] | 生成式 | 93.3 |

#### 第 1 段:这个任务为什么难

> To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

**译文**

为评估 Transformer 能否泛化到其他任务,我们在英语成分句法分析上做了实验.这个任务有特殊挑战:输出受到很强的结构约束(必须是一棵合法的句法树),而且**比输入长得多**;此外,RNN 序列到序列模型在小数据场景下一直无法达到最先进水平 [37].

**我的感悟**

选这个任务是精心设计的"压力测试",它专挑 Transformer(以及 seq2seq 范式)的软肋:

- 输出不是自由文本,而是带嵌套、括号必须配对的句法树,结构约束极强;
- 输出比输入长很多(一个词可能对应一串树节点),对生成长序列能力要求高;
- 训练数据小(仅几万句),最能检验架构在"缺数据"时是否还扛得住——这也是摘要里特意强调 "large and limited data" 的原因.

#### 第 2 段:训练配置

> We trained a 4-layer transformer with $d_{\text{model}} = 1024$ on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

**译文**

我们在 Penn Treebank [25] 的华尔街日报(WSJ)部分训练了一个 4 层、$d_{\text{model}} = 1024$ 的 Transformer,约 4 万训练句.我们还在半监督设定下训练,使用更大的高置信语料和 BerkeleyParser 语料,约 1700 万句 [37].仅 WSJ 设定用词表 16K,半监督设定用词表 32K.

**我的感悟**

有意思的是这里**刻意没有为新任务大改架构**:层数从 6 减到 4、宽度保持 1024,其余基本照搬翻译模型.作者想证明的是 Transformer 的**通用性**——不是靠针对句法分析重新设计一堆结构才赢,而是开箱即用.

#### 第 3 段:几乎不做任务特定调参

> We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of 21 and $\alpha = 0.3$ for both WSJ only and the semi-supervised setting.

**译文**

我们只在 Section 22 开发集上做了少量实验来选定 dropout(注意力和残差两处,见 5.4)、学习率和束宽,其他所有参数都与英德 base 翻译模型保持一致.推理时把最大输出长度提高到输入长度 + 300;仅 WSJ 和半监督两种设定都使用束宽 21、$\alpha = 0.3$.

**我的感悟**

"all other parameters remained unchanged"(其他参数原样不动)是这一段的文眼.一个为机器翻译设计的模型,几乎不改参数就能迁移到结构完全不同的句法分析,这是泛化能力最有说服力的展示.最大输出长度从翻译的 +50 提到 +300,也印证了句法树输出显著长于输入.

#### 第 4、5 段:结果

> Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].
>
> In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

**译文**

表 4 结果表明,尽管缺乏任务特定调参,模型表现却出奇地好:除 RNN Grammar [8] 外,结果优于此前所有报告过的模型.

与 RNN 序列到序列模型 [37] 形成对比的是,**Transformer 仅在 4 万句 WSJ 训练集上训练,就超过了 BerkeleyParser** [29].

**我的感悟**

半监督设定下 Transformer 拿到 92.7 F1,超过一众半监督方法;仅 WSJ 的小数据设定也有 91.3,击败经典的 BerkeleyParser(90.4).要知道 BerkeleyParser 是专为句法分析设计的、积累多年的专用系统,而 Transformer 是个"半路出家"的通用模型,小数据下还能赢——这正面回应了第 1 段说的"RNN seq2seq 在小数据下不行":**同样的小数据,换成 Transformer 就行了**,问题出在架构而非数据量.

唯一没超过的是 RNN Grammar [8](93.3),作者也如实交代,没有夸大成"全胜".这种诚实在 SOTA 论文里很重要.

## 7 Conclusion(结论)

### 第 1 段:总结贡献

> In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

**译文**

本工作中,我们提出了 Transformer——**第一个完全基于注意力的序列转换模型**,用多头自注意力替换了编码器-解码器架构中最常用的循环层.

**我的感悟**

一句话收束全文,和第 2 节 Background 末尾的"first"声明首尾呼应:完全基于注意力 + 多头自注意力替换循环层,就是这篇论文的全部历史定位.

### 第 2 段:重申结果

> For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

**译文**

在翻译任务上,Transformer 的训练速度显著快于基于循环层或卷积层的架构.在 WMT 2014 英德和英法两个任务上,我们都取得了新的最先进水平;英德任务上,我们最好的模型甚至超过了此前所有报告过的集成模型.

**我的感悟**

结论再次把"快"(训练显著更快)和"好"(双任务 SOTA、单模型胜集成)两个核心卖点钉一遍.论文论证至此形成闭环:动机(顺序计算之痛)→ 方案(全注意力 Transformer)→ 优势(三维度对比)→ 训练(可复现配方)→ 结果(又快又好还能泛化).

### 第 3 段:未来展望

> We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

**译文**

我们对基于注意力的模型的未来感到兴奋,并计划把它们应用到其他任务.我们计划把 Transformer 扩展到文本之外的输入/输出**模态**,研究局部的、受限的注意力,以高效处理图像、音频、视频等大型输入输出;让生成过程不那么顺序化,也是我们的另一个研究目标.

**我的感悟**

回头看,这段展望几乎成了此后几年 AI 发展的路线图,而且大部分都应验了:

- **多模态**:ViT(把图像切成 patch 当 token)、视觉 Transformer、音频/视频 Transformer,直到今天能同时处理图文音视频的多模态大模型;
- **局部/受限注意力**:Longformer、滑动窗口注意力、FlashAttention 等,解决长序列 $n^2$ 成本;
- **减少生成的顺序性**:非自回归生成(NAR)、并行解码、推测解码(speculative decoding)等,都在试图打破自回归一个词一个词吐的瓶颈.

读论文的结论部分,我习惯特别留意"future work",因为它往往是下一阶段研究的富矿——这篇的预测命中率高得惊人.

### 第 4 段:代码开源

> The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

**译文**

我们用于训练和评估模型的代码已开源:https://github.com/tensorflow/tensor2tensor .

**我的感悟**

开源代码是这篇论文影响力能迅速放大的重要原因.研究者能直接复现、改造,Transformer 才得以快速扩散到整个社区.今天的 tensor2tensor 早已完成历史使命,但"重要工作配开源实现"自此成为顶级论文的惯例.

(Acknowledgements:作者感谢了 Nal Kalchbrenner 和 Stephan Gouws 富有成效的评论、修正与启发,此处不再展开.)

## 小结

读到这里,我把整篇论文串成一条主线和一张总图.

**一条主线**:RNN 的顺序计算是写在定义里的基因病($h_t = f(h_{t-1}, x_t)$,训练样本内无法并行、长程路径 $O(n)$)→ 注意力本就能无视距离建立依赖,却长期被绑在 RNN 上当补丁 → Transformer 把注意力扶正,**完全抛弃循环与卷积** → 用多头子空间解决单次加权平均"抹平"信息的问题,用位置编码补上注意力对顺序的盲目 → 换来 $O(1)$ 路径长度、$O(1)$ 顺序操作和高度并行,以几分之一的训练成本全面刷新 SOTA,还能泛化到句法分析.

**一张总图(单层结构)**:

```text
编码器层 ×6                          解码器层 ×6
┌───────────────────────┐           ┌───────────────────────────────┐
│ Multi-Head Self-Attn  │           │ Masked Multi-Head Self-Attn   │  只能看 ≤ 当前
│   + Residual & LN     │           │   + Residual & LN             │
│ Position-wise FFN     │           │ Encoder-Decoder Attn (Q←dec,  │  看编码器全部
│ 512→2048→512          │           │   K/V←encoder z) + Res & LN   │
│   + Residual & LN     │           │ Position-wise FFN + Res & LN  │
└───────────────────────┘           └───────────────────────────────┘
        │ z (n×512) ───────────────────────► 供每层第②个子层使用
  底部:Input Embedding + Positional Encoding   底部:Output Embedding(偏移1位)+ PE
  顶部输出 → Linear → Softmax → 下一 token 概率
```

**几个必须记住的数字**:

| 项目 | base | big |
|---|---|---|
| 编码器/解码器层数 $N$ | 6 | 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| 注意力头数 $h$ | 8 | 16 |
| 每头维度 $d_k = d_v$ | 64 | 64 |
| FFN 内层 $d_{ff}$ | 2048 | 4096 |
| 训练时长 / 硬件 | 12 小时 / 8×P100 | 3.5 天 / 8×P100 |
| EN-DE / EN-FR BLEU | 27.3 / 38.1 | 28.4 / 41.8 |

最后说点我自己的感受.这篇论文最打动我的,不是它数学多艰深——恰恰相反,它的核心公式简单到一张纸就能写完——而是它那种**敢于把"公认必需"的东西整个拿掉**的判断力:别人都在想"怎么把 RNN 优化得更快",它问的是"为什么非得有 RNN".这种从根上重新审视前提的思维方式,比 Transformer 架构本身更值得学.

而 Transformer 后来的故事我们都知道了:编码器孕育出 BERT,解码器孕育出 GPT,完整结构孕育出 T5,再往后是参数规模成千上万倍膨胀的大模型时代.2017 年那句轻描淡写的 "Attention Is All You Need",事后看几乎是一封新时代的开幕宣言.





