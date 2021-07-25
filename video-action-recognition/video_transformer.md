Video Transformers Paper and Notes
=======
## Best view exporting .md to PDF
# Some Transformer basics
1. [3W字长文带你轻松入门视觉transformer](https://zhuanlan.zhihu.com/p/308301901) (Chinese)

# Video Transformers
## [(ICML 2021) Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf)
Code: [mmaction](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README.md), [Official](https://github.com/facebookresearch/TimeSformer), [Popular one](https://github.com/lucidrains/TimeSformer-pytorch)


<p align="center"><img src="./images/video_transformers/TimeSformer.png" width="500px"></img> 

<p align="center"><img src="./images/video_transformers/TimeSformer2.png" width="650px"></img>

### Overview
This paper proposes Divided Space-Time Attention (T+S) operation for the standard Transformer architecture ([ViT backbone](https://arxiv.org/pdf/2010.11929.pdf) is adapted for videos). As seen in Fig. 2, T+S calculates 1) temporal attention for **only** current patch (marked in blue) and the same location patch at different time stamps, and 2) the spatial attention for each patch in each frame. 

### Details from [Official code](https://github.com/facebookresearch/TimeSformer):
Given a video clip input $\mathbf{X}\in \mathbb{R}^{B\times C\times T\times H\times W}$.

1) PatchEmbed (image to patch embedding): each frame is decomposed into $N$ size of $P\times P$ patches ($N=\frac{H\times W}{P^2}$, $P=16$ in this work, 2D convolution operates on each patch to get linear emebedding). Then flatten over spatial and time: $\mathbf{X}\in \mathbb{R}^{(B*T)\times (N+1)\times D}$ ($D$ is the output channel of first 2D convolution); 

2) Spatial position encoding: Add *cls_token* $\to$ $\mathbf{X}\in \mathbb{R}^{(B*T)\times (N+1)\times D}$, then add postion encoding;

3) Temporal embedding: Extract temporal *cls_tokens = X[:B, 0:1, :]*. Operate temporal position encoding on $\mathbf{X}\in \mathbb{R}^{(B*N)\times T\times D}$ and rearange back to $\mathbf{X}\in \mathbb{R}^{B\times (N*T)\times D}$. Add temporal *cls_tokens* to get $\mathbf{X}\in \mathbb{R}^{B\times (N*T+1)\times D}$

4) Temporal attention and spatial attention: It should be noted that process *cls_token* for temporal and spatial respectively. 1) Temporal attention: $\mathbf{X}\in \mathbb{R}^{(B*N)\times T\times D}$; 2) Spatial attention: $\mathbf{X}\in \mathbb{R}^{(B*T)\times N\times D}$. 
   
Performance: K400 78.0, SSv2 59.5 ($8\times 224\times 224$ video clips).

## [(ICML 2021) Perceiver: General Perception with Iterative Attention](https://arxiv.org/pdf/2103.03206.pdf)
Code: [PyTorch](https://github.com/lucidrains/perceiver-pytorch)

[Explaination: Yannic Kilcher YouTube](https://www.youtube.com/watch?v=P_xeshTnPZg&ab_channel=YannicKilcher)

<p align="center"><img src="./images/video_transformers/perceiver.png" width="600px"></img>

### Overview
This paper introduces a general transformer model *Perceiver* with few architectural assumptions about the relationship between its inputs i.e., can be used for many input types such as images, texts. The model leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck, allowing it to scale to handle very large inputs. 

*Perceiver* generalizes the input by treating the input (big data like an image) as **Byte array (K, V)** as seen in the figure above. The **Latent array (Q)** (small) is **randomly initialized** and used for learning the feature from the **Byte array**. It should be noted that only **Q** is updated and fed to the next stage. The same **Byte array (K, V)** is used in the next stage. This is how *"iterative learning"* for the input mentioned in the paper. 

### Problems and Solutions
It can be noticed the iterative learning mentioned above is very computationally expensive regarding the cross attention between (Q, K, V). Suppose we have $K,V\in \mathbb{R}^{M\times D}$ and $Q\in \mathbb{R}^{M\times C}$, softmax$(QK^\top)V$ is $O(M^2)$. A $224\times 224\times 3$ image input will have $M$~150k. So this work project (Q, K, V) into lower dimension $N$ firstly. Then do the cross attention operation. The output is fed to the next latent transformer (do the self-attention).


## [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
Code: [Official PyTorch](https://github.com/microsoft/Swin-Transformer)

[Explaination: SOTA模型Swin Transformer是如何炼成的](https://zhuanlan.zhihu.com/p/376486858)

<p align="center"><img src="./images/video_transformers/swin_transformer.png" width="600px"></img>

### Overview
Direct transforming the transformer from langauge tasks to vision tasks causes problems such as heavy computation due to long patch tokens. This paper tackles this side by **reducing length of patch tokens** at each stage by concatenating them to channel dimension as seen in Fig. 3. This can be achieved by partitioning patches into several windows i.e., window multi-head self-attention (W-MSA), in which self-attention is applied inside windows and parameters are shared by different windows.  

### Shifted Windows
Splitted window operation is able to significantly reduce the computation, but it only focuses on the local attention and ignores the global field. The author overcome this by adding a shifted window multi-head self-attention (SW-MSA) after a W-MSA block. As seen in Fig. 2, each window in Layer 1 is shifted by $\pm\delta$ in the next layer. The extra computation is caused by this operation since there are 9 tokens compared to 4 tokens in the previous stage. The authors proposes the cyclic shift as seen in Fig. 4 to avoid this extra computation. The A, B, and C are moved to the down right, which now construct four windows same as pervious W-MSA. The calculated self-attention can be masked before the output.

<p align="center"><img src="./images/video_transformers/swin_transformer2.png" width="250px"></img>  <img src="./images/video_transformers/swin_transformer3.png" width="400px"></img>

## [Video Swin Transformer](https://arxiv.org/pdf/2106.13230.pdf)
Code: [Official](https://github.com/SwinTransformer/Video-Swin-Transformer)

