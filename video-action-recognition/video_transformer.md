Video Transformers Paper and Notes
=======
# Some Transformer basics
1. [3W字长文带你轻松入门视觉transformer](https://zhuanlan.zhihu.com/p/308301901) (Chinese)

# Video Transformers
* [Is Space-Time Attention All You Need for Video Understanding? ICML 2021](https://arxiv.org/pdf/2102.05095.pdf). 

    Code: [mmaction](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README.md), [Official](https://github.com/facebookresearch/TimeSformer), [Popular one](https://github.com/lucidrains/TimeSformer-pytorch)


<p align="center"><img src="./images/video_transformers/TimeSformer.png" width="500px"></img> 

<p align="center"><img src="./images/video_transformers/TimeSformer2.png" width="600px"></img>

This paper proposes Divided Space-Time Attention (T+S) operation for the standard Transformer architecture ([ViT backbone](https://arxiv.org/pdf/2010.11929.pdf) is adapted for videos). As seen in Fig. 2, T+S calculates 1) temporal attention for **only** current patch (marked in blue) and the same location patch at different time stamps, and 2) the spatial attention for each patch in each frame. 

### Details from [Official code](https://github.com/facebookresearch/TimeSformer):
Given a video clip input $\mathbf{X}\in \mathbb{R}^{B\times C\times T\times H\times W}$. 

1) PatchEmbed (image to patch embedding): each frame is decomposed into $N$ size of $P\times P$ patches ($N=\frac{H\times W}{P^2}$, $P=16$ in this work, 2D convolution operates on each patch to get linear emebedding). Then flatten over spatial and time: $\mathbf{X}\in \mathbb{R}^{(B*T)\times (N+1)\times D}$ ($D$ is the output channel of first 2D convolution); 

2) Spatial position encoding: Add *cls_token* $\to$ $\mathbf{X}\in \mathbb{R}^{(B*T)\times (N+1)\times D}$, then add postion encoding;

3) Temporal embedding: Extract temporal *cls_tokens = X[:B, 0:1, :]*. Operate temporal position encoding on $\mathbf{X}\in \mathbb{R}^{(B*N)\times T\times D}$ and rearange back to $\mathbf{X}\in \mathbb{R}^{B\times (N*T)\times D}$. Add temporal *cls_tokens* to get $\mathbf{X}\in \mathbb{R}^{B\times (N*T+1)\times D}$

4) Temporal attention and spatial attention: It should be noted that process *cls_token* for temporal and spatial respectively. 1) Temporal attention: $\mathbf{X}\in \mathbb{R}^{(B*N)\times T\times D}$; 2) Spatial attention: $\mathbf{X}\in \mathbb{R}^{(B*T)\times N\times D}$. 




