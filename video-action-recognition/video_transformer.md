Video Transformers Paper and Notes
=======
# Some Transformer basics
1. [3W字长文带你轻松入门视觉transformer](https://zhuanlan.zhihu.com/p/308301901) (Chinese)

# Video Transformers
* [Is Space-Time Attention All You Need for Video Understanding? ICML 2021](https://arxiv.org/pdf/2102.05095.pdf). 

    Code: [mmaction](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README.md), [Official](https://github.com/facebookresearch/TimeSformer), [Popular one](https://github.com/lucidrains/TimeSformer-pytorch)


<p align="center"><img src="./images/video_transformers/TimeSformer.png" width="600px"></img> <p align="center"><img src="./images/video_transformers/TimeSformer2.png" width="600px"></img>

This paper proposes Divided Space-Time Attention (T+S) operation for the standard Transformer architecture ([ViT backbone](https://arxiv.org/pdf/2010.11929.pdf) is adapted for videos). As seen in Fig. 2, T+S calculates 1) temporal attention for **only** current patch (marked in blue) and the same location patch at different time stamps, and 2) the spatial attention for each patch in each frame. Given a video clip input $X$



