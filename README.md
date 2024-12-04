# S<sup>3</sup>AM-Net
**S<sup>3</sup>AM: A Spectral-Similarity-Based Spatial Attention Module for Hyperspectral Image Classification** [Source](https://ieeexplore.ieee.org/document/9832463)  
Authors: Ningyang Li, Zhaohui Wang, Faouzi Alaya Cheikh, Mohib Ullah  
Journal: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing  
Environment: Python 3.6., Tensorflow 2.2.2, Keras 2.3.1, Numpy 1.19.  

**Abstract:**  
Recently, hyperspectral image (HSI) classification based on deep learning methods has attracted growing attention and made great progress. Convolutional neural networks based models, especially the residual networks (ResNets), have become the architectures of choice for extracting the deep spectral-spatial features. However, there are generally some interfering pixels in the neighborhoods of the center pixel, which are unfavorable for the spectral-spatial feature extraction and will lead to a restraint classification performance. More important, the existing attention modules are weak in highlighting the effect of the center pixel for the spatial attention. To solve this issue, this article proposes a novel spectral-similarity-based spatial attention module (S<sup>3</sup>AM) to emphasize the relevant spatial areas in HSI. The S<sup>3</sup>AM adopts the weighted Euclidean and cosine distances to measure the spectral similarities between the center pixel and its neighborhoods. To alleviate the negative influence of the spectral variability, the full-band convolutional layers are deployed to reweight the bands for the robust spectral similarities. Both kinds of weighted spectral similarities are then fused adaptively to take their relative importance into full account. Finally, a scalable Gaussian activation function, which can suppress the interfering pixels dynamically, is installed to transform the spectral similarities into the appropriate spatial weights. The S<sup>3</sup>AM is integrated with the ResNet to build the S<sup>3</sup>AM-Net model which is able to extract the discriminating spectral-spatial features. Experimental results on four public HSI data sets demonstrate the effectiveness of the proposed attention module and the outstanding classification performance of the S<sup>3</sup>AM-Net model.

**Contibutions:**  
1. A novel S<sup>3</sup>AM is proposed to capture the relevant spatial areas effectively in the HSI cube. Specifically, the WED and WCD sub-modules, which adopt the FBC layers to relieve the adverse influence of the spectral variability, are first applied to improve the robustness of the spectral similarities. Both weighted spectral similarities are then integrated adaptively to gain the representative composite spectral similarity. Finally, an SG activation function is designed to convert the spectral similarities to the appropriate spatial weights flexibly in diverse scenes. The S<sup>3</sup>AM excels at emphasizing the spatial areas relevant intensively to the center pixel and preserving these crucial areas even in a wider HSI cube.
2. An end-to-end S<sup>3</sup>AM-Net model, which contains the S<sup>3</sup>AM and ResNet, is designed to obtain the discriminating features for HSI classification. With the support of the functional S<sup>3</sup>AM, this model is capable of handling the spatial features as well as the spectral-spatial features efficiently.

<img src="https://github.com/ningyang-li/S3AM-Net/blob/8102cc5ac219c6b53dbca452073ab5252acbb73f/pic/S3AM.png" width="500" />  
<img src="https://github.com/ningyang-li/S3AM-Net/blob/8102cc5ac219c6b53dbca452073ab5252acbb73f/pic/WED.png" width="500" />  
<img src="https://github.com/ningyang-li/S3AM-Net/blob/8102cc5ac219c6b53dbca452073ab5252acbb73f/pic/WCD.png" width="500" />  
<img src="https://github.com/ningyang-li/S3AM-Net/blob/8102cc5ac219c6b53dbca452073ab5252acbb73f/pic/SG.png" width="500" />  
<img src="https://github.com/ningyang-li/S3AM-Net/blob/8102cc5ac219c6b53dbca452073ab5252acbb73f/pic/Net.png" width="500" />  

**Citation:**  
N. Li, Z. Wang, F. A. Cheikh and M. Ullah, "S3AM: A Spectral-Similarity-Based Spatial Attention Module for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 5984-5998, 2022, doi: 10.1109/JSTARS.2022.3191396.

<code>@ARTICLE{9832463,
  author={Li, Ningyang and Wang, Zhaohui and Cheikh, Faouzi Alaya and Ullah, Mohib},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={S3AM: A Spectral-Similarity-Based Spatial Attention Module for Hyperspectral Image Classification}, 
  year={2022},
  volume={15},
  number={},
  pages={5984-5998},
  keywords={Feature extraction;Convolutional neural networks;Correlation;Data mining;Residual neural networks;Kernel;Euclidean distance;Center pixel;hyperspectral image classification;residual network;spatial attention;spectral similarity},
  doi={10.1109/JSTARS.2022.3191396}}
</code>
