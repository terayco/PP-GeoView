# 各预处理、后处理功能说明
## 图像预处理功能说明：
- 1.**直方图匹配**：直方图匹配又称为直方图规定化，是指将一幅图像的直方图变成规定形状的直方图而进行的图像增强方法。即将某幅影像或某一区域的直方图匹配到另一幅影像上。使两幅影像的色调保持一致。
此功能可以减少因季节变化等外部原因造成图像风格差异较大对预测结果带来的影响，[详情参考请点这里](https://blog.csdn.net/mmmmmk_/article/details/82927411)。其效果如下图所示：
  <p align="center">
    <img src="https://user-images.githubusercontent.com/78073130/198608558-7f5a1b9c-c9c6-4686-871d-e22a69b6e235.png" align="middle" width = "600" />
  </p>
- 2.**CLAHE**： 即限制对比度自适应直方图均衡化，其功能是在增强图像对比度的同时限制噪声的放大，适用于图像背景与目标物比较难区分的场景，可以有效提高图像的对比度。[详情参考请点这里](https://blog.csdn.net/qq_43743037/article/details/107195117)其效果图如下所示:
  <p align="center">
    <img src="https://user-images.githubusercontent.com/78073130/198608351-671ef6c5-77c9-408a-a4d8-6cb4417b4293.png" align="middle" width = "600" />
  </p>
- 3.**锐化**：锐化就是通过增强高频分量来减少图象中的模糊，因此又称为高通滤波。锐化处理在增强图象边缘的同时增加了图象的噪声。[详情参考请点这里](https://blog.csdn.net/qq_50559644/article/details/123925265)
## 图像降噪功能说明：
- 1.**平滑**：PP-GeoView为了使用户理解更加方便，用“平滑”这一称呼代替了中值滤波，即二者是等价的。有关中值滤波的[详情请参考这里](https://blog.csdn.net/qq_50559644/article/details/123925265)
- 2.**滤波**：PP-GeoView为了使用户理解更加方便，用“滤波”这一称呼代替了高斯滤波，即二者是等价的。有关高斯滤波的[详情请参考这里](https://blog.csdn.net/qq_50559644/article/details/123925265)
## 变化检测图像后处理说明：
  针对变化检测这一任务的特殊性，我们提供了连通域滤波+建筑物孔洞填充这一操作，用户如果对系统预测结果不满意即可在结果图展示处上方勾选“开启连通域滤波并填充孔洞处理”选项，之后会自动将结果图中的零散伪变化点去除，并填充建筑物内部的孔洞，使结果更加接近真实情况。具体效果如下图所示：
    <p align="center">
      <img src="https://user-images.githubusercontent.com/78073130/198609084-a9b27edb-b6ba-40da-9e14-782a134ddc33.png" align="middle" width = "600" />
    </p>
