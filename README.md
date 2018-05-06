# Style-transfer-with-neural-algorithm

The above code is a simple implementation with TensorFlow of the paper， this paper [Image Style Transfer Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), which is published in CVPR2016. The idea of one image's style transfer to another image is really cool, and it is very easy to implement with TensorFlow. Meanwhile, training time for one image just cost a few minutes.

Method
-------

![algorithm](https://github.com/MingtaoGuo/Style-transfer-with-neural-algorithm/raw/master/method/method.jpg)

We can see the image above that is from the paper, it shows a simple way to synthesize an image from other style.In this method, x is the variable which we want to update, and it is also an synthesized image as the final result. The squared error is used to control the content which makes the synthesized image is similar to the original content image, and the Gram matrix is used to control the style which makes the synthesized image has the similar style with original style image.
