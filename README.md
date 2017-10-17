# End-to-End Learning for Visual Navigation
By [Raymond Chua](https://github.com/raymondchua), [Natalie Reppekus](https://github.com/Natalie1993) and [Ute Schiehlen](https://github.com/uteschiehlen/)

Deep Learning Course Project at Technical University of Munich.

### Table of Contents
1. Introduction
2. Data Preprocessing
3. Our Approach
4. Training and Evaluation
6. Test Results
7. Poster Presentation at Technical University of Munich.
8. References


### Introduction

This repository contains the code, proposal and the poster for our project, "End to end Learning for Visual Navigation". This is for the "Deep Learning for Computer Vision" course offered in Summer 2017 at the Technical University of Munich. 

We attempted to solve a supervised learning problem for autonomous driving. The training data consisted of RGB image sequences with the steering angles as training labels. The project is motivated by Nvidia's paper, "End to end Learning for Self Driving Cars." We considered additional data in the form of Optical Flow which improved the training, however the performance on the test set was marginal. In addition, we considered using a LSTM cell to learn about dependencies in the sequence. This model caused faster convergence during the training phase and resulted in a 0.12 improvement of the mean squared error in the test set. The error improved from 11 degrees from the Nvidia's model to 4 degrees in our LSTM based model. 

### Data Preprocessing
We performed data augmentation during the preprocessing phase. At first a random horizontal flip was applied to the RGB images and the corresponding steering angle multiplied with -1. We then normalized the image to [0,1] by dividing the pixel values with 255. In order to encode human perception better, the RGB images were then converted to YUV colorspace. Afterwards, the YUV images were zero-mean centered by subtracting their respective mean values.

### Our Approach
![Network Architecture](https://github.com/uteschiehlen/dlcv_visnav/blob/master/poster/images/modelOpt.png "Nvidia, Nvidia + Optical Flow, CNN + Optical Flow + LSTM")

We first build the model based on Nvidia's paper. However, during training, we observed that the loss converged to a value of 2.5. We argue that this is due to the lack of complexity of Nvidia's model. We then added a sub-network with Optical Flow as inputs, which resulted in a near zero training loss convergence. However, the improvement on the test set remained marginal. We then build a LSTM based model by concatenting the features of the YUV images as well as the Optical Flow data and used them as inputs for the LSTM cell. The whole network was trained end to end in order to fully exploit the data. All networks were trained using mini-batch Stochastic Gradient Descent with a learning rate of 0.001 over 200 epochs. We used a Mean Square Error (MSE) as our loss function. In our experiments, we have shown that the LSTM model converges faster during training and achieves a better performance in the testing phase. 

### Training and Evaluation
![Training and Evaluation](https://github.com/uteschiehlen/dlcv_visnav/blob/master/poster/images/train_eval_v7.png "Training and Validation Loss")

For the Nvidia and CNN+Optical Flow network, we considered a balanced subset of the training data in order to reduce the number of samples where the car is driving straight. However this was not applied to the LSTM model due to the requirement of having sequential images. The training for the Nvidia and Optical Flow model converged in about 140 epochs. The LSTM model needed only 80 epochs to converge. Since the validation set had similar environment settings to the training set, it is unsurprising that the loss value is similar to their respective training loss. 

### Test Results
| Model        	| Nvidia        | Nvidia + Optical Flow | Nvidia + Optical Flow + LSTM |
| ------------- |:-------------:| :--------------------:| :---------------------------:|
| MSE     		| 0.1911 		| 0.1910				|0.072						   |

The values above are in radians. 0.19 is about 11 degrees and 0.072 is about 4 degrees.



### Poster Presentation at Technical University of Munich.
![Poster](https://github.com/uteschiehlen/dlcv_visnav/blob/master/poster/dl4cv_latex_postertemplate/final_poster.png "Poster Presentation")

### References
1. Udacity Self-driving-car dataset. https://github.com/udacity/self-driving-car/tree/master/datasets/CH2. Accessed: 2017-06-26.1
2. M. Bojarski, D. D. Testa, D. Dworakowski, B. Firner,B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller,J. Zhang, X. Zhang, J. Zhao, and K. Zieba. End to end learning for self-driving cars. CoRR, abs/1604.07316, 2016. 1,2
3. S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997. 1
4. J. Y. Ng, M. J. Hausknecht, S. Vijayanarasimhan, O. Vinyals, R. Monga, and G. Toderici. Beyond short snippets: Deep networks for video classification. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 4694–4702, 2015.1
5. M. Oquab, L. Bottou, I. Laptev, and J. Sivic. Is object localization for free? - weakly-supervised learning with convolutional
neural networks. In CVPR, pages 685–694. IEEE Computer Society, 2015. 1
6. P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. CoRR, abs/1312.6229, 2013. 1
7. E. Shelhamer, J. Long, and T. Darrell. Fully convolutional networks for semantic segmentation. IEEE Trans. Pattern Anal. Mach. Intell., 39(4):640–651, 2017. 1
8. K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pages 568–576, 2014. 1