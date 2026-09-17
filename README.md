# Denoising Autoencoder (DAE)
#### My report covers the entire experiment and background in more depth, please check it out if interested!
  This project examines a denoising autoencoder of a neural network model, designed to reconstruct image data from the Fashion MNIST dataset. While general implementations for denoising autoencoders utilize various prebuilt machine learning libraries in Python, this project focuses on using solely NumPy for the forward pass, backpropagation, and weight updates. The model is evaluated on its ability to recover noisy images applied with Gaussian and Salt and Pepper techniques on unseen data. Through 16 different permutations of hyperparameters, the model achieved a lowest test MSE of 0.0117. Additionally, the model was proven to be highly generalized by having extremely similar train and test MSE.

# The Fashion MNIST Dataset
  The dataset used for this study contains 70,000 28 x 28 grayscale images of the Zalando article images. More specifically, 60,000 training and 10,000 testing images. Each pixel in the image is associated with a value that indicates lightness or darkness (an integer in the range of 0 to 255).
  To denoise these images, the model would need to understand how clothing drapes and shadows are affected by noise. For example, when looking at a section of a dress, the model should recognize that all parts of the clothing are connected without being affected by the pixels corrupted by noise. Additionally, the contrasts that naturally show up in the pixels from the intricacies of clothing is very different than the digits dataset, which typically feature the digit as a solid color and the paper as white.

# Algorithm
  The goal of this algorithm is to train a model that can recognize general features from an image even if it is a corrupted or noisy version of the input. DAEs are a step up from autoencoders as they are forced to learn the structure of the data rather than how autoencoders just copy input to output. The model’s main way of “getting better” is by ignoring noise and filling in the gaps between the missing pixels.

# Experiment Setup
  The main question to be answered is how the hyperparameters of a model affect its ability to rebuild the image as close to the original as possible. More specifically, the hyperparameters that were tested are the size of the hidden dimensions, the learning rate, and the number of epochs or amount of full passes of the training data. Additionally, how the two noise techniques, Gaussian and Salt and Pepper, effect on the final picture was observed. To keep the experiments consist, the same noise of 20% was used for both techniques. All 60,000 training instances and 10,000 testing instances were used.

# Experiment Results

| Exp # |	Noise Type | Learning Rate | Epochs | Hidden Layers | Train MSE | Test MSE |
| -------- | -------- | -------- | -------- |  -------- |  -------- |  -------- | 
| 1	| Gauss	| 0.001	| 50 | 512,256,128 | 0.01305 | 0.0129 |
| 2 | Gauss | 0.001 | 50 | 256,128,64 | 0.01588 | 0.0155 |
| 3	| Gauss	| 0.01	| 50 | 512,256,128 | 0.01104 | 0.0117 |
| 4 | Gauss | 0.01 | 50 | 256,128,64 | 0.01341 | 0.013 |
| 5 | Gauss | 0.001 | 100 | 512,256,128 | 0.01124 | 0.0112 |
| 6 | Gauss | 0.001 | 100 | 256,128,64 | 0.01374 | 0.0136 |
| 7 | Gauss | 0.01 | 100 | 512,256,128 | 0.00967 | 0.0098 |
| 8 | Gauss | 0.01 | 100 | 256,128,64 | 0.01191 | 0.0119 |
| 9 | SP | 0.001 | 50 | 512,256,128 | 0.01568 | 0.0156 |
| 10 | SP | 0.001 | 50 | 256,128,64 | 0.01826 | 0.0183 |
| 11 | SP | 0.01 | 50 | 512,256,128 | 0.01276 | 0.0132 |
| 12 | SP | 0.01 | 50 | 256,128,64 | 0.01507 | 0.015 |
| 13 | SP | 0.001 | 100 | 512,256,128 | 0.01372 | 0.0137 |
| 14 | SP | 0.001 | 100 | 256,128,64 | 0.01607 | 0.0164 |
| 15 | SP | 0.01 | 100 | 512,256,128 | 0.01158 | 0.0119 |
| 16 | SP | 0.01 | 100 | 256,128,64 | 0.01338 | 0.0134 |


# Analysis
  Considering all the permutations of hyperparameters and noise tested on our model, the parameters which resulted in the image that looks closest to the original consisted of 512, 256, and 128 dimensions, a learning rate of 0.01 for both types of noises. The number of epochs 50 or 100 results in very similar outputs and therefore, 50 epochs seems to be a better fit as the model can be trained faster while still learning essential details. 
