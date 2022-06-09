# Landmark Detection Project

## Implementing a CNN from scratch in PyTorch with custom weight initialization
For this project, we built a convolutional neural network (CNN) that, given an image of a landmark predicts which landmark it is. For example, if the input image is a photo of The Eiffel Tower, our classifier will classify it as "Eiffel Tower". For this project, we first transform the images and implement the neural network using **PyTorch**. Specifically, we use the DataLoader class for creating datasets and iterating over the batches and we use ```nn``` class for implementing the network. The architecture of the network is as follows: 

![Alt text](nn.png?raw=true "Title")

We compared the default weight initialization of PyTorch and the custom weight initizalization method that we implemented, which initializes the weights with values drawn from a normal distribution of mean 0 and variance $\frac{1}{\sqrt{n}}$, where $n$ is the number of inputs to the layer that the weights belongs to. We observed that this weight initialization strategy **outperforms** the default weight initialization strategy by comparing the loss values.

We then evaluated the test accuracy of our model with default initialization and observed 38% accuracy. The threshold for this value was 20%. Our model almost doubled the test accuracy threshold! Still not a good accuracy value. 

## Implementing transfer learning

We decided to utilize transfer learning since this seems to be a hard task. We chose to utilize a pre-trained ResNet50 model from ```torchvision``` library. We train this model for 10 epochs for it to learn our dataset a little bit and then we observe that for this 50-class classification problem we get 72% accuracy. Not so bad! 