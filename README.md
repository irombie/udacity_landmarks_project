# Landmark Detection Project

For this project, we built a convolutional neural network (CNN) that, given an image of a landmark predicts which landmark it is. For example, if the input image is a photo of The Eiffel Tower, our classifier will classify it as "Eiffel Tower". For this project, we first transform the images and implement the neural network using **PyTorch**. Specifically, we use the DataLoader class for creating datasets and iterating over the batches and we use ```nn``` class for implementing the network. The architecture of the network is as follows: 
[](nn.png)