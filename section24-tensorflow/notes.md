# Deep Learning

## Perceptron model

- modeled after a biological perceptron 
- dendrites, the "input" 
- Axon, the "output"
- Nucleus, calculation / transformation

- Mathematically a simple perceptron is:
$$ {\hat{y} = \sum_{i=1}^{n} x_{i}w_{i} + b_{i}} $$


## Nueral Networks
- To build a network of perceptrons, we can connect layers of perceptrons using a multi-layer perceptron model.
- first layer is the input layer
- last layer is the output layer, can be more than one neruon. Estimates the final value.
- any layers in between the input and output are known as hidden layers
- hidden layers are difficult to interpret, due to their high interconnectivity and distance away from known input or output values
- Neural networks become **Deep Neural Networks** if they have two or more hidden layer
- What makes this incredible is that it can be used to approximate any function
- This has been proven mathematically by Zhou lu and Boris Hanin.
- If classification tasks it would be useful if call outputs fell between 1 and 0
- These values can then present probability assignments to each class.
- In the next lecture, we'll explore how to use activation functions to set boundaries to output values from the neuron 

## Activation Functions
- Recall that inputs x have a weight w and a bais b attached to them in the perceptron model
- Which means we have $$ x*w + b $$
- Clearly w implies how much weight or strength we should assign to the incoming input. 
- We can think of b as an offset value, making x*w reach a certain threshold before having an effect. 
- example: b = -10
- Then the effects of x*w won't start to overcome the bias until their product surpasses 10, after that the effect is solely based on the value of y
- Total 
- Common activation functions:
    - The most simple networks rely on a basic step function that outputs 1 or 0
    - Regardless of values, always outputs 1 or 0
    - Can be useful for classification
    - Is a strong function, since small changes are not reflected
    - Would be nice if we could have a dynamic function, example the red line
    - Also known as the sigmoid function
    $$ f(z) = { 1 \over 1 + e^(-z)} $$
    - still worksf for classification but is more sensitive to small changes 
    - **Hyperbolic Tangent**: tanh(z)
        - cosh x = $$ {e^x + e^-x \over 2} $$
        - sinh x = $$ {e^x - e^-x \over 2} $$
        - tanh x = $$ {sinh x \over cosh x} $$
    - **Rectified Linear Unit (ReLU)**: This is actually a relatively simple function: max(0,z)
        - has been found to have a very good performance, especially when dealing with the issue of vanishing gradient
        - We'll often default to Relu due to it's overall good performance.

## Multi-class activation functions
- previous activation functions mentioned make sense for a single output either a binary classification or trying to predict a binary classication
- Two main types of multi class situations:
    - Non-exclusive Classes, a single data point can have multiple classes assigned to it. 
    - Mutually Exclusive classes, only one class assigned.
- Organizing multiple classes:
    - This means we will not need to organize categories for this output layer
    - We can't just have categories like red, blue, green, ect...'
- Choosing the correct classification activation function:
    - Non-exclusive: Sigmoid Function
        - Each neuron will output a value between 0 and 1, indicating the probability of having that class assigned to it. 
    - Mutually Exclusive Classes
        - We can use a **softmax** function for this.
    $$ {o(z)_{i}} = {e^{z_{i}}\over \sum_{j=1}^k e^{z_{j}}} $$
    - The range will be 0 - 1 and teh sum of all the probabilities will be equal to one.
    - The model returns the proabilities of each class and the target class chosen will have the highest probability. 
    - The main thing to keep in mind is htat if you use softmax for multi-class problems you get this sort of output ['red', 'blue', 'green'], [.1, .6, .3]

## Cost Functions and Gradient Descent 
- cost functions: how far we are off on the output prediction
- gradient decent, how to minimize that prediction error
- This output y^ is the models estimation of what it predicts the label to be.
- So after the network creates its prediction, how do we evaluate it?
- And after evaluation how can we update the networks weights and biases?
    - We to take the estimates of outputs of the network and then compare them to the real values of the label
    - Keep in mind this is using the training data set during the fitting/training of the model 
    - The cost function (or loss function) must be an average so it can output a single value
    - We can keep track of our loss / cost during training to monitor performance.
- One of the most common is the quadratic cost function:
    $$ c = {1 \over 2n}{\sum_{x}|y(x) - a^L(x)|^2} $$
    - we simply calculate the difference between the real values y(x) against our predicted values of a(x)
    - this notation refers to vector inputs and outputs (arrays), since we will be dealing with a batch of training points and predictions
    - notice how squaring this does 2 useful things, keeps everything positive, and punishes large errors.
    - in real case, this means we have 


## Back Propagation
- main idea is that we can use the gradient to go back through the network and adjust our weights and biases to minimize the output of the error vector on the last output layer. 

## Training (Keras)

Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

* Sample: one element of a dataset.
    * Example: one image is a sample in a convolutional network
    * Example: one audio file is a sample for a speech recognition model
* Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model.A batch generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).
* Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
* When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.
* Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples of these are learning rate changes and model checkpointing (saving).