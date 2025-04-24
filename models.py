import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # Returns the score which is the dot product of the weights and the input data point x 
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # The perceptron predicts the class based on the sign of the score
        score = self.run(x)
        # return the predicted class which is 1 if the score is greater than 0, -1 otherwise
        return nn.Constant(1) if nn.as_scalar(score) > 0 else nn.Constant(-1)
    

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Initialize the convergence flag
        converged = False
        # Continue training until convergence is reached 
        while not converged:                    
            converged = True          # Set the convergence flag to True
            # Iterate through the dataset 
            for x, y in dataset.iterate_once(1):
                # Get the predicted class
                prediction = self.get_prediction(x)
                # If the predicted class is not equal to the true class, update
                # the weights
                if nn.as_scalar(prediction) != nn.as_scalar(y):
                    converged = False
                    # Update the weights
                    self.w.update(x, nn.as_scalar(y))
                    # print("Updated weights: ", self.w)
                    break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # regression model to approximate sin(x)
        # Define the weights and biases
        self.hidden_size = 512
        self.learning_rate = 0.05

        # Layer 1: Input (1) → Hidden (512)
        self.w1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Layer 2: Hidden (512) → Output (1)
        self.w2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Hidden layer: Linear → ReLU
        h = nn.Linear(x, self.w1)       # Linear transformation 
        h_bias = nn.AddBias(h, self.b1) # Add bias 
        h_relu = nn.ReLU(h_bias)        # Apply ReLU activation function 

        # Output layer: Linear
        y_pred = nn.Linear(h_relu, self.w2)         # Linear transformation
        # Add bias to the output layer
        y_pred_bias = nn.AddBias(y_pred, self.b2)
        # Return the predicted y-values
        return y_pred_bias
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get the predicted y-values 
        pred = self.run(x)
        # Calculate the loss using the square loss function and return it
        return nn.SquareLoss(pred, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # for epoch in range(2000): # Iterating through the dataset for 2000 epochs the final loss of this model was more than 0.02
        for epoch in range(20000): # Iterating through the dataset for 20000 epochs, due to above increased the number of epochs, which reduced the final loss below the limit of 0.02
            # Iterate through the dataset in batches of size 200  
            for x, y in dataset.iterate_once(200):  # Batch size = 200
                # Get the loss for the current batch
                loss = self.get_loss(x, y)
                # Calculate the gradients of the loss with respect to the model parameters 
                grads = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                # Update the model parameters using the gradients and the learning rate 
                self.w1.update(grads[0], -self.learning_rate)
                self.b1.update(grads[1], -self.learning_rate)
                self.w2.update(grads[2], -self.learning_rate)
                self.b2.update(grads[3], -self.learning_rate)



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
         # Model Parameters
        self.hidden_size = 256
        self.learning_rate = 0.1

        # Input layer (784) -> Hidden layer
        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        # Hidden layer -> Output layer (10)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
         # First layer (Linear + ReLU)
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        # Output layer (Linear only)
        scores = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return scores
    

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get the predicted scores
        scores = self.run(x)
        # Calculate the loss using the softmax loss function and return it
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Iterate through the dataset for 10 epochs 
        for epoch in range(10):
            # Iterate through the dataset in batches of size 100 
            for x, y in dataset.iterate_once(100):
                # Get the loss for the current batch 
                loss = self.get_loss(x, y)
                # Calculate the gradients of the loss with respect to the model parameters 
                grads = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                # Update the model parameters using the gradients and the learning rate 
                self.w1.update(grads[0], -self.learning_rate)
                self.b1.update(grads[1], -self.learning_rate)
                self.w2.update(grads[2], -self.learning_rate)
                self.b2.update(grads[3], -self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Model Parameters 
        self.hidden_size = 128
        self.learning_rate = 0.2

        # RNN weights
        self.w_input = nn.Parameter(self.num_chars, self.hidden_size)
        self.w_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_hidden = nn.Parameter(1, self.hidden_size)

        # Output layer
        self.w_output = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
         # Initialize h = 0 for the first timestep
        h = None
        # Iterate through the characters in the word 
        for x in xs:
            # Apply the input layer (Linear) to the current character 
            x_proj = nn.Linear(x, self.w_input)
            # If h is None, this is the first character, so we initialize h 
            if h is None:
                h = nn.AddBias(x_proj, self.b_hidden)
            # Otherwise, we apply the hidden layer (Linear + ReLU) to the current character and the previous hidden state 
            else:
                h = nn.AddBias(nn.Add(nn.Linear(h, self.w_hidden), x_proj), self.b_hidden)
            # Apply the ReLU activation function to the hidden state 
            h = nn.ReLU(h)

        # Final output (after last character)
        output = nn.AddBias(nn.Linear(h, self.w_output), self.b_output)
        # Return the predicted scores 
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Get the predicted scores 
        scores = self.run(xs)
        # Calculate the loss using the softmax loss function and return it 
        return nn.SoftmaxLoss(scores, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Iterate through the dataset for 15 epochs 
        for epoch in range(15):
            # Iterate through the dataset in batches of size 100 
            for xs, y in dataset.iterate_once(100):
                # Get the loss for the current batch 
                loss = self.get_loss(xs, y)
                # Calculate the gradients of the loss with respect to the model parameters 
                grads = nn.gradients(loss, [
                    self.w_input, self.w_hidden, self.b_hidden,
                    self.w_output, self.b_output
                ])
                # Update the model parameters using the gradients and the learning rate 
                self.w_input.update(grads[0], -self.learning_rate)
                self.w_hidden.update(grads[1], -self.learning_rate)
                self.b_hidden.update(grads[2], -self.learning_rate)
                self.w_output.update(grads[3], -self.learning_rate)
                self.b_output.update(grads[4], -self.learning_rate)