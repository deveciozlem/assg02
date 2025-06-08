import numpy as np
import h5py



def flatten_image_dataset(image_dataset):
    """ Flatten a 4-D tensor which is a dataset of images of the shape
    (num_images, width, height, channels) into a 2-D tensor where each
    image sample has been flattened into a vector.  The resulting shape
    returned should be a 2-D tensor of shape (num_images, (width * height * channels)).
    Do not hardcode or assume any particular number of samples or pixel size or
    channel size for the images.  All of this information should be
    extracted from the input argument.

    Arguments
    ----------
    image_dataset - a 4-D tensor of shape (num_images, width, height, channels)

    Returns
    -------
    image_dataset_flattened - a 2-D tensor of shape (num_images, (width * height * channels)), where
       the row/image samples of the input dataset have been flattened to a 1-D vector.
    """
    n_samples = image_dataset.shape[0]
    n_features = int(np.prod(image_dataset.shape[1:]))
    return image_dataset.reshape(n_samples, n_features)

def standardize_image_dataset(image_dataset):
    """ Standardize a 4-D tensor which is a dataset of images of the shape
    (num_images, width, height, channels)
    This method should first flatten into a 2-D tensor where each
    image sample has been flattened into a vector.  The resulting shape
    returned should be a 2-D tensor of shape (num_images, (width * height * channels)).
    This should be done by reusing the flatten_image_dataset() method.  This
    function should also standardize all pixel values so that they are in the
    range from [0.0 - 1.0] and are now real valued numbers.  It can be assumed that
    the original pixels are RGB integer values in the range [0 - 255]
    Do not hardcode or assume any particular number of samples or pixel size or
    channel size for the images.  All of this information should be
    extracted from the input argument.

    Arguments
    ----------
    image_dataset - a 4-D tensor of shape (num_images, width, height, channels)

    Returns
    -------
    image_dataset_standardized - a 2-D tensor of shape (num_images, (width * height * channels)), where
       the row/image samples of the input dataset have been flattened to a 1-D vector.  Original
       integer RGB pixel values in range [0 - 255] should be standardized to be real valued
       floats in the range [0.0 - 1.0]
    """
    return np.zeros((2,2))



def sigmoid(x):
    """Compute sigmoid of the input parameter x and return. In this version
    the input parameter might be a scalar, but it could be a list or
    a numpy array.  Your implementation should be vectorized and able to
    hanld all of these.

    Arguments
    ---------
    x - a scalar, python list or numpy array of real valued (float/double) numbers.

    Returns
    -------
    s - Result will be of the same shape as the input and will be the element wise
      calculation of the sigmoid for all values given as input.
    """
    return np.zeros((2,))



def initialize_parameters(dim):
    """This function initializes the trainable parameters of our "network".
    The parameter dim gives the input size to the network layer we are
    creating.  The function creates the weight tensor w of shape (dim, 1)
    and a scalar bias b.  All of the weights and biases need to be initialized
    to 0 in this function.

    Arguments
    ---------
    dim - size of the input to this layer

    Returns
    -------
    w - initialized tensor of shape (dim,), with all weights initialized to 0
    b - initialized scalar bias term b, also should be initialized to
    """
    return 0, np.zeros((2,))



def forward_pass(x, w, b):
    """Implement the forward pass of our network "layer".  Given some samples
    x of shape (numsamples, numfeatures), and weights w of shape (numfeatures,)
    and a bias term, perform the affine transformation for the forward pass,
    a = sigmoid(x w + b).  You are required to reuse your sigmoid() function
    here, and as usual you must perform operations using vectorized
    computation, no loops here.

    Arguments
    ---------
    x - The input samples tensor, expected to be a 2-D tensor of shape (samples, features)
    w - The weights trainable parameters, expected to be a 1-D tensor of shape (features,)
    b - The bias trainable parameters, should be a scalar value for this assignment

    Returns
    -------
    a - The output activations / predictions from the forward pass.  Should be 1-d tensor of shape
        (samples,)
    """
    return np.zeros((2,))


def backward_pass(x, y, a):
    """Implement the backard pass of our network "layer".  You need to calculate the summed
    and average cost J.  Also you need to calculate `dw` the gradients of the cost with
    respect to the current weights, and `db` the gradients of the cost with respect to the
    current biases.

    Arguments
    ---------
    x - The input samples tensor, expected to be a 2-D tensor of shape (samples, features)
    y - The true target labels we are training to predict, should shape (samples,)
    a - The output activations / predictions from the layer affine transformation, also of shape (samples,)

    Returns
    -------
    cost - The summed average cost of applying the logit of the true labels and the output activation predictions,
       should be a single scalar result.
    dw - The gradients of the cost with respect to the weights.  Should be a 1-d tensor of shape
       (features,), which are the gradients for each of the input dimensions / features
    db - The gradient of the cost with respect to the bias term.  Should be a simple scalar value.
    """
    return 0.0, np.zeros((2,)), 0.0


def optimize(x, y, w, b, num_iterations=100, learning_rate=0.01, print_cost=False):
    """Implement gradient descent optimization.  In this function a forward pass
    to calculate the output activations / predictions is made, then a backward
    pass is done to compute the current cost and the gradients with respect to the
    learnable parameters w and b.  The learnable parameters are updated using the
    gradient information, scaled by the learning rate.  The cost after each iteration
    of gradient descent is saved.  The function returns the final weights w, biases
    b and list of costs seen during the optimization.

    Arguments
    ---------
    x - The input samples tensor, expected to be a 2-D tensor of shape (samples, features)
    y - The true target labels we are training to predict, should shape (samples,)
    w - The trainable weights parameters, a 1-D tensor of shape (features,)
    b - The trainable biases parameters, a single scalar value
    num_iterations - The number of iterations of gradient descent to perform, defaults to 100
    learning_rate - The scaling factor to multiply the gradients by for performing parameter
       update steps.
    print_cost - A boolean flag that defaults to false, if true a message should be displayed every
       100 iterations of the cost of the form "Cost after iteration %i: %f" % (iteration, cost)

    Returns
    -------
    w - The final updated weights after gradient descent is performed.
    b - The final update bias after gradient descent is performed.
    costs - A history of the costs after each iteration of gradient descent, this is a
       regular python list of values.
    """
    return np.zeros((2,)), 0.0, [0]*100


def predict(x, w, b):
    """Given some (possibly new) inputs x of shape (numsamples, features),
    and a set of trained parameters w and b which constitute some trained
    model, calculate the predictions of the model.  You are required to
    reuse your `forward_pass()` method here.  The predictions should
    be thresholded at 0.5 such that we predict 0 if a <= 0.5 and 1 if
    a > 0.5

    Arguments
    ---------
    x - The input samples tensor, expected to be a 2-D tensor of shape (samples, features)
    w - The trained weight parameters of a model, a 1-D tensor of shape (features,)
    b - The trained bias parameters of a model, a single scalar value

    Returns
    -------
    y_pred - Returns the models predictions threshold at 0.5 for binary prediction.  These
      values need to be returned as float values, you cannot return boolean true/false results.
    """
    return np.zeros((2,))

def load_dataset():
    """Convenience method to load the dataset from this assignment into
    train / test numpy arrays.

    Returns
    --------
    train_x, train_y, test_x, test_y  - numpy arrays with the
      training set inputs and labels, and the test set inputs and labels.
    classes - An array of string values / names of the list of classes
    """
    train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    #train_y = train_y.reshape((1, train_set_y_orig.shape[0]))
    #test_y = test_y.reshape((1, test_set_y_orig.shape[0]))
    
    return train_x, train_y, test_x, test_y, classes

