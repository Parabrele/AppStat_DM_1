"""
This homework is due by May 1st, 2022. It is to be returned by email to ulysse.marteau-ferey@inria.fr as a pdf
report of maximum 3 pages together with the ipython notebook used for the code. The results and the figures
must be included into the pdf report but not the code.
The goal of this project is to automatically classify letters from different computer fonts. An example of samples
of the letter “A” can be seen below
"""

"""
The data comes from the notMNIST dataset and can be downloaded at http://www.di.ens.fr/appstat/spring-2022/
project/data.zip. The zip archive contains two folders:
– train: contains n = 6 000 labelled images of three classes “A”, “B” and “C” (2 000 each)
– test: contains n1 = 750 labelled images (250 for each of the three classes).
The train folder will be used to train the forecasting methods. The test folder will be used to assess their performance. If for some reasons, the datasets are too large to be used on your computer, you can use subsets of with n
and n1 sufficiently small to be computable but large enough to get prediction accuracy.
The goal is to classify if an image Xi corresponds to the letter “A”: i.e., the output is Yi = 1 if image i is “A” and
−1 otherwise (if the image is “B” or “C”).
"""

"""
###############
#Question 1 : #
###############

formalize the problem by defining the input space X , the output space Y and the training data set. What are their dimension?

Answer:
X is the set of black and white images of size 28x28 pixels. Y is the set of labels {-1,1} corresponding to the letter A or not.
"""

"""
###############
#Question 2 : #
###############

If f : X → Y is a predictor from images to Y = {−1, 1}, we define for a couple image/label (Xi, Yi):
– the 0-1 loss: L`1(f(Xi), Yi) = 1f(Xi)6=Yi
– the square loss: L`2(f(Xi), Yi) = f(Xi) − Yi ** 2
– the logistic loss: L`3(f(Xi), Yi) = log(1 + e**(−Yif(Xi)))


(a) What are the empirical risk (training error) and the true risk associated with the 0-1 loss? Why is it
complicated to minimize the empirical risk in this case ?

Answer:
The empirical risk is the average of the 0-1 loss over the training set. The true risk is the average over the whole input space following the distribution by which the training set is sampled.
It is complicated to minimize the empirical risk because it is not differentiable and thus we cannot use gradient descent.


(b) Why should we use the test data to assess the performance ?

Answer:
We should use the test data to assess the performance because it is the only way to estimate the true risk of the predictor with its empirical risk.
Indeed, as we train only on the training set, there might be an overfitting problem, and the empirical risk might be very low, but the true risk might be high because the predictor is not general enough.


(c) Recall the definition of the optimization problems associated with the linear least square regression and
the linear logistic regression

Answer:
Linear least square regression:
minimize 1/n * sum((f(Xi) - Yi)**2)

Linear logistic regression:
minimize 1/n * sum(log(1 + e**(-Yif(Xi))))

In both this problems, we consider f as a linear function from R^(28*28) to R.
Thus, we can consider that we want to find an image of size 28*28 pixels, that will be used to compute the corellation between it and the input image.

My hypothesis at this point is that the resulting function image will resemble an A, and will probably be a somewhat average of the A images in the training set, maybe with mystical values in places where there might be confusion with B or C images.
This would be the most satisfying result, but another possibility is that the resulting image will be composed of zeros everywhere except for a few high values where A images have almost always high values and not B and C, and highly negative values in the opposite situation.
Thus the image might turn out to be a black image with a few bright spots.
"""

"""
###############
#Question 3 : #
###############

Implement the gradient descent algorithm (GD) and the stochastic gradient descent algorithm (SGD) to solve
these two minimization problem

(a) Consider the logistic regression minimization problem. Plot the training errors and the test errors as
functions of the number of access to the data points1 of GD and SGD for well-chosen (by hand) values of
the step sizes.
(b) Plot the estimators beta_hat_n^(logist) in R ^(28*28) and beta_hat_n^(lin) in R ^(28*28) respectively associated with the logistic and linear
regression as two images of size 28 × 28.
(c) Denote by beta_hat_n^(logist) (t) in R ^(28*28) the estimator of logistic regression after t gradient iterations of SGD. Plot
as images the averaged estimators beta_hat_n^(logist) (t) = 1/t * sum(beta_hat_n^(logist) (s)) for s = 1, . . . , t, for t = 10, 100, 1000, 10000.
Repeat for the linear regression estimator.
"""

from fancy_einsum import einsum
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data

train_data = torch.load('data/train_data.pt').to(device)
train_labels = torch.ones(6000, dtype=torch.long).to(device)
train_labels[2000:] = -1

test_data = torch.load('data/test_data.pt').to(device)
test_labels = torch.ones(750, dtype=torch.long).to(device)
test_labels[250:] = -1

# define the forward function

def forward(beta : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
    #beta : (28, 28)
    #x : (batch_size, 28, 28)
    #compute the correlation between the image and the model
    return einsum('ij, bij -> b', beta, x)

# define the loss function

def lin_loss(pred, label):
    return (pred - label) ** 2

def logist_loss(pred, label):
    return torch.log(1 + torch.exp(-label * pred))

# define the gradient function

def lin_grad(pred, label, x):
    #pred and label : (batch_size)
    #x : (batch_size, 28, 28)
    #return the gradient of the beta matrix according to the linear loss

    #pred = sum(beta_ij * x_ij)
    #so dloss/dbeta_ij = 2 * (pred - label) * x_ij
    return 2 * (pred - label) * x


def logist_grad(pred, label, x):
    #pred and label : (batch_size)
    #x : (batch_size, 28, 28)
    #return the gradient of the beta matrix according to the logistic loss

    #dloss/dbeta_ij = -label * x_ij * e**(-label * pred) / (1 + e**(-label * pred))
    return -label * torch.exp(-label * pred) / (1 + torch.exp(-label * pred)) * x

# initialize the hyperparameters and the model (it is just a 28*28 matrix) following a normal distribution

n = 6000
n1 = 750

lr = 0.01
epochs = 1
batch_size = 6000

beta = torch.randn((28, 28))
beta.to(device)

# train the model

def train(loss_function):
    training_loss = []
    test_loss = []

    x_trainings = range(0, epochs * n // batch_size)
    x_tests = range(0, epochs).map(lambda x : x * n // batch_size)

    for epoch in tqdm(range(epochs)):
        #shuffle the data
        perm = torch.randperm(n)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

        #train the model
        for i in range(0, n//batch_size):
            x = train_data[i*batch_size:(i+1)*batch_size]
            y = train_labels[i*batch_size:(i+1)*batch_size]
            pred = forward(beta, x)
            loss = loss_function(pred, y)

            if loss_function == lin_loss:
                grad = lin_grad(pred, y)
            else:
                grad = logist_grad(pred, y)
            
            beta = beta - lr * grad

            training_loss.append(loss)
        
        #test the model
        pred = forward(beta, test_data)
        loss = loss_function(pred, test_labels)

        test_loss.append(loss)
    
    return beta, training_loss, test_loss, x_trainings, x_tests

lin_beta, lin_training, lin_test, lin_x_trainings, lin_x_tests = train(lin_loss)
logist_beta, logist_training, logist_test, logist_x_trainings, logist_x_tests = train(logist_loss) 

