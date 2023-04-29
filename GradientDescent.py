from fancy_einsum import einsum
import torch
from tqdm import tqdm

from PIL import Image as img
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data

train_data = torch.zeros(6000, 28, 28).to(device)
train_labels = torch.ones(6000, dtype=torch.long).to(device)
train_labels[2000:] = -1

test_data = torch.zeros((750, 28, 28)).to(device)
test_labels = torch.ones(750, dtype=torch.long).to(device)
test_labels[250:] = -1

print('Loading data...')
for i in tqdm(range(2000)):
    train_data[i] = torch.tensor(np.array(img.open('data/train/A/A_train_' + str(i+1) + '.png'))).to(device)
    train_data[i + 2000] = torch.tensor(np.array(img.open('data/train/B/B_train_' + str(i+1) + '.png'))).to(device)
    train_data[i + 4000] = torch.tensor(np.array(img.open('data/train/C/C_train_' + str(i+1) + '.png'))).to(device)

for i in tqdm(range(250)):
    test_data[i] = torch.tensor(np.array(img.open('data/test/A/A_test_' + str(i+1) + '.png'))).to(device)
    test_data[i + 250] = torch.tensor(np.array(img.open('data/test/B/B_test_' + str(i+1) + '.png'))).to(device)
    test_data[i + 500] = torch.tensor(np.array(img.open('data/test/C/C_test_' + str(i+1) + '.png'))).to(device)
print('Done.')

def normalize(data):
    return (data - data.mean()) / data.std()

train_data = normalize(train_data)
test_data = normalize(test_data)

# define the forward function

def forward(beta : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
    #beta : (28, 28)
    #x : (batch_size, 28, 28)
    #compute the correlation between the image and the model
    product = einsum('b i j, i j -> b', x, beta)
    return product.sign()

# define the loss function

def lin_loss(pred, label):
    return (pred - label) ** 2

def logist_loss(pred, label):
    return torch.log(1 + torch.minimum(torch.exp(-label * pred), torch.ones_like(pred) * 1e16))

# define the gradient function

def lin_grad(pred, label, x):
    #pred and label : (batch_size)
    #x : (batch_size, 28, 28)
    #return the gradient of the beta matrix according to the linear loss

    #pred = sum(beta_ij * x_ij)
    #so dloss/dbeta_ij = 2 * (pred - label) * x_ij

    # we sum along the batch axis
    return einsum('b, b i j -> i j', 2 * (pred - label), x) / x.shape[0]


def logist_grad(pred, label, x):
    #pred and label : (batch_size)
    #x : (batch_size, 28, 28)
    #return the gradient of the beta matrix according to the logistic loss

    #dloss/dbeta_ij = -label * x_ij * e**(-label * pred) / (1 + e**(-label * pred))
    
    return einsum('b, b i j -> i j', -label * torch.sigmoid(-pred * label), x) / x.shape[0]

# initialize the hyperparameters

n = 6000
n1 = 750

lr = 1e-3

batch_size = 6000
epochs = 10000

# train the model

def train(model, loss_function, lr, batch_size, epochs):
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []

    max_accuracy = 0

    x_trainings = []
    x_tests = []

    average_model = torch.zeros_like(model)
    averages = []

    for epoch in tqdm(range(epochs)):
        #shuffle the data
        perm = torch.randperm(n)
        train_data_shuffle = train_data[perm]
        train_labels_shuffle = train_labels[perm]

        #train the model
        for i in range(0, n//batch_size):
            x = train_data_shuffle[i*batch_size:(i+1)*batch_size]
            y = train_labels_shuffle[i*batch_size:(i+1)*batch_size]
            pred = forward(model, x)
            loss = loss_function(pred, y)

            if loss_function == lin_loss:
                grad = lin_grad(pred, y, x)
            else:
                grad = logist_grad(pred, y, x)
            
            model = model - lr * grad

            training_loss.append(torch.mean(loss).item())

            #compute the accuracy for the pred :
            #the prediction is just the sign of the correlation ("pred")

            pred = torch.sign(pred)
            training_accuracy.append(torch.mean((pred == y).float()).item())

            x_trainings.append(i + epoch * n // batch_size)
        
        #test the model
        pred = forward(model, test_data)
        loss = loss_function(pred, test_labels)

        test_loss.append(torch.mean(loss).item())

        pred = torch.sign(pred)
        test_accuracy.append(torch.mean((pred == test_labels).float()).item())
        max_accuracy = max(max_accuracy, test_accuracy[-1])

        x_tests.append(epoch * n // batch_size)

        average_model = epochs / (epochs + 1) * average_model + 1 / (epochs + 1) * model

        if epoch in [0, 9, 99, 999, 9999]:
            averages.append(average_model)
        
        if max_accuracy > 1.05*test_accuracy[-1]:
            break
    
    return model, training_loss, training_accuracy, test_loss, test_accuracy, x_trainings, x_tests, averages


class Question_3():
    def __init__(self, n = 6000, n1 = 750, lr = 5e-8, batch_size = 6000, epochs = 10000) -> None:
        self.n = n
        self.n1 = n1
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
    
    def train(self, problem = "logist", lr = 1e-7, batch_size = 6000, epochs = 10000):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        model = torch.randn((28, 28)).to(device)
        model /= torch.norm(model)

        print("Training the model...")
        if problem == "logist":
            loss_function = logist_loss
        else:
            loss_function = lin_loss
        
        self.beta, self.training_loss, self.training_accuracy, self.test_loss, self.test_accuracy, self.x_trainings, self.x_tests, self.averages = train(model, loss_function, lr, batch_size, epochs)
        
        print("empirical error : ", 1 - self.training_accuracy[-1])
        print("test error : ", 1 - self.test_accuracy[-1])
        
        print("Done.")

    def a(self):
        #plot the loss and the accuracy on two different graphs
        plt.plot(self.x_trainings, self.training_loss, label = "training loss")
        plt.plot(self.x_tests, self.test_loss, label = "test loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss according to the epoch")
        plt.legend()

        plt.loglog()
        plt.show()
        
        plt.plot(self.x_trainings, self.training_accuracy, label = "training accuracy")
        plt.plot(self.x_tests, self.test_accuracy, label = "test accuracy")

        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy according to the epoch")
        plt.legend()

        plt.show()

    def b(self):
        plt.imshow(self.beta.cpu().detach().numpy())
        plt.title("Learned model")
        plt.show()
    
    def c(self):
        fig, ax = plt.subplots(1, 5)
        for i, beta in enumerate(self.averages):
            ax[i].imshow(beta.cpu().detach().numpy())
        plt.title("average model at epochs 1, 10, 100, 1000, 10000")
        plt.show()

    def do(self, problem = "logist", lr = 1e-7, batch_size = 6000, epochs = 10000):
        self.train(problem, lr, batch_size, epochs)
        self.a()
        self.b()
        self.c()

question_3 = Question_3()

question_3.do("logist", lr = 1e-3, batch_size = 6000, epochs = 10000)
question_3.do("logist", lr = 1e-3, batch_size = 1, epochs = 1)

question_3.do("lin", lr = 1e-3, batch_size = 6000, epochs = 10000)
question_3.do("lin", lr = 1e-3, batch_size = 1, epochs = 1)