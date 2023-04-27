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
    return (data - torch.mean(data)) / torch.std(data)

train_data = normalize(train_data)
test_data = normalize(test_data)

def knn(k, train_data, train_labels, test_data, test_labels):
    train_data_reshaped = train_data.reshape((train_data.shape[0], 28 * 28))
    test_data_reshaped = test_data.reshape((test_data.shape[0], 28 * 28))

    #compute the distances between each test point and each train point
    distances = torch.cdist(test_data_reshaped, train_data_reshaped)

    #find the k closest points for each test point
    closest = torch.argsort(distances, dim = 1)[:, :k]

    #find the labels of the k closest points
    closest_labels = train_labels[closest]

    #find the majority label for each test point
    majority_labels = torch.mode(closest_labels, dim = 1)[0]

    #compute the error
    error = torch.sum(majority_labels != test_labels).item() / 750

    return error

k_errors = torch.zeros(10)
ks = torch.arange(1, 11)
for k in range(1, 11):
    err = knn(k, train_data, train_labels, test_data, test_labels)
    print('k =', k, ':', err)
    k_errors[k - 1] = err

plt.plot(ks, k_errors)
plt.xlabel('k')
plt.ylabel('error')
plt.title('Error as a function of k')
plt.show()

#now implement k-fold cross validation to find the best k
#description of the algorithm :
#first, we shuffle the data
#then, we split the data into K folds
#then, for each fold, we train the model on the other folds and test it on the current fold
#finally, we average the errors on the K folds and return the best k

def K_fold_cross_validation(K, train_data, train_labels, test_data, test_labels):
    #first, shuffle the data
    perm = torch.randperm(6000)
    shuffled_train_data = train_data[perm]
    shuffled_train_labels = train_labels[perm]

    #split the data into K folds
    folds = torch.split(shuffled_train_data, 6000 // K)
    folds_labels = torch.split(shuffled_train_labels, 6000 // K)

    #now, for each fold, train the model on the other folds and test it on the current fold
    k_list = torch.arange(1, 11)
    errors = torch.zeros(K, 10)
    for i in range(K):
        #first, we concatenate all the folds except the current one
        train_data = torch.cat(folds[:i] + folds[i + 1:])
        train_labels = torch.cat(folds_labels[:i] + folds_labels[i + 1:])

        #then, we test the model on the current fold
        test_data = folds[i]
        test_labels = folds_labels[i]

        #compute the error
        for k in range(1, 11):
            errors[i, k - 1] = knn(k, train_data, train_labels, test_data, test_labels)

    #finally, we average the errors on the K folds and return the best k
    errors = torch.mean(errors, dim = 0)
    best_k = torch.argmin(errors) + 1

    return best_k.item()

best_k = K_fold_cross_validation(5, train_data, train_labels, test_data, test_labels)
print('\nbest k :', best_k)


def kmeans(k, train_data, train_labels, test_data, test_labels):
    #reshape the images into vectors
    train_data = train_data.reshape((6000, 28 * 28))
    test_data = test_data.reshape((750, 28 * 28))

    #initialize the k points
    k0 = torch.randint(0, 6000, (1,)).item()
    k0 = train_data[k0]

    k_stack = torch.zeros((k, 28 * 28)).to(device)
    k_stack[0] = k0

    for i in range(k - 1):
        distances = torch.cdist(train_data, k_stack[:i + 1])
        
        #first, find the minimum distance between each point and the k points we already have
        #this gives the distance from each point to the closest point among the k points we already have
        #then, find the maximum of these minimums
        #this is the furthest point from the k points we already have

        min_distances = torch.min(distances, dim = 1)[0]
        #we take the [0] because the function returns a tuple with the values and the indices, we only want the values
        max_index = torch.argmax(min_distances)

        k_stack[i + 1] = train_data[max_index]

    #now we have initialized the k points, we can start the clustering process

    #until the representative points do not change anymore
    while True:
        #find the closest representative point for each data point
        distances = torch.cdist(train_data, k_stack)
        closest = torch.argmin(distances, dim = 1)

        #find the new representative points
        new_k_stack = torch.zeros((k, 28 * 28)).to(device)
        for i in range(k):
            #find the center of mass of the cluster
            new_k_stack[i] = torch.mean(train_data[closest == i], dim = 0)

        #check if the representative points have changed
        if torch.all(torch.eq(new_k_stack, k_stack)):
            break
        else:
            k_stack = new_k_stack

    #As the k clusters are not points of the data, we cannot use the labels of the data to test the model.
    #Instead, we will use the labels of the representative points.
    #We will then check if the closest representative point has the same label as the test point.
    #If it does, we will say that the test point is correctly classified.
    #If it does not, we will say that the test point is incorrectly classified.

    #find the labels of the representative points
    #to do so, we say that the label of a representative point is the label of the closest data point to it
    #thus the distance is computed between the representative point and the data points as the transpose of the distance between the data points and the representative points
    distances = torch.cdist(k_stack, train_data)
    closest = torch.argmin(distances, dim = 1)
    # TODO : test with the ponderated mean of the labels of the closest data points
    k_labels = train_labels[closest]

    #now we can test the model
    #find the closest representative point for each test point
    distances = torch.cdist(test_data, k_stack)
    closest = torch.argmin(distances, dim = 1)

    #check if the closest representative point has the same label as the test point
    correct = torch.eq(k_labels[closest], test_labels)

    #return the percentage of correctly classified test points
    test_accuracy = torch.sum(correct).item() / 750

    #repeat for the training data
    distances = torch.cdist(train_data, k_stack)
    closest = torch.argmin(distances, dim = 1)

    correct = torch.eq(k_labels[closest], train_labels)

    train_accuracy = torch.sum(correct).item() / 6000

    return k_stack, k_labels, train_accuracy, test_accuracy

#plot the training and test errors as a function of k
k_values = torch.arange(1, 11)

k_stack_list = []
k_labels_list = []

train_errors = torch.zeros(10)
test_errors = torch.zeros(10)
for i in tqdm(range(10)):
    k_stack, k_labels, train_accuracy, test_accuracy = kmeans(k_values[i], train_data, train_labels, test_data, test_labels)

    k_stack_list.append(k_stack)
    k_labels_list.append(k_labels)

    train_errors[i] = 1 - train_accuracy
    test_errors[i] = 1 - test_accuracy

plt.plot(k_values, train_errors, label = 'Training error')
plt.plot(k_values, test_errors, label = 'Test error')
plt.xlabel('k')
plt.ylabel('Error')
plt.legend()
plt.show()

#plot the images in k_stack that represent each cluster (only for k = 2, 3 and 10)

def plot_k(k):
    #reshape the images into 28 * 28 matrices
    k_stack = k_stack_list[k - 1].reshape((k, 28, 28))

    #plot the images
    fig, ax = plt.subplots(1, k, figsize = (28, 28))
    for i in range(k):
        ax[i].imshow(k_stack[i].cpu().numpy(), cmap = 'gray')
        ax[i].axis('off')
    plt.show()

plot_k(2)
plot_k(3)
plot_k(10)