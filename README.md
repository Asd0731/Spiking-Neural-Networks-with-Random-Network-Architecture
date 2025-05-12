# Spiking-Neural-Networks-with-Random-Network-Architecture
This repository is the official implementation of  Spiking Neural Networks with Random Network Architecture
For the implementation examples of the main algorithms in the thesis, please refer to the training method. 
Currently, this repository showcases the implementation process of some original figures in the thesis, as well as the data used for drawing these figures. 
In the subsequent section discussing hyperparameters, you only need to change the hyperparameter settings within the loop.  

### Section 1: Preparation Work
Before starting to run the code for experiments, you need to download four required datasets: mnist, fmnist, kmnist, and emnist (corresponding to those in the text), and save them in the current folder. Additionally, you need to download the corresponding packages.
```python
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.nn import init
import time
```

### Section 2: Loading Data
The following is the data loading section.
#### Mnist
```python
# dataloader arguments
batch_size = 128
data_path='/mnist'
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
```
The loading processes of Fmnist, Kmnist are almost the same as that of mnist. Emnist is slightly different, as follows:
```python
# dataloader arguments
batch_size = 128
data_path='./Emnist'
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
mnist_train = datasets.EMNIST(data_path, split = 'byclass', train=True, download=True, transform=transform)
mnist_test = datasets.EMNIST(data_path, split = 'byclass', train=False, download=True, transform=transform)
# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
```

### Section 3: Building the Network
The following is about building the network.
```python
# Network Architecture
num_inputs = 28*28
num_hidden = 2000
num_outputs = 10
# Temporal Dynamics
num_steps = 25
beta = 0.95
# Define Network
class my_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        #self.fc2 = nn.Linear(num_hidden, num_hidden)
        #self.lif2 = snn.Leaky(beta=beta)
    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        #mem2 = self.lif2.init_leaky()
        # Record the final layer
        spk1_rec = []
        mem1_rec = []
        #spk2_rec = []
        #mem2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            #cur2 = self.fc2(spk1)
            #spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0)
def generate_spike(x):
    data_input = x.view(128,-1)
    spk_input = []
    for batch in range(128):
        spk_input_batch = spikegen.rate(data_input[batch], num_steps = 25)
        spk_input.append(spk_input_batch)
    return torch.stack(spk_input, dim = 1)
```

### Section 4: Training and Testing
The following corresponds to the training method in the text. First, randomly generate weights and perform forward propagation, and record the results of the last layer.
```python
# Load the network onto CUDA if available
net = my_Net().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
#generate train data
num_epochs = 1
train_result = []
train_tgt = []
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)
    # Train set
    with torch.no_grad():
        net.eval()
        for train_data, train_targets in train_batch:
            train_data = train_data.to(device)
            train_targets = train_targets.to(device)
            # Test set forward pass
            train_spk, train_mem = net(generate_spike(train_data))
            train_tgt.append(train_targets)
            train_result.append(train_spk)
            iter_counter += 1
            if iter_counter == 400:
                break
train_input = torch.stack(train_result, dim=0).sum(dim = 1)
train_t = torch.stack(train_tgt, dim=0)
```
Then, perform training and test the accuracy after each training. By commenting out the "test my net" part, you can obtain the time for a complete training. Note that if the dataset is emnist, the output dimension of train_net should be changed to 62; otherwise, it is 10.
```python
start_time = time.time()
train_net = torch.nn.Linear(num_hidden,10)
optimizer1 = torch.optim.Adam(train_net.parameters(), lr=5e-4, betas=(0.9, 0.999))
my_train_loss_hist = []
train_accuracy_hist = []
test_accuracy_hist = []
for batch in range(400):
    train_net_out = train_net(train_input[batch])
    max_value_train, max_index_train = train_net_out.max(dim=1)
    train_accuracy = (max_index_train == train_t[batch]).sum()/128
    train_accuracy_hist.append(train_accuracy)
    # test my net
    num_epochs = 1
    my_test_result = []
    my_test_tgt = []
    for epoch in range(num_epochs):
        iter_counter = 0
        test_batch = iter(test_loader)
        # Test set
        with torch.no_grad():
            net.eval()
            for my_test_data, my_test_targets in test_batch:
                my_test_data = my_test_data.to(device)
                my_test_targets = my_test_targets.to(device)
                # Test set forward pass
                my_test_spk, my_test_mem = net(generate_spike(my_test_data))
                my_test_tgt.append(my_test_targets)
                my_test_result.append(my_test_spk)
                iter_counter += 1
                if iter_counter == 50:
                    break
    max_value, max_index = train_net(torch.stack(my_test_result, dim=0).sum(dim = 1)).max(dim=2)
    testacc = sum(max_index.reshape(-1) == torch.stack(my_test_tgt, dim=0).reshape(-1))/6400
    test_accuracy_hist.append(testacc)
    train_loss = torch.zeros((1), dtype=dtype, device=device)
    train_loss = loss(train_net_out, train_t[batch])
    # Gradient calculation + weight update
    optimizer1.zero_grad()
    train_loss.backward()
    optimizer1.step()
    my_train_loss_hist.append(train_loss.item())
end_time = time.time()
elapsed_time = end_time - start_time
print(f"time: {elapsed_time} s")
```

### Section 5: Summary
Finally, we retain the three results of my_train_loss_hist, train_accuracy_hist, and test_accuracy_hist for plotting. In subsequent experiments discussing hyperparameters and random generation, we only need to adjust the initialization method based on the above process. 
