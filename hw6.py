import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class Block1(nn.Module):

    def __init__(self, f_size):
        super(Block1, self).__init__()

        self.bn1 = nn.BatchNorm2d(f_size)
        self.bn2 = nn.BatchNorm2d(f_size)

        self.conv1 = nn.Conv2d(f_size, f_size, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(f_size, f_size, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        res = x

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = res + x

        return x

class Block2(nn.Module):

    def __init__(self, f_size):
        super(Block2, self).__init__()

        self.bn1 = nn.BatchNorm2d(f_size)
        self.bn2 = nn.BatchNorm2d(f_size*2)

        self.conv1 = nn.Conv2d(f_size, f_size*2, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(f_size*2, f_size*2, kernel_size = 3, stride = 1, padding = 1)
        self.conv_r = nn.Conv2d(f_size, f_size*2, kernel_size = 1, stride = 2)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)

        res = x
        res = self.conv_r(res)

        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = res + x

        return x

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()

        self.conv_input = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)

        stage1_block = [Block1(64)]*nblk_stage1
        self.stage1 = nn.Sequential(*stage1_block)

        stage2_block = [Block2(64)] + [Block1(128)]*(nblk_stage2 - 1)
        self.stage2 = nn.Sequential(*stage2_block)

        stage3_block = [Block2(128)] + [Block1(256)]*(nblk_stage3 - 1)
        self.stage3 = nn.Sequential(*stage3_block)

        stage4_block = [Block2(256)] + [Block1(512)]*(nblk_stage4 - 1)
        self.stage4 = nn.Sequential(*stage4_block)

        self.fc = nn.Linear(512, 10)

    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################

        x = self.conv_input(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, kernel_size = 4, stride = 4)
        x = x.squeeze()
        out = self.fc(x)

        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')


