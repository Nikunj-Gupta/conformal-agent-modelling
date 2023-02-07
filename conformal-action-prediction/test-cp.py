import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
from lenet import LeNet 

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
# parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

    # Get the conformal calibration dataset
    # imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib,50000-args.num_calib])

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck') 
    
    imagenet_calib_data, imagenet_val_data = trainset, testset 

    # Initialize loaders 
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)


    cudnn.benchmark = True

    # Get the model 
    model = LeNet() 
    # model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(200):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs, targets
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
    
    model = torch.nn.DataParallel(model) 
    model.eval()

    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False 
    # use the randomized version of conformal
    randomized = True 

    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0, randomized=randomized, allow_zero_sets=allow_zero_sets)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    # validate(val_loader, model, print_bool=True)
    validate_new(val_loader, model, print_bool=True)

    print("Complete!")