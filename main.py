import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import Dataset, Importer
import argparse
import os



class MarshNet(nn.Module):
    def __init__(self):
        super(MarshNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 5, 1) 
        #self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 128, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1) 
        self.conv3_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(25*25*256, 500)
        self.fc1_drop = nn.Dropout(p = 0.25)
        self.fc2 = nn.Linear(500, 100)
        self.fc2_drop = nn.Dropout(p = 0.25)
        self.fc3 = nn.Linear(100, 40)
        self.fc3_drop = nn.Dropout(p = 0.25)
        self.fc4 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(self.conv2_bn(x), 2, 2)
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(self.conv3_bn(x), 2, 2)
        x = x.view(-1, self.num_flat_features(x))        
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension [batch_size, features, width, height]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Marsh test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')                            
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    # Data preprocessing
    transform = transforms.Compose(
                    [transforms.Resize((224, 224)),   
                    transforms.ColorJitter(brightness=16, contrast=16, saturation=16, hue=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Dataset loading and splitting
    data, labels = Importer().data_importer()
    full_dataset = Dataset(data, labels, transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Setting batch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, **kwargs)

    # Creating model and loading on gpu
    model = MarshNet().to(device, dtype = torch.double)  
    model = nn.DataParallel(model) #As multi-gpu in Keras

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"marsh_cnn.pt")
        
if __name__ == '__main__':
    main()