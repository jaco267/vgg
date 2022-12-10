# ```py
#%%
import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #* for import upper directory modules
from utils.utils import Utils
import torch.backends.cudnn as cudnn

# Hyper-parameters   #we use Adam        #20 2500  0.01
num_epochs, batch_size, learning_rate  = 5 ,128, 0.1 
#                   should set batch_size to 128 (so batch_norm can have better regularization with batch norm)
#  batch norm --> smaller batch size, better regularization
#                   because I dont know how to use drop out in pytorch
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

layer_structure = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGG16_ConvNet(nn.Module):
    def __init__(self,vgg_name):
        super(VGG16_ConvNet, self).__init__()               
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1) 
        self.bn11   = nn.BatchNorm2d(num_features=64); self.relu11  = nn.ReLU()
        self.pool1   = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv21 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1) 
        self.bn21   = nn.BatchNorm2d(num_features=128); self.relu21  = nn.ReLU()
        self.pool2   = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv31 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1) 
        self.bn31   = nn.BatchNorm2d(num_features=256); self.relu31  = nn.ReLU()
        self.conv32 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1) 
        self.bn32   = nn.BatchNorm2d(num_features=256); self.relu32  = nn.ReLU()
        self.pool3   = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv41 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1) 
        self.bn41   = nn.BatchNorm2d(num_features=512); self.relu41  = nn.ReLU()
        self.conv42 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1) 
        self.bn42   = nn.BatchNorm2d(num_features=512); self.relu42  = nn.ReLU()
        self.pool4   = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv51 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1) 
        self.bn51   = nn.BatchNorm2d(num_features=512); self.relu51  = nn.ReLU()
        self.conv52 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1) 
        self.bn52   = nn.BatchNorm2d(num_features=512); self.relu52  = nn.ReLU()
        self.pool5   = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(512, 10)  
    def forward(self, x):                           # (batch_size, channel, w, h)    
        x = self.relu11(self.bn11(self.conv11(x)))  #?(n,3,32,32)  -> (n,64,32,32)   
        x = self.pool1(x)                           #*(n,64,32,32) -> (n,64,16,16) 
        x = self.relu21(self.bn21(self.conv21(x)))  #?(n,64,16,16)  -> (n,128,16,16) 
        x = self.pool2(x)                           #*(n,128,16,16) -> (n,128,8,8)   
        x = self.relu31(self.bn31(self.conv31(x)))  #?(n,128,8,8)  -> (n,256,8,8)                 
        x = self.relu32(self.bn32(self.conv32(x)))  #?(n,256,8,8)  -> (n,256,8,8)   
        x = self.pool3(x)                           #*(n,256,8,8) -> (n,256,4,4)   
        x = self.relu41(self.bn41(self.conv41(x)))  #?(n,256,4,4) -> (n,512,4,4)                 
        x = self.relu42(self.bn42(self.conv42(x)))  #?(n,512,4,4) -> (n,512,4,4)   
        x = self.pool4(x)                           #*(n,512,4,4) -> (n,512,2,2)   
        x = self.relu51(self.bn51(self.conv51(x)))  #?(n,512,2,2) -> (n,512,16,2)                 
        x = self.relu52(self.bn52(self.conv52(x)))  #?(n,512,2,2) -> (n,512,16,2)     
        x = self.pool5(x)                           #*(n,512,2,2) -> (n,512,1,1)   
        x = x.view(-1, 512)                         # (n,512,1,1) -> (n, 512)
        x = self.fc1(x)                             # (n, 512)    -> (n, 10)
        return x
print_step = 100
if __name__ == "__main__":
    # Device configuration
    outdir = "./__train_ML_data"
    model_type = "VGG16"  #!!!  maybe try VGG19 next time?
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    model = VGG16_ConvNet(model_type).to(device)
    print(device)
    if device == 'cuda':
        net = tc.nn.DataParallel(model)
        cudnn.benchmark = True    
    loss_list = []


    epoch_prev = 0
    print("file not found")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = tc.optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader, test_loader = Utils.loadCIFAR_10_data_aug(batch_size,"_python_ML_data")    
    n_total_steps = len(train_loader)

    checkpoint_interval = 20  #20

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [batch_size, 3, 32, 32] = 5000, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images,labels = images.to(device),labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (i+1) % print_step == 0:
                acc = 100.*correct/total
                print (f'Epoch [{epoch+1+epoch_prev}/{epoch_prev+num_epochs}], Step [{i+1}/{n_total_steps},accuracy {acc:.1f}], Loss: {loss.item():.4f}')
            loss_list.append(loss)  
        acc,_,_ =Utils.forward_testset(model,test_loader,device,batch_size)
        print(f"epoch {epoch + epoch_prev} accuracy :  {acc}")
        scheduler.step()  

    print('Finished Training')

    check_point_epoch = epoch_prev + num_epochs

    acc,n_class_correct,n_class_samples =Utils.forward_testset(model,test_loader,device,batch_size)
    print(f'Accuracy of the network: {acc} %')

    fig,ax=plt.subplots(1,1,figsize=(8,8))
    plt.ylim([0, 1])
    plot_y = tc.tensor(n_class_correct)/tc.tensor(n_class_samples)
    plot_x = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ax.plot(plot_x ,plot_y)

    fig,ax=plt.subplots(1,1,figsize=(8,8))
    plot_y = tc.tensor(loss_list)
    plot_x = tc.arange(len(loss_list))
    plt.ylim([0, 3])
    ax.plot(plot_x ,plot_y)
    plt.show()

#  ```