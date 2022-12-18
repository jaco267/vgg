#%%
import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #* for import upper directory modules
from utils.utils import Utils
import torch.backends.cudnn as cudnn
import time 
# Hyper-parameters   #we use Adam        #20 2500  0.01
num_epochs, batch_size, learning_rate  = 200 ,128, 0.1 
#                   should set batch_size to 128 (so batch_norm can have better regularization with batch norm)
#  batch norm --> smaller batch size, better regularization
#                   because I dont know how to use drop out in pytorch
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

layer_structure = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
    def __init__(self,vgg_name):
        super(VGG_net, self).__init__()                       
        self.features = self._make_layers(layer_structure[vgg_name])
        self.linear_input = 512
        self.fc1 = nn.Linear(self.linear_input, 10)        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.linear_input)           
        x = self.fc1(x)             
        return x
    def _make_layers(self,layer_structure):
        layers = []
        in_channels = 3
        for x in layer_structure:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(negative_slope=0.01,inplace=True), 
                ]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers) 

if __name__ == "__main__":
    # Device configuration
    outdir = "./__train_ML_data";
    if not os.path.exists(outdir): os.makedirs(outdir)
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

    model_type = "VGG11"  #!!!  maybe try VGG19 next time?
    checkpoint_interval = 10  #20
    former_check_point_file = f"{outdir}/{model_type}_epoch00.pth"
    # former_check_point_file = f"{outdir}/former/{model_type}_epoch200.pth"

    model = VGG_net(model_type).to(device);   print(device)
    if device == 'cuda':    net = tc.nn.DataParallel(model);  cudnn.benchmark = True;    

    loss_list = [];   train_acc_list  = []; test_acc_list  = [];  print_step = 100
    train_time = 0;
    try:
        path_loader = tc.load(former_check_point_file)
        print(f"loading fromer check point file:  {former_check_point_file}")
        model.load_state_dict(path_loader["model_state_dict"])
        batch_size = path_loader["batch_size"]
        loss_list = path_loader["loss_list"]
        train_acc_list  = path_loader["train_acc_list"]
        test_acc_list   = path_loader["test_acc_list"]  
        epoch_prev = path_loader["epoch"]
        train_time = path_loader["train_time"]
    except:
        epoch_prev = 0;  print("file not found")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = tc.optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader, test_loader = Utils.loadCIFAR_10_data_aug(batch_size,folder="./_python_ML_data")    
    n_total_steps = len(train_loader)


    
    start_time = time.time()
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

            train_acc = 100.*correct/total
            loss_list.append(loss)  
            train_acc_list.append(train_acc)
            
            if (i+1) % print_step == 0:
                print (f'Epoch [{epoch+1+epoch_prev}/{epoch_prev+num_epochs}], Step [{i+1}/{n_total_steps},train accuracy {train_acc:.1f}], Loss: {loss.item():.4f}')

        test_acc,_,_ =Utils.forward_testset(model,test_loader,device,batch_size)
        test_acc_list.append(test_acc)
        print(f"epoch test {epoch_prev + epoch + 1} accuracy :  {test_acc}")
        if( (epoch_prev + epoch + 1) % checkpoint_interval == 0):
            check_point_epoch = epoch_prev + epoch + 1
            end_time = time.time()  
            print("check point epoch ",check_point_epoch,"???")
            saveObj = {"model_state_dict":model.state_dict(),
                "batch_size": batch_size,
                "loss_list": loss_list,
                "train_acc_list":train_acc_list,
                "test_acc_list":test_acc_list,
                "epoch": check_point_epoch,
                "train_time":train_time + end_time-start_time}    
            tc.save(saveObj, f'{outdir}/{model_type}_epoch{check_point_epoch}.pth')
        scheduler.step()  
    
    print('Finished Training')

    end_time = time.time() 
    test_acc,n_class_correct,n_class_samples =Utils.forward_testset(model,test_loader,device,batch_size)
    print(f'Accuracy of the network: {test_acc} %')
    print("former train time",train_time)
    print("this train time ",format(end_time-start_time))
    print("training time",train_time+end_time-start_time)

    fig,ax=plt.subplots(3,1,figsize=(8,18))
    fig.suptitle(f'{model_type} epoch{len(test_acc_list)}', fontsize=20, y=0.99)

    ax[0].set_title("accuracy",pad=0)
    ax[0].set_ylim([0, 100])
    ax[0].set_xlabel('epochs',fontsize =18)
    ax[0].set_ylabel('accuracy',fontsize =18)
    ax[0].plot( tc.arange(len(test_acc_list)),   #x
                tc.tensor(test_acc_list),label="test")        #y

    step = len(train_acc_list) //len(test_acc_list)
    train_acc_epoch_list = train_acc_list[0::step]
    ax[0].set_ylim([0, 100])
    ax[0].plot( tc.arange(len(train_acc_epoch_list)),   #x
                tc.tensor(train_acc_epoch_list),label="train")        #y
    ax[0].legend()

    ax[1].set_title("loss",pad=0)
    ax[1].set_ylim([0, 3])
    ax[1].set_xlabel('iterations',fontsize =18)
    ax[1].set_ylabel('loss',fontsize =18)
    ax[1].plot( tc.arange(len(loss_list)),   #x
                tc.tensor(loss_list))        #y
    
    ax[2].set_title("accuracy at each label",pad=0)
    ax[2].set_ylim([0,1])
    ax[2].set_ylabel('accuracy',fontsize =18)
    ax[2].plot(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               tc.tensor(n_class_correct)/tc.tensor(n_class_samples))

    plt.tight_layout() 
    plt.show()
#18

