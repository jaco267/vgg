#%%
import torch as tc
import matplotlib.pyplot as plt
from utils.utils import Utils
from vgg import VGG_net
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
outdir = "./__train_ML_data";
batch_size=128
train_loader, test_loader = Utils.loadCIFAR_10_data_aug(batch_size=128,folder="./_python_ML_data")   
def load_info(filename_dic):  #{file1: {'model':model},file2:{'model'}}
    for filename in filename_dic.keys():
        path_loader = tc.load(filename)
        print(f"loading fromer check point file:  {filename}")
        filename_dic[filename]['loss_list'] = path_loader["loss_list"]
        filename_dic[filename]["train_acc_list"]  = path_loader["train_acc_list"]
        filename_dic[filename]["test_acc_list"]  = path_loader["test_acc_list"]  
        filename_dic[filename]["model"].load_state_dict(path_loader["model_state_dict"])
    return filename_dic
def plot_test_acc(filename_dic):
    fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.set_title("accuracy",pad=0)
    ax.set_ylim([0, 100])
    ax.set_xlabel('epochs',fontsize =18)
    ax.set_ylabel('accuracy',fontsize =18)
    for info_dic in filename_dic.values():
        ax.plot( tc.arange(len(info_dic['test_acc_list'])),   #x
                    tc.tensor(info_dic['test_acc_list']),label=info_dic['model_type'])        #y
    ax.legend()
    plt.tight_layout() 
    plt.show()    
def plot_train_acc(train_acc_list):
    fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.set_ylim([0, 100])
    ax.plot( tc.arange(len(train_acc_list)),   #x
                tc.tensor(train_acc_list),label="train")        #y
    ax.legend()
    plt.tight_layout() 
    plt.show() 
def plot_loss(loss_list):
    fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.set_title("loss",pad=0)
    ax.set_ylim([0, 3])
    ax.set_xlabel('iterations',fontsize =18)
    ax.set_ylabel('loss',fontsize =18)
    ax.plot( tc.arange(len(loss_list)),   #x
                tc.tensor(loss_list))        #y
    ax.legend()
    plt.tight_layout() 
    plt.show() 
def plot_acc(model):
    fig,ax=plt.subplots(1,1,figsize=(8,8))
    test_acc,n_class_correct,n_class_samples =Utils.forward_testset(model,test_loader,device,batch_size)    
    ax.set_title("accuracy at each label",pad=0)
    ax.set_ylim([0,1])
    ax.set_ylabel('accuracy',fontsize =18)
    ax.plot(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               tc.tensor(n_class_correct)/tc.tensor(n_class_samples))
    ax.legend()
    plt.tight_layout() 
    plt.show() 
if __name__ == "__main__":

    filename_dic = { }
    model_type_list = ["VGG11","VGG13","VGG16","VGG19"]
    for model_type in model_type_list:
        filename_dic[f"{outdir}/{model_type}_epoch60.pth"]= {'model':VGG_net(model_type).to(device),'model_type':model_type}
    '''
    filename_dic = {
        'vgg11_epoch60' : {'model' VGG_net(vgg11)},
        ...
        'vgg19_epoch60' : {'model' VGG_net(vgg11)},
    }
    '''
    filename_dic = load_info(filename_dic)
    plot_test_acc(filename_dic)
     
   
