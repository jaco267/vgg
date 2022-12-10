import torch as tc
import torchvision
import torchvision.transforms as transforms
class Utils():
    def loadCIFAR_10_data_aug(batch_size,folder):
        # dataset has PILImage images of range [0, 1]. 
        # We transform them to Tensors of normalized range [-1, 1]
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
        train_dataset = torchvision.datasets.CIFAR10(root=folder, train=True,download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=folder, train=False,download=True, transform=test_transform)
        train_loader = tc.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
        test_loader = tc.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def forward_testset(model,test_loader,device,batch_size):
        with tc.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # max returns (value ,index)
                _, predicted = tc.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                
                for i in range(batch_size):
                    try:
                        label = labels[i]
                        pred = predicted[i]
                        if (label == pred):
                            n_class_correct[label] += 1
                        n_class_samples[label] += 1
                    except:
                        pass
                        # print(labels.shape,"out of bound")

            acc = 100.0 * n_correct / n_samples        
        return acc, n_class_correct, n_class_samples
