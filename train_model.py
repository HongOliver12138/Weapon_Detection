import os.path

from torch import nn, optim
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from vit import VIT
import torch


def prepare_data(train_data_path, test_data_path, cfg):
    batch_size =cfg['batch_size']

    # training transform

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load the data
    train_dataset = datasets.ImageFolder(train_data_path,transform = train_transform)
    test_dataset = datasets.ImageFolder(test_data_path,transform = test_data_path)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size,pin_memory = True)

    return train_loader, test_loader














# set up config

def get_cfg():
    cfg = dict()
    cfg['initial_lr'] = 5e-5
    cfg['optimizer'] = 'adam'
    cfg['num_epochs'] = 100
    cfg['criterion'] = 'improved_focal'
    cfg['batch_size'] = 64
    cfg['eval_interval'] = 3
    cfg['save_checkpoint_interval'] = 10



    return cfg


# training dataset

def train(cfg, model_name = "dswin"):

    device = torch.device("mps")

    model = None
    # create model
    if model_name == "vit":
        model = VIT()

    model = model.to(device)

    # load the data
    train_data_path,test_data_path = "frames/train","frames/test"
    train_loader, test_loader = prepare_data(train_data_path,test_data_path,cfg)

    # load training parameters
    writer = SummaryWriter()
    training_loss = []

    initial_lr = cfg['initial_lr']
    num_epochs = cfg['num_epochs']
    eval_interval = cfg['eval_interval']
    save_checkpoint_interval = cfg['save_checkpoint_interval']

    criterion_name = cfg['criterion']
    criterion = nn.CrossEntropyLoss()

    optimizer_name = cfg['optimizer']
    optimizer = optim.Adam(model.parameters(),lr = initial_lr)

    """
    log_filename = ""
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    """

    num_epochs = 5

    # train
    for epoch in range(num_epochs+1):
        running_loss = 0
        acc, correct,tot =0,0,0
        optimizer.zero_grad()


        for i, (data, label) in enumerate(train_loader):

            cur_acc, cur_correct, cur_tot = 0,0,0

            data,label = data.to(device, dtype = torch.float), label.to(device,dtype = torch.long)
            outs = model(data)
            loss = criterion(outs, label)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            _,pred = torch.max(outs, 1) # pick the max, as col dim 1

            cur_tot += label.size(0)    # process images in one batch
            cur_correct += (pred == label).sum().item()

            print(cur_correct)

            cur_acc = cur_correct * 100/ cur_tot

            correct += cur_correct
            tot += cur_tot

        running_loss /= len(train_loader)
        acc = correct *100.0 /tot

        training_loss.append(running_loss)

    print(acc, correct,tot)







import matplotlib.pyplot as plt

def visualize_images_and_labels(dataset, num_images=5):
    class_names = dataset.classes

    for i in range(num_images):
        image, label = dataset[i]
        # Convert image tensor back to PIL image and display it
        img = transforms.ToPILImage()(image).convert("RGB")
        plt.imshow(img)
        plt.title(f"Label: {class_names[label]} ({label})")
        plt.show()







if __name__  == "__main__":
    cfg = get_cfg()


    train(cfg,"vit")









