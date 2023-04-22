from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
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

    # create model
    if model_name == "vit":
        model = Vit()

    model = model.to(device)

    # load the data
    train_data_path,test_data_path = "frames/train","frames/test"
    train_loader, test_loader = prepare_data(train_data_path,test_data_path,cfg)

    # load training parameters
    writer = SummaryWriter()
    training_lost = []

    initial_lr = cfg['initial_lr']
    num_epochs = cfg['num_epochs']
    eval_interval = cfg['eval_interval']
    save_checkpoint_interval = cfg['save_checkpoint_interval']

    criterion_name = cfg['criterion']
    criterion = nn.CrossEntropyLoss()

    optimizer_name = cfg['optimizer']
    optimizer = optim.Adam(model.parameters(),lr = initial_lr)








