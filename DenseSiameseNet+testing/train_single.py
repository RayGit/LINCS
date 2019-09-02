import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from read_singleimg import *
from Densenet import DenseNet
from utils_single_centerloss import Trainer
from losses import CenterLoss
BATCH_SIZE=16
F_LEN=12328
TRAIN_LIST="trainsample_12328.csv"
VAL_LIST="validationsample_12328.csv"
SAVE_PATH='checkpoints/exp2'
def get_dataloader(batch_size):
    '''mytransform = transforms.Compose([
        transforms.ToTensor()])'''

    # torch.utils.data.DataLoader
    train_loader = torch.utils.data.DataLoader(
        ImageFolder(TRAIN_LIST
                      ),
        batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(
        ImageFolder(VAL_LIST
                      ),
        batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader

def main(batch_size):
    train_loader, test_loader = get_dataloader(batch_size)
    # model= DinkNet34(num_classes=1)
    model=DenseNet()

    # optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    optimizer=optim.Adam(params=model.parameters())
    trainer = Trainer(model, optimizer, nn.CrossEntropyLoss, save_freq=1,save_dir=SAVE_PATH)
    trainer.loop(400, train_loader, test_loader)


if __name__ == '__main__':
    main(BATCH_SIZE)