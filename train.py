import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fdvm_net import FDVMNet
from data.dataset import EndoExposureDataset
from utils.losses import FDVMLoss
from configs.train import config
import os


def train():
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FDVMNet(num_blocks=config['num_blocks'],
                    num_channels=config['num_channels']).to(device)

    # Loss and optimizer
    criterion = FDVMLoss(lambda_per=config['lambda_per'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Dataset and dataloader
    train_dataset = EndoExposureDataset(config['data_root'], split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4)

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            low = batch['low'].to(device)
            normal = batch['normal'].to(device)

            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, normal)
            loss.backward()
            optimizer.step()

            if i % config['print_freq'] == 0:
                print(
                    f'Epoch [{epoch + 1}/{config["epochs"]}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(config['save_dir'], f'fdvm_epoch_{epoch + 1}.pth'))


if __name__ == '__main__':
    train()