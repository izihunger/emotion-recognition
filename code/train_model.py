import torch
import os
import tqdm
import click
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from pathlib import Path
from model import DeepEmotionCNN
from preprocessing import preprocessing
from dataset import Emotion_Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Allows you to save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f"{checkpoint_path}/model_epoch_{epoch}.checkpoint")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, model, optimizer):
    """Allows you to load checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

def train(net: DeepEmotionCNN, epochs: int, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer, device, writer, output_path="./models", start_epoch=0):
    """Allows you to train and evaluate your model."""
    
    print("===================================Start Training===================================")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm.tqdm(range(start_epoch, epochs)):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # model training
        net.train()
        for data, labels in tqdm.tqdm(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        # log training loss and accuracy
        writer.add_scalar('Training Loss', train_loss / len(train_loader), e)
        writer.add_scalar('Training Accuracy', train_correct.double() / len(train_loader.dataset), e)
        # model validation
        net.eval()
        with torch.no_grad():
            for data,labels in tqdm.tqdm(val_loader):
                data, labels = data.to(device), labels.to(device)
                val_outputs = net(data)
                val_loss = criterion(val_outputs, labels)
                validation_loss += val_loss.item()
                _, val_preds = torch.max(val_outputs,1)
                val_correct += torch.sum(val_preds == labels.data)
        # log validation loss and accuracy
        writer.add_scalar('Validation Loss', validation_loss / len(val_loader), e)
        writer.add_scalar('Validation Accuracy', val_correct.double() / len(val_loader.dataset), e)

        train_loss = train_loss/len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        validation_loss =  validation_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))
        save_checkpoint(net, optimizer, e, loss, output_path)
    return train_losses, train_accuracies, val_losses, val_accuracies


@click.command()
@click.option('--dataset-path', type=str, required=True, help='Path to the csv file.')
@click.option('--images-dir', type=str, required=True, help='Path to the images directory.')
@click.option('--epochs', '-e', default=10, help='Number of epochs.')
@click.option('--batchsize', '-b', default=32, help='Batch size.')
@click.option('--output', '-o', default=Path("./models"), help='Batch size.')
@click.option('--device', '-d', type=click.Choice(['cpu', 'gpu'], case_sensitive=False), default='cpu', help="Choose between 'cpu' or 'gpu'.")
def main(dataset_path: str, images_dir: str, epochs: int, batchsize: int, output: Path, device: str):
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using GPU for inference.")
        else:
            print("GPU is not available, using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU for inference.")
        
    # log to tensorboard
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(f'runs/CNN_{current_time}')
        
    # preprocessing
    output_path = Path(output + "/augmented_data")
    if not output_path.exists():
        print("Preprocessing images...")
        preprocessing(csv_file=dataset_path, image_dir=images_dir, output_dir=output_path, mode="train")
     
    # read data
    df_training = pd.read_csv(output_path/"augmented_data.csv", sep=",")
    # split train / valid
    df_train, df_valid = train_test_split(df_training, test_size=0.1, random_state=42)
    # fit label_encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df_training["labels"].unique())
    # transform & scale images
    transformation= transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ]
    )
    # dataset & dataloader
    ds_train = Emotion_Dataset(df_train.values, label_encoder.fit_transform(df_train["labels"].values), output_path.absolute().as_posix(), transformation)
    ds_val = Emotion_Dataset(df_valid.values, label_encoder.fit_transform(df_valid["labels"].values), output_path.absolute().as_posix(), transformation)
    train_loader = DataLoader(ds_train, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batchsize, shuffle=False)
    
    lr = 0.001
    net = DeepEmotionCNN()
    net.to(device)
    print("Model architecture: ", net)
    # optimizer & loss
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.Adam(net.parameters(), lr=lr)
    # run training
    os.makedirs(output_path/current_time, exist_ok=True)
    training  = train(net, epochs, train_loader, val_loader, criterion, optimizer, device, writer=writer, output_path=output_path/current_time)
    
if __name__ == '__main__':
    main()
