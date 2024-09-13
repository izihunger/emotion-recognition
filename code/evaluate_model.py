import torch
import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from model import DeepEmotionCNN
from preprocessing import preprocessing
from dataset import Emotion_Dataset
from sklearn.preprocessing import LabelEncoder

def evaluate_and_save_metrics(model, test_loader, class_names, device, output_dir="evaluation_results", file_name="metrics.txt"):
    """Allows you to evaluate your model and save metrics about this evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_pred_classes = np.array(all_preds)
    y_true = np.array(all_labels)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    accuracy = accuracy_score(y_true, y_pred_classes)
    metrics_file_path = os.path.join(output_dir, file_name)
    with open(metrics_file_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=', '))
        file.write("\n\nClassification Report:\n")
        file.write(report)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    conf_matrix_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_image_path)
    print(f"Metrics saved to {metrics_file_path}")
    print(f"Confusion matrix image saved to {conf_matrix_image_path}")

@click.command()
@click.option('--model-path', '-m', type=str, required=True, help='Path to the model.')
@click.option('--dataset-path', '-d', type=str, required=True, help='Path to the csv file.')
@click.option('--images-dir', '-i', type=str, required=True, help='Path to the images directory.')
@click.option('--output', '-o', default=Path("./results"), help='Batch size.')
@click.option('--device', '-d', type=click.Choice(['cpu', 'gpu'], case_sensitive=False), default='cpu', help="Choose between 'cpu' or 'gpu'.")
def main(model_path: str, dataset_path: str, images_dir: str, output: Path, device: str):
    os.makedirs(output, exist_ok=True)
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
    
    # load model  
    print("Loading model...")  
    model = DeepEmotionCNN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # preprocessing
    output_path = Path(output + "/augmented_data")
    if not output_path.exists():
        print("Preprocessing images...")
        preprocessing(csv_file=dataset_path, image_dir=images_dir, output_dir=output_path, mode="test")
     
    # read data
    df_test = pd.read_csv(output_path/"augmented_data.csv", sep=",")
    # fit label_encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df_test["labels"].unique())
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
    ds_test = Emotion_Dataset(df_test.values, label_encoder.fit_transform(df_test["labels"].values), output_path.absolute().as_posix(), transformation)
    test_loader = DataLoader(ds_test, batch_size=10, shuffle=False)
    print("Evaluating your model...")
    evaluate_and_save_metrics(model, test_loader, label_encoder.classes_, device, output_dir=output)
    
if __name__ == '__main__':
    main()