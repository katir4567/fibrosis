import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_accuracy(predictions, targets):
    _, preds = torch.max(predictions, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_model(model, train_loader, val_loader, args):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_correct += calculate_accuracy(outputs, targets) * targets.size(0)
            total_samples += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy:.2%}")

        # Validate
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2%}")

    return train_losses, val_losses

def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            total_correct += calculate_accuracy(outputs, targets) * targets.size(0)
            total_samples += targets.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = total_correct / total_samples
    return val_loss, val_accuracy

def evaluate_model(model, val_loader):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    print(classification_report(all_targets, all_predictions))

    cm = confusion_matrix(all_targets, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fibrosis', 'Pneumonia'], yticklabels=['Normal', 'Fibrosis', 'Pneumonia'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
