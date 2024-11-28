



import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import Precision, Recall, Accuracy, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torchvision import models
import pytorch_lightning as pl


class BrainTumorEfficientNetB1(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        # Load EfficientNet B1 pretrained model
        self.model = models.efficientnet_b1(pretrained=True)

        self.lr = lr
        
        # Modify the classifier layer for binary classification
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

        # Loss function
        self.loss_function = nn.BCEWithLogitsLoss()

        # Metrics
        self.precision = Precision(average='binary', task="binary")
        self.recall = Recall(average='binary', task="binary")
        self.accuracy = Accuracy(average='binary', task="binary")
        self.f1_score = F1Score(average='binary', task="binary")
        self.confusion_matrix = ConfusionMatrix(num_classes=2, task="binary")

        # For validation
        self.precision_val = Precision(average='binary', task="binary")
        self.recall_val = Recall(average='binary', task="binary")
        self.accuracy_val = Accuracy(average='binary', task="binary")
        self.f1_score_val = F1Score(average='binary', task="binary")

        # For testing
        self.precision_test = Precision(average='binary', task="binary")
        self.recall_test = Recall(average='binary', task="binary")
        self.accuracy_test = Accuracy(average='binary', task="binary")
        self.f1_score_test = F1Score(average='binary', task="binary")

        # Store outputs for validation and tests
        self.val_outputs = []
        self.test_outputs = []
        
        # Store losses and metrics
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def forward(self, x):
        # if len(x.shape) == 2:  # In case images are flattened
        #     x = x.view(-1, 3, 224, 224)  # Reshape to [batch_size, 3, 224, 224]
        x = self.model(x)  # Pass through the model
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        loss = self.loss_function(logits, labels.unsqueeze(1).float())  # Use probabilities
        
        # Binary predictions
        preds = (logits.sigmoid() > 0.5).squeeze().float()
        labels = labels.squeeze().float()

        # Update metrics
        self.accuracy(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1_score(preds, labels)

        # Log metrics
        self.log('train_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_precision', self.precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_recall', self.recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_f1_score', self.f1_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        loss = self.loss_function(logits, labels.unsqueeze(1).float())  # Use probabilities

        # Binary predictions
        preds = (logits.sigmoid()>0.5).squeeze().float()
        labels = labels.squeeze().float()

        # Update metrics
        self.accuracy_val(preds, labels)
        self.precision_val(preds, labels)
        self.recall_val(preds, labels)
        self.f1_score_val(preds, labels)

        # Log metrics
        self.log('val_accuracy', self.accuracy_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_score', self.f1_score_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # Save outputs for confusion matrix
        self.val_outputs.append({'preds': preds, 'labels': labels, 'loss': loss})
        self.val_losses.append(loss.item())  # Save loss
        return loss

    def on_validation_epoch_end(self):
        # Calculate confusion matrix
        all_preds = torch.cat([x['preds'] for x in self.val_outputs])
        all_labels = torch.cat([x['labels'] for x in self.val_outputs])

        conf_matrix = self.confusion_matrix(all_preds, all_labels)

        # Log confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        wandb.log({"confusion_matrix_val": wandb.Image(plt)})

        # Reset outputs for next epoch
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        loss = self.loss_function(logits, labels.unsqueeze(1).float())  # Use probabilities

        # Binary predictions
        preds = (logits.sigmoid()>0.5).squeeze().float()
        labels = labels.squeeze().float()

        # Update metrics
        self.accuracy_test(preds, labels)
        self.precision_test(preds, labels)
        self.recall_test(preds, labels)
        self.f1_score_test(preds, labels)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', self.accuracy_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_precision', self.precision_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', self.recall_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1_score', self.f1_score_test, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        
    # Optionnel : Calcul et affichage de la matrice de confusion
        if self.test_outputs:  # Vérifiez qu'il y a des données enregistrées
          all_preds = torch.cat([x['preds'] for x in self.test_outputs])
          all_labels = torch.cat([x['labels'] for x in self.test_outputs])
          conf_matrix = self.confusion_matrix(all_preds, all_labels)

          # Afficher et journaliser la matrice de confusion
          plt.figure(figsize=(6, 6))
          sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues', cbar=False)
          plt.xlabel('Prédictions')
          plt.ylabel('Vérités')
          wandb.log({"matrice_de_confusion_test": wandb.Image(plt)})
          plt.close()
          # Effacer les sorties stockées pour préparer la prochaine exécution
        self.test_outputs = []

