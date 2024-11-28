from .models import BrainTumorEfficientNetB1
import pytorch_lightning as pl
import torch
import wandb
from ..dataset import MyDataModule
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger


def train(batch_size=32,lr=1e-4,
          wandb_project="brain-tumor-classification",
          run_name="debug",
          ckpt_dir=None,
          log_model=False,
          epochs=20,
          use_early_stopping=False):

    # Logger WandB
    wandb_logger = WandbLogger(project=wandb_project, name=run_name, log_model=log_model)

    # Configuration des callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1_score',               # Surveiller la perte de validation
        dirpath=ckpt_dir,                     # Dossier pour sauvegarder les checkpoints
        filename='best_model',                # Nom du fichier du modèle sauvegardé
        save_top_k=1,                         # Sauvegarder le meilleur modèle
        mode='max',                      
        verbose=True,
        save_weights_only=True
    )
    callbacks = [checkpoint_callback,]
    if use_early_stopping:
        early_stopping_callback = EarlyStopping(
        monitor='val_f1_score',       # Surveiller la perte de validation (peut être remplacé par une autre métrique)
        patience=10,               # Nombre d'époques sans amélioration avant d'arrêter
        verbose=False,             # Afficher les informations
            mode='max',               # 'min' car nous cherchons à minimiser la perte
        )
        callbacks.append(early_stopping_callback)


    # data
    data = MyDataModule(batch_size=batch_size)

    # Initialisation du modèle
    model = BrainTumorEfficientNetB1(lr=lr)

    # Choisir l'accélérateur
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # Initialisation du Trainer avec les callbacks
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=wandb_logger,
        max_epochs=epochs,
        callbacks=callbacks
    )

    # Entraînement
    trainer.fit(model,datamodule=data)

    # Tracer les métriques après l'entraînement
    #model.plot_metrics()

    # Finaliser le run de WandB
    wandb.finish()
