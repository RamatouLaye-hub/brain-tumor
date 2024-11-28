from .models import BrainTumorEfficientNetB1
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from ..dataset import MyDataModule


def test(wandb_project,run_name,batch_size=32,lr):

    wandb_logger = WandbLogger(project=wandb_project,name=run_name)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

       # Initialisation du modèle
    model = BrainTumorEfficientNetB1(lr=1e-4)
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=wandb_logger,
    )

    data = MyDataModule(batch_size=batch_size)
    trainer.test(model, datamodule=data)

    # Tracer les métriques après l'entraînement
    #model.plot_metrics()

    # Finaliser le run de WandB
    wandb.finish()