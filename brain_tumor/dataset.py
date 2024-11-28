from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
from datasets import load_dataset
import numpy as np
import torchvision.transforms as transforms
import os
from datasets import load_dataset


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Initialisation du dataset.
        data : DataFrame contenant les chemins des images et les labels.
        transform : transformations à appliquer aux images.
        """
        self.data = data
        self.transform = transform

        # Calcul des statistiques (mean et std)
        # self.mean, self.std = self.calculate_mean_std()
        # print(f"Moyenne par canal (R, G, B) : {self.mean}")
        # print(f"Écart-type par canal (R, G, B) : {self.std}")

        # Normalisation dans les transformations
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        """Retourne la taille du dataset."""
        return len(self.data)

    def __getitem__(self, idx):
   
        if idx >= len(self.data):
            raise IndexError(f"L'indice {idx} est hors des limites.")
        
        try:

            # Extraction des paths et labels
            image_path = self.data["image"][idx]["path"]
            label = self.data["label"][idx]
                        # Gestion des labels inversés
            label = 0.0 if label == 1 else 1.0
    
            # Chargement et prétraitement de l'image
            img = Image.open(image_path).convert('RGB')
    
            # Appliquer des transformations (comme la normalisation, CLAHE, etc.)
            img = self.transform(img)
    
            # Retourner l'image et le label
            return img, label
        
        except Exception as e:
            print(f"Erreur à l'indice {idx}: {e}")
            raise ValueError(f"Erreur à l'indice {idx}: {e}")


    def calculate_mean_std(self):
        """
        Calcule la moyenne et l'écart-type par canal pour les images du dataset.
        """
        mean = np.zeros(3)
        std = np.zeros(3)
        total_pixels = 0

        transform = transforms.Compose([
            transforms.ToTensor()  # Conversion en tenseur (C, H, W)
        ])

        for index, row in self.data.iterrows():
            image_path = row['image']['path']

            # Vérifier si le fichier existe
            if not os.path.exists(image_path):
                print(f"Fichier introuvable : {image_path}")
                continue

            try:
                # Charger l'image
                image = Image.open(image_path).convert("RGB")
                tensor_image = transform(image)

                # Accumuler les statistiques
                mean += tensor_image.mean(dim=(1, 2)).numpy()
                std += tensor_image.std(dim=(1, 2)).numpy()
                total_pixels += 1
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {image_path} : {e}")

        if total_pixels > 0:
            mean /= total_pixels
            std /= total_pixels
        else:
            raise ValueError("Aucune image valide pour calculer les statistiques.")

        return mean, std
 
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.5], std=[0.5]) 
                        ])
        self.val_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.5], std=[0.5]) 
                        ])

    def setup(self, stage: str):

        if stage == 'fit':
            train_data = load_dataset('Simezu/brain-tumour-MRI-scan', split='train',cache_dir='../data')
            train_val_split = train_data.train_test_split(test_size=0.23)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']
            train_dataset  = train_dataset.to_pandas()
            val_dataset  = val_dataset.to_pandas()
            self.train_data = CustomDataset(data=train_dataset,transform=self.train_transform)
            self.val_data = CustomDataset(data=val_dataset,transform=self.val_transform)
        
        elif stage == 'test':
            self.test_data = load_dataset('Simezu/brain-tumour-MRI-scan', 
                                          split='test',
                                          cache_dir='../data').to_pandas()
            self.test_data = CustomDataset(data=self.test_data,transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,shuffle=False)