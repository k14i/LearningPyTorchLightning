import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

# For type hinting
from typing import Union, Dict, List, Any
from torch import Tensor


class MNISTConvNet(pl.LightningModule):
    def __init__(self,
            conv1_in_channels: int, conv1_out_channels: int, conv1_kernel_size: int, conv1_stride: int,
            conv2_in_channels: int, conv2_out_channels: int, conv2_kernel_size: int, conv2_stride: int,
            pool1_kernel_size: int, dropout1_p: float, dropout2_p: float,
            fullconn1_in_features: int, fullconn1_out_features: int, fullconn2_in_features: int, fullconn2_out_features: int,
            adadelta_lr: float, adadelta_rho: float, adadelta_eps: float, adadelta_weight_decay: float,
            dataset_root: str, dataset_download: bool,
            dataloader_mean: tuple, dataloader_std: tuple, dataloader_batch_size: int, dataloader_num_workers: int
            ) -> None:
        super(MNISTConvNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=conv1_in_channels, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, stride=conv1_stride)
        self.conv2 = torch.nn.Conv2d(in_channels=conv2_in_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, stride=conv2_stride)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=pool1_kernel_size)
        self.dropout1 = torch.nn.Dropout2d(p=dropout1_p, inplace=False)
        self.dropout2 = torch.nn.Dropout2d(p=dropout2_p, inplace=False)
        self.fullconn1 = torch.nn.Linear(in_features=fullconn1_in_features, out_features=fullconn1_out_features)
        self.fullconn2 = torch.nn.Linear(in_features=fullconn2_in_features, out_features=fullconn2_out_features)

        self.adadelta_params = {
            'lr': adadelta_lr,
            'rho': adadelta_rho,
            'eps': adadelta_eps,
            'weight_decay': adadelta_weight_decay,
        }

        self.dataset_params = {
            'root': dataset_root,
            'download': dataset_download,
        }

        self.dataloader_params = {
            'mean': dataloader_mean,
            'std': dataloader_std,
            'batch_size': dataloader_batch_size,
            'num_workers': dataloader_num_workers,
        }

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(input=x)
        x = self.conv2(x)
        x = F.relu(input=x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.flatten(input=x, start_dim=1)
        x = self.fullconn1(x)
        x = F.relu(input=x)
        x = self.dropout2(x)
        x = self.fullconn2(x)
        return F.log_softmax(input=x, dim=1)
    
    def _common_step(self, batch: Any, log_name: str,
            log_on_step: Any = None, log_on_epoch: Any = None, log_prog_bar: bool = False
            ) -> Union[Tensor, Dict[str, Any], None]:
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(input=y_pred, target=y)

        self.log(name=log_name, value=loss, prog_bar=log_prog_bar, on_step=log_on_step, on_epoch=log_on_epoch)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
        return self._common_step(batch=batch, log_name="train_loss", log_prog_bar=True, log_on_epoch=True)
    
    def validation_step(self, batch: Any, batch_idx: int) -> Union[Tensor, Dict[str, Any], None]:
        return self._common_step(batch=batch, log_name="val_loss", log_prog_bar=True, log_on_step=True, log_on_epoch=True)
    
    def validation_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]] ) -> None:
        return torch.stack(tensors=outputs).mean() # NOTE: Average loss
    
    def test_step(self, batch: Any, batch_idx: int) -> Union[Tensor, Dict[str, Any], None]:
        return self._common_step(batch=batch, log_name="test_loss", log_prog_bar=True, log_on_step=True, log_on_epoch=True)
    
    def test_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]] ) -> None:
        return torch.stack(tensors=outputs).mean() # NOTE: Average loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adadelta(params=self.parameters(),
            lr=self.adadelta_params['lr'],
            rho=self.adadelta_params['rho'],
            eps=self.adadelta_params['eps'],
            weight_decay=self.adadelta_params['weight_decay'])
    
    def _get_dataloader(self, train: bool) -> Any:
        transform_objects = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.dataloader_params['mean'], std=self.dataloader_params['std'])
        ]
        transform = torchvision.transforms.Compose(transforms=transform_objects)
        dataset = torchvision.datasets.MNIST(root=self.dataset_params['root'],
            train=train,
            download=self.dataset_params['download'],
            transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
            batch_size=self.dataloader_params['batch_size'],
            num_workers=self.dataloader_params['num_workers'])
        return dataloader

    def train_dataloader(self) -> Any:
        return self._get_dataloader(train=True)
    
    def val_dataloader(self) -> Any:
        return self._get_dataloader(train=True)
    
    def test_dataloader(self) -> Any:
        return self._get_dataloader(train=False)
