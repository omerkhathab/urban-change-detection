import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
import torchvision.transforms.functional as TF
from tqdm import tqdm
from collections import OrderedDict

# Model Definition
class DualTaskLateFusionSiameseUnet(nn.Module):
    def __init__(self):
        super(DualTaskLateFusionSiameseUnet, self).__init__()

        n_classes = 1
        self.topology = [64, 128, 256, 512]
        self.s1_bands = [0, 1]
        self.s2_bands = [2, 1, 0, 3]

        # sar
        self.inc_sar = InConv(len(self.s1_bands), self.topology[0], DoubleConv)
        self.encoder_sar = Encoder(self.topology)
        self.decoder_sar_change = Decoder(self.topology)
        self.decoder_sar_sem = Decoder(self.topology)
        self.outc_sar_change = OutConv(self.topology[0], n_classes)
        self.outc_sar_sem = OutConv(self.topology[0], n_classes)

        # optical
        self.inc_optical = InConv(len(self.s2_bands), self.topology[0], DoubleConv)
        self.encoder_optical = Encoder(self.topology)
        self.decoder_optical_change = Decoder(self.topology)
        self.decoder_optical_sem = Decoder(self.topology)
        self.outc_optical_change = OutConv(self.topology[0], n_classes)
        self.outc_optical_sem = OutConv(self.topology[0], n_classes)

        # fusion
        self.outc_fusion_change = OutConv(2 * self.topology[0], n_classes)
        self.outc_fusion_sem = OutConv(2 * self.topology[0], n_classes)

    @ staticmethod
    def difference_features(features_t1: torch.Tensor, features_t2: torch.Tensor):
        features_diff = []
        for f_t1, f_t2 in zip(features_t1, features_t2):
            f_diff = torch.sub(f_t2, f_t1)
            features_diff.append(f_diff)
        return features_diff

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:

        # sar
        # encoding
        s1_t1, s1_t2 = x_t1[:, :len(self.s1_bands), ], x_t2[:, :len(self.s1_bands), ]
        x1_sar_t1 = self.inc_sar(s1_t1)
        features_sar_t1 = self.encoder_sar(x1_sar_t1)
        x1_sar_t2 = self.inc_sar(s1_t2)
        features_sar_t2 = self.encoder_sar(x1_sar_t2)
        features_sar_diff = self.difference_features(features_sar_t1, features_sar_t2)

        # decoding change
        x2_sar_change = self.decoder_sar_change(features_sar_diff)
        out_sar_change = self.outc_sar_change(x2_sar_change)

        # deconding semantics
        x2_sar_sem_t1 = self.decoder_sar_sem(features_sar_t1)
        out_sar_sem_t1 = self.outc_sar_sem(x2_sar_sem_t1)

        x2_sar_sem_t2 = self.decoder_sar_sem(features_sar_t2)
        out_sar_sem_t2 = self.outc_sar_sem(x2_sar_sem_t2)

        # optical
        # encoding
        s2_t1, s2_t2 = x_t1[:, len(self.s1_bands):, ], x_t2[:, len(self.s1_bands):, ]
        x1_optical_t1 = self.inc_optical(s2_t1)
        features_optical_t1 = self.encoder_optical(x1_optical_t1)
        x1_optical_t2 = self.inc_optical(s2_t2)
        features_optical_t2 = self.encoder_optical(x1_optical_t2)
        features_optical_diff = self.difference_features(features_optical_t1, features_optical_t2)

        # decoding change
        x2_optical_change = self.decoder_optical_change(features_optical_diff)
        out_optical_change = self.outc_optical_change(x2_optical_change)

        # deconding semantics
        x2_optical_sem_t1 = self.decoder_optical_sem(features_optical_t1)
        out_optical_sem_t1 = self.outc_optical_sem(x2_optical_sem_t1)

        x2_optical_sem_t2 = self.decoder_optical_sem(features_optical_t2)
        out_optical_sem_t2 = self.outc_optical_sem(x2_optical_sem_t2)

        # fusion
        x2_fusion_change = torch.concat((x2_sar_change, x2_optical_change), dim=1)
        out_fusion_change = self.outc_fusion_change(x2_fusion_change)

        # fusion semantic decoding
        x2_fusion_sem_t1 = torch.concat((x2_sar_sem_t1, x2_optical_sem_t1), dim=1)
        out_fusion_sem_t1 = self.outc_fusion_sem(x2_fusion_sem_t1)

        x2_fusion_sem_t2 = torch.concat((x2_sar_sem_t2, x2_optical_sem_t2), dim=1)
        out_fusion_sem_t2 = self.outc_fusion_sem(x2_fusion_sem_t2)

        return out_fusion_change, out_sar_sem_t1, out_sar_sem_t2, out_optical_sem_t1, out_optical_sem_t2,\
            out_fusion_sem_t1, out_fusion_sem_t2

class Encoder(nn.Module):
    def __init__(self, topology):
        super(Encoder, self).__init__()
        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs

class Decoder(nn.Module):
    def __init__(self, topology):
        super(Decoder, self).__init__()
        # Variable scale
        n_layers = len(topology)
        up_topo = [topology[0]]  # topography upwards
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]  # last layer
            up_topo.append(out_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:

        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        return x1

# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def load_checkpoint(cfg_name: str, device: torch.device, folder: str):
    net = nn.DataParallel(DualTaskLateFusionSiameseUnet())
    net.to(device)

    save_file = f'{folder}{cfg_name}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    net.load_state_dict(checkpoint['network'])

    return net

# GeoTIFF I/O Functions
def read_tif(file):
    with rasterio.open(file) as dataset:
        arr = dataset.read()
        transform = dataset.transform
        crs = dataset.crs
    return arr.transpose((1, 2, 0)), transform, crs

def write_tif(file, arr, transform, crs):
    if len(arr.shape) == 3:
        height, width, bands = arr.shape
    else:
        height, width = arr.shape
        bands = 1
        arr = arr[:, :, None]
    with rasterio.open(file, 'w', driver='GTiff', height=height, width=width,
                      count=bands, dtype=arr.dtype, crs=crs,
                      transform=transform) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)

# Dataset Class
class SceneInferenceDataset(torch.utils.data.Dataset):

  def __init__(self, s2_t1: np.ndarray, s2_t2: np.ndarray, s1_t1: np.ndarray,
               s1_t2: np.ndarray, tile_size: int = 128):
      super().__init__()

      self.tile_size = tile_size
      self.s1_t1, self.s1_t2 = s1_t1, s1_t2
      self.s2_t1, self.s2_t2 = s2_t1, s2_t2
      print(s1_t1.shape, s1_t2.shape, s2_t1.shape, s2_t2.shape)

      self.s1_bands = [0, 1]
      self.s2_bands = [2, 1, 0, 3]

      m, n, _ = self.s1_t1.shape

      self.m = (m // self.tile_size) * self.tile_size
      self.n = (n // self.tile_size) * self.tile_size
      print(f"m: {self.m}, n: {self.n}")
      self.tiles = []
      for i in range(0, self.m , self.tile_size):
          for j in range(0, self.n, self.tile_size):
              tile = {
                  'i': i,
                  'j': j,
              }
              self.tiles.append(tile)
      self.length = len(self.tiles)

  def __getitem__(self, index):
    tile = self.tiles[index]
    i, j = tile['i'], tile['j']

    tile_s1_t1 = self.s1_t1[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
    tile_s1_t2 = self.s1_t2[i:i + self.tile_size, j:j + self.tile_size, self.s1_bands]
    tile_s2_t1 = self.s2_t1[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]
    tile_s2_t2 = self.s2_t2[i:i + self.tile_size, j:j + self.tile_size, self.s2_bands]

    tile_s1_t1, tile_s1_t2 =  TF.to_tensor(tile_s1_t1), TF.to_tensor(tile_s1_t2)
    tile_s2_t1, tile_s2_t2 = TF.to_tensor(tile_s2_t1), TF.to_tensor(tile_s2_t2)

    x_t1 = torch.concat((tile_s1_t1, tile_s2_t1), dim=0)
    x_t2 = torch.concat((tile_s1_t2, tile_s2_t2), dim=0)

    item = {
        'x_t1': x_t1,
        'x_t2': x_t2,
        'i': i,
        'j': j,
    }
    return item

  def get_arr(self, c: int = 1):
    if c == 1:
      return np.zeros((self.m, self.n), dtype=np.uint8)
    else:
      return np.zeros((self.m, self.n, c), dtype=np.uint8)

  def __len__(self):
      return self.length

  def __str__(self):
      return f'Dataset with {self.length} samples.'

# Change Detection Function
def run_change_detection(s1_t1_path, s1_t2_path, s2_t1_path, s2_t2_path, output_path, model_path, tile_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint("model", device, model_path)

    s1_t1, transform, crs = read_tif(s1_t1_path)
    s1_t1 = np.nan_to_num(s1_t1).astype(np.float32)

    s1_t2, *_ = read_tif(s1_t2_path)
    s1_t2 = np.nan_to_num(s1_t2).astype(np.float32)

    s2_t1, *_ = read_tif(s2_t1_path)
    s2_t1 = np.nan_to_num(s2_t1).astype(np.float32)

    s2_t2, *_ = read_tif(s2_t2_path)
    s2_t2 = np.nan_to_num(s2_t2).astype(np.float32)

    dataset = SceneInferenceDataset(s1_t1, s1_t2, s2_t1, s2_t2, tile_size)
    pred = dataset.get_arr(3)

    net.eval()
    for index in tqdm(range(len(dataset))):
        tile = dataset.__getitem__(index)

        x_t1 = tile['x_t1'].to(device)
        x_t2 = tile['x_t2'].to(device)
        with torch.no_grad():
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))

        # Unpack outputs
        logits_ch = logits[0]
        logits_sem_t1 = logits[5]
        logits_sem_t2 = logits[6]

        # Post-process
        y_pred_ch = torch.sigmoid(logits_ch).squeeze().cpu().numpy()
        y_pred_sem_t1 = torch.sigmoid(logits_sem_t1).squeeze().cpu().numpy()
        y_pred_sem_t2 = torch.sigmoid(logits_sem_t2).squeeze().cpu().numpy()

        y_pred_ch = np.clip(y_pred_ch * 100, 0, 100).astype(np.uint8)
        y_pred_sem_t1 = np.clip(y_pred_sem_t1 * 100, 0, 100).astype(np.uint8)
        y_pred_sem_t2 = np.clip(y_pred_sem_t2 * 100, 0, 100).astype(np.uint8)

        i, j = tile['i'], tile['j']
        pred[i:i + tile_size, j:j + tile_size, 0] = y_pred_ch
        pred[i:i + tile_size, j:j + tile_size, 1] = y_pred_sem_t1
        pred[i:i + tile_size, j:j + tile_size, 2] = y_pred_sem_t2

    write_tif(output_path, pred, transform, crs)
    return output_path