import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
		super(ConvBlock, self).__init__()
		# kernel (3,3,3) avec padding (1,1,1) : chaque angle voit ses voisins ±1
		# ce qui garantit qu'un angle n peut influencer n-1 et n+1 (bave MLC locale)
		self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		return x

class DownSampling(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DownSampling, self).__init__()
		# Anisotropic: no downsampling on angle dim (leaf leakage is local)
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0))
		self.bn = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.bn(self.conv(x)))
		return x

class UpSampling(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UpSampling, self).__init__()
		# Anisotropic: upsample only detector dims
		self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2))
		self.bn = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.bn(self.upsample(x)))
		return x

class Vnet(nn.Module):
	def __init__(self, in_channels=2, out_channels=1, base_filters=16):
		"""
		VNet for sinogram-to-sinogram prediction.
		Input:  [B, in_channels, 1300, 64, 64]  (angles, det_z, det_x)
		Output: [B, out_channels, 1300, 64]      (projection along det_x)

		3 encoder levels with anisotropic downsampling:
		  angles dim stays at 1300 throughout (leaf leakage is local)
		  detector dims: 64 -> 32 -> 16 -> 8 (bottleneck)
		"""
		super(Vnet, self).__init__()

		# Encoding path
		self.encoder_block1 = ConvBlock(in_channels, base_filters)
		self.down1 = DownSampling(base_filters, base_filters*2)      # det: 64->32

		self.encoder_block2 = ConvBlock(base_filters*2, base_filters*2)
		self.down2 = DownSampling(base_filters*2, base_filters*4)    # det: 32->16

		self.encoder_block3 = ConvBlock(base_filters*4, base_filters*4)
		self.down3 = DownSampling(base_filters*4, base_filters*8)    # det: 16->8

		# Bottleneck
		self.bottleneck = ConvBlock(base_filters*8, base_filters*8)

		# Decoding path
		self.up3 = UpSampling(base_filters*8, base_filters*4)
		self.decoder_block3 = ConvBlock(base_filters*8, base_filters*4)

		self.up2 = UpSampling(base_filters*4, base_filters*2)
		self.decoder_block2 = ConvBlock(base_filters*4, base_filters*2)

		self.up1 = UpSampling(base_filters*2, base_filters)
		self.decoder_block1 = ConvBlock(base_filters*2, base_filters)

		# Output layer
		self.output = nn.Conv3d(base_filters, out_channels, kernel_size=1)

		self.relu = F.relu

	def forward(self, x):
		# Encoding path with skip connections
		x1 = self.encoder_block1(x)                  # [B, F,   1300, 64, 64]
		x2 = self.encoder_block2(self.down1(x1))     # [B, 2F,  1300, 32, 32]
		x3 = self.encoder_block3(self.down2(x2))     # [B, 4F,  1300, 16, 16]

		# Bottleneck
		bottleneck = self.bottleneck(self.down3(x3)) # [B, 8F,  1300,  8,  8]

		# Decoding path
		d3 = self.up3(bottleneck)
		if d3.size()[2:] != x3.size()[2:]:
			d3 = F.interpolate(d3, size=x3.size()[2:], mode='trilinear', align_corners=False)
		d3 = self.decoder_block3(torch.cat([d3, x3], dim=1))
		del x3

		d2 = self.up2(d3)
		if d2.size()[2:] != x2.size()[2:]:
			d2 = F.interpolate(d2, size=x2.size()[2:], mode='trilinear', align_corners=False)
		d2 = self.decoder_block2(torch.cat([d2, x2], dim=1))
		del x2

		d1 = self.up1(d2)
		if d1.size()[2:] != x1.size()[2:]:
			d1 = F.interpolate(d1, size=x1.size()[2:], mode='trilinear', align_corners=False)
		d1 = self.decoder_block1(torch.cat([d1, x1], dim=1))
		del x1

		output_3D = self.relu(self.output(d1))       # [B, 1, 1300, 64, 64]
		output = torch.sum(output_3D, dim=4, keepdim=True)  # integrate over det_x -> [B, 1, 1300, 64, 1]
		return output

class DosePrediction(nn.Module):
	def __init__(self, base_filters=16, in_channel=2):
		"""
		Sinogram-to-sinogram prediction model.

		The machine applies rotation + table translation to the CT and dose volumes,
		producing sinograms of shape [B, 2, 1300, 64, 64]:
		  - dim 0 (channels) : CT sinogram + dose sinogram
		  - dim 1 (angles)   : 1300 projection angles
		  - dim 2,3 (det_z, det_x) : 64x64 detector pixels per angle

		The model predicts the output sinogram [B, 1, 1300, 64, 1] by:
		  1. Running a 3-level VNet with anisotropic downsampling:
		     - angle dim (1300) stays fixed throughout the network because
		       leaf leakage only couples neighbouring angles (local dependency)
		     - detector dims (64x64) are downsampled normally: 64->32->16->8
		  2. Summing the output over det_x (dim=4) to get a 1D projection per angle,
		     simulating physical ray integration through the volume.
		"""
		super(DosePrediction, self).__init__()
		self.vnet = Vnet(in_channels=in_channel, out_channels=1, base_filters=base_filters)

	def forward(self, input):
		# input: [B, 2, 1300, 64, 64]
		# output: [B, 1, 1300, 64, 1]
		return self.vnet(input)
 
 
