import torch.nn as nn
import torch
import torch.nn.functional as F

from reyes_tan_utilities import *

from src.dataset.urmp.urmp_sample import *
from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

from src.inference.inference import merge_batches

import torchaudio
#reyes_models patterned after https://github.com/anonymous-16/a-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis/tree/main/src/models

#Temporary
torch.autograd.set_detect_anomaly(True)

class Encoder(nn.Module):
	
	def __init__(self, config):
		#needed else "AttributeError: cannot assign module before Module.__init__() call"
		super(Encoder, self).__init__()
		
		#Unpack the config
		
		#Channels
		self.in_channels = config['in_channels']
		self.out_channels = 2
		
		#Momentum
		self.momentum = config['momentum']
		
		
		
		#initialize the blocks
		#Each block is 
		#conv - bn -film
		self.num_blocks = config['num_blocks']
		self.conv_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		self.film_layers = nn.ModuleList()
		
		#Source code has an extra one
		for i in range(self.num_blocks + 1):
			self.conv_layers.append(nn.Conv2d(in_channels = self.in_channels, \
							out_channels = self.out_channels, \
							kernel_size = (3,3), \
							stride = (1,1), \
							padding = (1,1), \
							bias = False))
			self.bn_layers.append(nn.BatchNorm2d(self.out_channels))
			
			self.in_channels = self.out_channels
			self.out_channels *= 2
			
	def forward(self, input):
	
		#There model seems to be not a strict Unet in that the concatenation only happens once
		#Update. It is a list which is unpacked decoder side. It is indeed a strcit Unet
		x = input
		concat_tensors = []
		for i in range(self.num_blocks):
			x = self.conv_layers[i](x)
			x = self.bn_layers[i](x)
			concat_tensors.append(x)
			x = F.avg_pool2d(x, kernel_size = (1, 2))
			
		x_new = self.conv_layers[self.num_blocks](x).clone()
		return x_new, concat_tensors
		
		
class Decoder(nn.Module):

	def __init__(self, config):
		super(Decoder, self).__init__()
		
		#Unpack config
		
		
		layers = nn.ModuleList()
		self.in_channels = config['in_channels']
		self.momentum = config['momentum']
		self.num_blocks = config['num_blocks']

		#Layers lists
		self.conv_tr_layers = nn.ModuleList()
		self.bn_layers = nn.ModuleList()
		
		#
		self.conv_layers = nn.ModuleList()
		self.bn_layers_2 = nn.ModuleList()
		
		#Decoder building blocks
		self.in_channels = self.in_channels
		self.out_channels = self.in_channels//2
		for i in range(self.num_blocks):
			
			self.conv_tr_layers.append(torch.nn.ConvTranspose2d(in_channels = self.in_channels, \
				out_channels = self.out_channels, \
				kernel_size = (3,3), \
				stride = (1,2), \
				padding = (0,0), \
				output_padding = (0,0),
				bias = False)
				)
			
			self.bn_layers.append(nn.BatchNorm2d(self.out_channels, momentum = self.momentum))
			
			
			
			#The 2 "channels" are the dec output and the corresponding concatenation
			self.conv_layers.append(nn.Conv2d(in_channels = self.out_channels*2, \
				out_channels = self.out_channels, \
				kernel_size = (3, 3)))
			
			self.bn_layers_2.append(\
				nn.BatchNorm2d(self.out_channels,\
					momentum = self.momentum)
				)
				
			#the next in channels will be divded by 2
			self.in_channels = self.out_channels
			self.out_channels = self.out_channels//2
			
		#The last layers
		#Bottom as they call it
		self.bottom = nn.Conv2d(in_channels = self.in_channels, \
							out_channels = self.out_channels, \
							kernel_size = (1,1), \
							stride = (1,1), \
							bias = True)
	def forward(self, input, concat_tensors):
		
		x = input
		for i in range(self.num_blocks):
		
			print(x.size())
			x = self.conv_tr_layers[i](x)
			x = self.bn_layers[i](x)
			x = F.relu_(x)
			
			
			#Pruning to match
			x = x[:, :, 1:-1, : -1]
			
			#print(x.size())
			#print(concat_tensors[-i-1].size())
			#Since they the unpacking is FIFO use -i-1
			#x = torch.cat((x, concat_tensors[-i-1]), dim = 1)
			
			#x = self.conv_layers[i](x)
			#x = self.bn_layers_2[i](x)
			print(x.size(), "x per dec cycle")
			
		x = self.bottom(x)
		return x



if __name__ == "__main__":
	
	config_enc = {'num_blocks' : 1, \
				'in_channels' : 1, \
				'momentum' : 0.01}
				
	config_dec = {'num_blocks' : 1, \
				'in_channels' : 4, \
				'momentum' : 0.01}
	
	#n_fft is the same as win_length
	config_spec = {'center' : True, \
				'freeze_parameters' : True, \
				'n_fft' : 256, \
				'hop_length' : 160, \
				'pad_mode' : "reflect", \
				'window' : "hann", \
				'win_length' : 256}
				
				
	# should be 256 not 128
	#n_fft is the same as win_length
	config_s2w = {'fps' : 100, \
				'samp_rate' : 16000, \
				'window' : "hann", \
				'n_fft' : 256, \
				'hop_length' : 160, \
				'win_length' : 256, \
				'power' : 1, \
				'normalized' : False, \
				'n_iter' : 200, \
				'momentum' : 0, \
				'rand_init' : False}

	urmpsamp = torch.randn((2, 1, 48000))
	newinitial = wav2spec(config_spec, urmpsamp)
	
	print(urmpsamp.shape, newinitial.shape, "urmpsamp, newitinital shapes reyesmodels.py")
	
	enc = Encoder(config_enc)
	dec = Decoder(config_dec)
	
	initial = torch.randn((2,1,301,128))
	out, out_conc = enc(initial)
		
	
	final = dec(out, out_conc)
	
	print(final.size(), "final size")
	
	finalwav = spec2wav(final, config_s2w)
	print(finalwav.size(), "final wav size reyesmodels.py")
	
	
	### Data loading taken from their code

	urmp_data = UrmpSample('utilities_taken_as_is/urmp.cfg', 'train')
	print(urmp_data)
	
	urmp_loader = DataLoader(urmp_data, \
			batch_size = 2, \
			shuffle = False, \
			num_workers = 1, \
			pin_memory = True, \
			persistent_workers = False,
			collate_fn = urmp_data.get_collate_fn())
	

	parameters = {}
	parameters['enc'] = list(enc.parameters())
	parameters['dec'] = list(dec.parameters())
	
	optimizers = []
	#since resume epoch is 0
	for param in parameters:
		optimizer = torch.optim.Adam(parameters[param], \
						lr = 5e-4/(2**(0 // 100)))
		optimizers.append({'mode' : param, 'opt' : optimizer, 'name' : param})
	
	
	sep_spec = merge_batches(final, duration_axis =-2)
	print(sep_spec.shape)
	sep_spec = sep_spec.unsqueeze(dim = 0)
	print(sep_spec.shape)
	finalwav = spec2wav(sep_spec, config_s2w)
	print(finalwav.shape)
	
	#wavs are 2d
	finalwav = finalwav.detach().squeeze(dim = 0)
	#Their sample rate is 16k apparently
	torchaudio.save("first.wav", finalwav, 16000)
	
	for urmp_batch in urmp_loader:
		urmp_batch = urmp_batch[0]
		print(urmp_batch.shape, "size of urmpbatch[0] r_t_models.oy")
		urmp_batch = urmp_batch[1,:,:]
		print(urmp_batch.shape, "size after taking only first of abtch r_t_models.py")
		torchaudio.save("datasampleorig2.wav", urmp_batch, 16000)
		
		urmp_batch = urmp_batch.unsqueeze(dim = 0)
		urmp_batch = wav2spec(config_spec, urmp_batch)
		print(urmp_batch.shape, "size after wav2spec r_t_models.py")
		
		urmp_batch = spec2wav(urmp_batch, config_s2w)
		print(urmp_batch.shape, "size after spec2wav r_t_models.py")
		urmp_batch = urmp_batch.detach().squeeze(dim = 0)
		torchaudio.save("datasample2.wav", urmp_batch, 16000)
		break
	'''

	for i_batch, urmp_batch in enumerate(urmp_loader):
		print(i_batch, "Doing this batch")
		loss = []
		
		out, out_conc = enc(initial)
		final = dec(out, out_conc)
		
		for j in range(len(optimizers)):
			op = optimizers[j]['opt']
			
			op.zero_grad()
			loss.append(torch.abs(out[0,0,0,0])) #hard code for now.
			loss[j].backward(retain_graph = True)
			
		#apparently you can't take a step yet until you process everything
		for j in range(len(optimizers)):
			op = optimizers[j]['opt']
			op.step()
			op.zero_grad()
		del loss
		
		if i_batch == 10:
			break
	print("reached sample")
	'''
	
	
	
	
	
	
	
	
	
	
	
	