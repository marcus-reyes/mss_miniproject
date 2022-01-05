from src.dataset.urmp.load_urmp_data import load_test_data

from src.utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader

import torchaudio
import torch
import argparse

from src.utils.utilities import mkdir

parser = argparse.ArgumentParser(description='')
parser.add_argument('--eval_dir', type=str, required=True, help='Output of this program')

	
args = parser.parse_args()
mkdir(args.eval_dir)

testdata = load_test_data(r"data/urmp/")

print(testdata.test_samples())
		
for item in testdata.test_samples():
	print(type(item))
	for k, v in item.items():
		print(k)
		
	print(type(item['instrs']))
	print(len(item['instrs']))
	print(item['instrs'][0])
	
	print(item['mix'].shape)
	print(type(item['mix']))
	#numtowav = item['mix'].astype('float16')
	numtowav = torch.from_numpy(item['mix'])
	print(type(numtowav))
	torchaudio.save(str(args.eval_dir)+"/testsample1.wav", numtowav[0,:,:], 16000)
	torchaudio.save(str(args.eval_dir)+"/testsample2.wav", numtowav[1,:,:], 16000)
	break
#print(len(testdata))