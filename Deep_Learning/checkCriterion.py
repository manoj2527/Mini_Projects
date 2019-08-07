import argparse, torchfile, torch
from Criterion import Criterion


if __name__ == '__main__':

	parser = argparse.ArgumentParser("checkCriterion.py")
	parser.add_argument("-i", dest="inp" , help="/path/to/input.bin", type=str)
	parser.add_argument("-t", dest="target" , help="/path/to/target.bin", type=str)
	parser.add_argument("-ig", dest="gradInp" , help="/path/to/gradInput.bin", type=str)
	args = parser.parse_args()

	Inp = torchfile.load(args.inp)
	Inp = torch.from_numpy(Inp).double()


	Target = torchfile.load(args.target)
	Target = torch.from_numpy(Target).long()
	Target -= 1

	criterion = Criterion()
	loss = criterion.forward(Inp, Target)
	print(loss)

	gradInp = criterion.backward(Inp, Target)


	with open(args.gradInp, 'wb') as f: 
		torch.save(gradInp, f)
		f.close()
