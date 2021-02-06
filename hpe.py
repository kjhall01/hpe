from pdf2image import convert_from_path
import cv2
import pytesseract as pt
import argparse, sys
from pathlib import Path
from PIL import Image

def displace(oldpath, rt='imgs'):
	if not oldpath.is_dir() and str(oldpath) == rt:
		oldpath.mkdir()
	elif oldpath.is_dir():
		displace(Path(str(oldpath) + '_old'), rt=rt)
	else:
		Path(str(oldpath)[:-4]).rename(oldpath)
		displace(Path(str(oldpath)[:-4]), rt=rt)

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='file to convert')
parser.add_argument('--angle', '-a', type=float, default=0.0, help='angle to rotate pdf counter clockwise')
parser.add_argument('--iter', type=int, default=1, help='test')
parser.add_argument('--img', '-i', type=str, default='no', help='path of img file to convert')
parser.add_argument('--output', '-o', type=str, default='out.txt', help='path of output file')

args = parser.parse_args()


if args.img == 'no':

	pages=convert_from_path(args.file)
	ndx, ndx2, pwd = 0,0, Path('.')
	displace(pwd / 'imgs', rt='imgs')

	n_imgs = len(pages)
	steps = [n_imgs / 10 * i for i in range(11)]
	print('converting imgs [          ]', end='\r')
	for page in pages:
		ndx += 1
		if ndx >= steps[ndx2]:
			print('converting imgs [' +ndx2*'*' + ' '*(10-ndx2) +']', end='\r')
			ndx2+=1
		page.save("./imgs/page_{}.jpg".format(ndx),"jpeg")
	print('converting imgs [**********]')

else:
	print('{} must already exist'.format(args.img))

ndx, ndx2, pwd = 0,0, Path('.')
displace(pwd / 'gray', rt='gray')

def conv_gray(imgn):
	img = cv2.imread(imgn)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	filename = './gray/' + imgn.split('/')[-1]
	cv2.imwrite(filename, gray)

if args.img != 'no':
	if Path(args.img).is_file():
		conv_gray(args.img)
		print('graying imgs [**********]')
	else:
		print('Not a valid file')
		sys.exit()
else:
	print('graying imgs [          ]', end='\r')
	p = Path(pwd / 'imgs').glob('**/*')
	files = [str(x.absolute()).split('/')[-1] for x in p if x.is_file()]
	for file in files:
		ndx += 1
		if ndx >= steps[ndx2]:
			print('graying imgs [' +ndx2*'*' + ' '*(10-ndx2) +']', end='\r')
			ndx2+=1
		sys.stdout.flush()
		conv_gray(str((Path(pwd / 'imgs') / file)))
	print('graying imgs [**********]')

ndx, ndx2, pwd = 0,0, Path('.')

if args.img != 'no':
	f = open(args.output, 'w')
	if Path('gray/' + args.img.split('/')[-1]).is_file():
		text = pt.image_to_string(Image.open('gray/' + args.img.split('/')[-1]))
		print('transcribing imgs [**********]')
		f.write(text)
		f.close()
	else:
		print('Not a valid file')
		sys.exit()

else:
	print('transcribing imgs [          ]', end='\r')
	f = open(args.output, 'w')
	p = Path(pwd / 'gray').glob('**/*')
	files = [str(x.absolute()).split('/')[-1] for x in p if x.is_file()]
	for file in files:
		ndx += 1
		if ndx >= steps[ndx2]:
			print('transcribing imgs [' +ndx2*'*' + ' '*(10-ndx2) +']', end='\r')
			ndx2+=1
		sys.stdout.flush()
		text = pt.image_to_string( Image.open('gray/' + file))
		f.write(text)
		f.write('\n\n\n')
	print('transcribing imgs [**********]')
	f.close()
