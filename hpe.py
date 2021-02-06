from pdf2image import convert_from_path
import cv2
import pytesseract as pt
import argparse, sys
from pathlib import Path
from PIL import Image
import numpy as np

def rotate_image(image, angle):
	#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def alignImages(im1, im2):
	MAX_FEATURES = 500
	GOOD_MATCH_PERCENT = 0.15

	im1Gray = im1
	im2Gray = im2
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)
	matches.sort(key=lambda x: x.distance, reverse=False)
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]
	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	height, width = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))
	return im1Reg, h


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
parser.add_argument('--reg', '-r', type=bool, default=False, help='Whether to use registration to first file or not')

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
		page.save("./imgs/page_{:03}.jpg".format(ndx),"jpeg")
	print('converting imgs [**********]')

else:
	print('{} must already exist'.format(args.img))

ndx, ndx2, pwd = 0,0, Path('.')
displace(pwd / 'gray', rt='gray')

def conv_gray(imgn, angle=0.0, reg=False, ref=None):
	img = cv2.imread(imgn)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	if angle != 0.0 and not reg:
		gray = rotate_image(gray, angle)
	filename = './gray/' + imgn.split('/')[-1]
	if reg:
		imReg, h = alignImages(gray, ref)
		cv2.imwrite(filename, imReg)
		return imReg
	else:
		cv2.imwrite(filename, gray)
		return gray



#graying , rotating and registering files
if args.img != 'no': #if only one file, just rotate by n degress
	if Path(args.img).is_file():
		conv_gray(args.img, angle=args.angle)
		print('graying imgs [**********]')
	else:
		print('Not a valid file')
		sys.exit()
else: # if multiple, rotate first file by n and then register to first file
	print('graying & rotating imgs [          ]', end='\r')
	p = Path(pwd / 'imgs').glob('**/*')
	files = [str(x.absolute()).split('/')[-1] for x in p if x.is_file()]
	file0 = files.pop(0)
	ndx += 1
	ref = conv_gray(str((Path(pwd / 'imgs') / file0)), angle=args.angle)
	for file in files:
		ndx += 1
		if ndx >= steps[ndx2]:
			print('graying & rotating imgs [' +ndx2*'*' + ' '*(10-ndx2) +']', end='\r')
			ndx2+=1
		sys.stdout.flush()
		conv_gray(str((Path(pwd / 'imgs') / file)), reg=args.reg, ref=ref )
	print('graying & rotating imgs [**********]')

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
