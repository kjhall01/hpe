# HPE (Harry's PDF Extractor) by Kyle Hall (hallkjc01@gmail.com)

## Usage: python hpe.py
	* [-h] Print details of hpe.py usage
	* [--file FILE] path to pdf file to transcribe
	* [--angle ANGLE] Degrees Counter-Clockwise rotation to align first page of pdf to vertical
	* [--iter ITER] For testing recursive directory renaming
	* [--img IMG] if you have one image to transcribe, path to that image. separate use case.
	* [--output OUTPUT] text file output path
	* [--reg REG] -whether to use auto-registration of images to angle of image of first page. May cut off  some parts of pages
	* [--noise NOISE] whether to use noise reduction strategy

## Works Cited:  
	* OpenCV
	* PyTesseract
	* NumPy
	* argparse
	* Pathlib
	* pdf2image
	* PIL
	* image rotation: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
	* image registration: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
	* noise reduction: https://stackoverflow.com/questions/42065405/remove-noise-from-threshold-image-opencv-python

good luck ! <3 <3 <3
