Thanks for downloading the CERTH Image Blur Dataset

Contributors:
Vasileios Mezarhs (bmezaris@iti.gr)
Eftichia Mavridaki (emavridaki@iti.gr)

In case of using the CERTH Image Blur Dataset in your research,  please refer to: 
"E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", 
Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014."


Description: 

The Training Set consists of:
		630 undistorted (clear) images
		220 naturally-blurred images
		150 artificially-distorted images

The Evaluation Set consists of two individual data sets :
	The Natural Blur Set which consists of:
		589 undistorted (clear) images
		411 naturally-blurred images
		
	The Digital Blur Set
		30 undistorted (clear) images
		450 artificially-blurred images

		
For the creation of the artificially-distorted images of Training Set we applied  the following filters :
	Motion blur filter : 
		* Length = 60, theta = 45 (angle theta in degrees)
		* Length = 100, theta = 0 (angle theta in degrees)
		* Length = 50, theta = 0 (angle theta in degrees)
		* Length = 50, theta = 90 (angle theta in degrees)
		* Length = 50, theta = 45 (angle theta in degrees)
		* Length = 20, theta = 45 (angle theta in degrees)
	Gaussian blur filter:
		* Hsize = 5x5, sigma = default
		* Hsize = 10x10, sigma = default
		* Hsize = 5x5, sigma = 20 (standard deviation)
		* Hsize = 1x50, sigma = 250 (standard deviation)
		* Hsize = 15x15, sigma = 100 (standard deviation)
		* Hsize = 25x25, sigma = 100 (standard deviation)
	Circular averaging blur filter:
		* Radius = 3
		* Radius = 5
		* Radius = 8
		* Radius = 10
		* Radius = 20
		* Radius = 30
		
For the creation of the artificially-distorted images of Evaluation Set we applied the following filters :
	Motion blur filter : 
		* Length = 80, theta = 45 (angle theta in degrees)
		* Length = 100, theta = 0 (angle theta in degrees)
		* Length = 100, theta = 45 (angle theta in degrees)
		* Length = 50, theta = 90 (angle theta in degrees)
	Gaussian blur filter:
		* Hsize = 1x50, sigma = 250 (standard deviation)
		* Hsize = 2x80, sigma = 300 (standard deviation)
		* Hsize = 15x15, sigma = 200 (standard deviation)
		* Hsize = 25x25, sigma = 100 (standard deviation)
		* Hsize = 50x50, sigma = 300 (standard deviation)
	Circular averaging blur filter:
		* Radius = 8
		* Radius = 10
		* Radius = 20
		* Radius = 30
		* Radius = 50
		
		
For all the aforementioned blur filters we used the MATLAB functions fspecial and imfilter.