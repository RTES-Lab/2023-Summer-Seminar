import numpy as np

# Numpy
@profile
def np_fft_2d(img):
	fft_img = np.fft.fft2(img)
	shifted_img = np.fft.fftshift(fft_img)
	return shifted_img

@profile
def np_ifft_2d(fft_img):
	ishifted_img = np.fft.ifftshift(fft_img)
	ifft_img = np.fft.ifft2(ishifted_img)
	return ifft_img
	
img_size = 1000
image_h = np.random.randn(img_size,img_size)
fft_img = np_fft_2d(image_h)
ifft_img = np_ifft_2d(fft_img)

