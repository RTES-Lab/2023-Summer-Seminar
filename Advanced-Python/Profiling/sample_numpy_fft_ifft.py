import numpy as np

# Numpy
def np_fft_2d(img):
	'''
	2차원 이미지를 입력으로 받아서 fft, fftshift를 수행하는 numpy 기반 함수
	[param]
	img: np.array
	2차원 numpy array
	[return]
	shifted_img: np.array
	입력 이미지를 FFT, fftshift한 numpy array
	'''
	# 2D FFT 수행
	fft_img = np.fft.fft2(img)
	# FFT 결과를 fftshift로 시프트
	shifted_img = np.fft.fftshift(fft_img)

	return shifted_img

def np_ifft_2d(fft_img):
	'''
	fft된 2d 이미지를 다시 역변환하는 함수 (numpy)
	[param]
	fft_img: np.array
	[return]
	ifft_img: np.array
	'''
	ishifted_img = np.fft.ifftshift(fft_img)
	ifft_img = np.fft.ifft2(ishifted_img)

	return ifft_img

if __name__ == '__main__':
	img_size = 1000
	image_h = np.random.randn(img_size,img_size)

	fft_img = np_fft_2d(image_h)
	ifft_img = np_ifft_2d(fft_img)

