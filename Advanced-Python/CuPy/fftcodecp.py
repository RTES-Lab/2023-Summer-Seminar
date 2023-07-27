import cv2
import numpy as np
import cupy as cp
from numpy.fft import fftshift, ifftshift

# 트랙바 이벤트 처리 함수 선언 ---①

# 이미지 선언과 푸리에 변환
img = cv2.imread('./images/crossroad.jpg', cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 이미지 읽기
img = img / 255  # 이미지를 0~1 사이 값으로 정규화

h, w = img.shape
H = int(h / 2)
W = int(w / 2)

# cupy 행렬로 변환
gimg = cp.asarray(img)

# 푸리에 변환을 cupy로 수행
fgimg = fftshift(cp.fft.fft2(gimg))

imgspectrum = 20 * np.log10(cp.asnumpy(np.abs(fgimg)))  # cupy 배열을 numpy 배열로 변환 후 계산
backimg = cv2.imread('./img/blank_500.jpg')

win_name = 'Trackbar'
cv2.namedWindow(win_name)


def onChange(x):
    print(x)


cv2.createTrackbar('A', win_name, 0, 500, onChange)
cv2.createTrackbar('C', win_name, 0, 500, onChange)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
    A = cv2.getTrackbarPos('A', win_name)
    C = cv2.getTrackbarPos('C', win_name)
    if A > C:
        maskA = np.zeros_like(imgspectrum).copy()
        maskB = np.ones_like(imgspectrum).copy()

        B = int((A / 5) * 3.30)
        cv2.rectangle(maskA, (W - A // 2, H - B // 2), (W + A // 2, H + B // 2), (255, 255, 255), -1)
        ##A사각형. 사각형 내부가 1로 차있음
        D = int((C / 5) * 3.30)
        cv2.rectangle(maskB, (W - C // 2, H - D // 2), (W + C // 2, H + D // 2), (0, 0, 0), -1)
        ##C사각형. 사각형 내부가 0으로 차있음
        ##흰색 마스킹 사각형 길이는 중심에서 양쪽으로 A/2, B/2 씩 떨어짐
        ##곱해서 0이 되면 값을 되돌릴 수 없다.
        ##트랙바에서 얻은 값으로 사각형 생성 후 이를 AND 연산으로 최종 MASK 생성
        maskEnd = cp.asarray(maskA * maskB)  # Convert to cupy array
    else:
        maskA = np.zeros_like(imgspectrum).copy()
        maskB = np.ones_like(imgspectrum).copy()

        B = int((C / 5) * 3.30)
        cv2.rectangle(maskA, (250 - C // 2, 150 - B // 2), (250 + C // 2, 150 + B // 2), (255, 255, 255), -1)
        ##A사각형. 사각형 내부가 1로 차있음
        D = int((A / 5) * 3.30)
        cv2.rectangle(maskB, (250 - A // 2, 150 - D // 2), (250 + A // 2, 150 + D // 2), (0, 0, 0), -1)
        ##C사각형. 사각형 내부가 0으로 차있음
        ##흰색 마스킹 사각형 길이는 중심에서 양쪽으로 A/2, B/2 씩 떨어짐
        ##곱해서 0이 되면 값을 되돌릴 수 없다.
        ##트랙바에서 얻은 값으로 사각형 생성 후 이를 AND 연산으로 최종 MASK 생성
        maskEnd = cp.asarray(maskA * maskB)  # Convert to cupy array

    # 최종 마스크와 스펙트럼을 AND 연산으로 마스킹된 스펙트럼 이미지 생성
    masked = fgimg * (maskEnd / 255)
    maskedp = (np.log10(np.abs(masked.get()) + 1))
    maskedp = (maskedp - np.min(maskedp)) / (np.max(maskedp) - np.min(maskedp))

    f_ishift = cp.fft.ifftshift(masked)
    img_back = cp.fft.ifft2(f_ishift)
    img_back = np.abs(img_back.get())  # cupy 배열을 numpy 배열로 변환
    numpy_horizontal = np.hstack((img_back, maskedp))

    cv2.imshow(win_name, numpy_horizontal)  # 새 이미지 창에 표시

cv2.destroyAllWindows()
