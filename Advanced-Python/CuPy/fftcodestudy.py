import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

# 트랙바 이벤트 처리 함수 선언 ---①

#이미지 선언과 푸리에 변환
img = cv2.imread('./images/crossroad.jpg')
h,w,nColor = img.shape
H = int(h/2)
W = int(w/2)
gimg = cv2.imread('./images/crossroad.jpg', cv2.IMREAD_GRAYSCALE)
gimg = gimg/255
fgimg = fftshift(fft2(gimg))

imgspectrum = 20*np.log10(np.abs(fgimg))
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
    if A>C:
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
        maskEnd = maskA * maskB
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
        maskEnd = maskA * maskB

        # 최종 마스크와 스펙트럼을 AND 연산으로 마스킹된 스펙트럼 이미지 생성
    masked = fgimg * (maskEnd / 255)
    maskedp = (np.log10(abs(masked)+1)); maskedp = (maskedp-np.min(maskedp))/(np.max(maskedp)-np.min(maskedp))


    f_ishift = np.fft.ifftshift(masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    numpy_horizontal = np.hstack((img_back, maskedp))
    # img_back과 스펙트럼을 같은 창에서 가로로 출력

    cv2.imshow(win_name, numpy_horizontal)  # 새 이미지 창에 표시

cv2.destroyAllWindows()


