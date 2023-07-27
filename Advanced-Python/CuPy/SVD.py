import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.sparse.linalg
from cupyx.profiler import benchmark
import scipy.sparse.linalg

def PrintSVDResultSize(Umat, Smat, Vmat_T, text):
    print('Umat_' + text + ' size: ', Umat.shape)
    print('Smat_' + text + ' size: ', Smat.shape)
    print('Vmat_T_' + text + ' size: ', Vmat_T.shape)
    print('Number of Multiplication Operations: ', 
          Umat.shape[0]*Umat.shape[1]*Smat.shape[0] + Umat.shape[0]*Vmat_T.shape[0]*Vmat_T.shape[1])

def RestoreImage(Umat, Smat, Vmat_T, text, imgname):
    Smat_Diag = cp.diag(Smat)
    # 아래 방향 행에 지정한 크기만큼 패딩, constant 옵션 통해 0으로 값 설정
    Smat_Diag = cp.pad(Smat_Diag, ((0, Umat.shape[1] - Vmat_T.shape[0]), (0,0)), 'constant', constant_values = 0)
    print(Smat_Diag.shape)
    # 행렬 곱셈
    RestoredImage = Umat @ Smat_Diag @ Vmat_T
    print(RestoredImage.device)
    # opencv에서는 (열, 행)으로 사용하므로 transpose함
    RestoredImage_CPU = cp.transpose(RestoredImage)
    # cpu로 옮긴 후 opencv imwrite함
    RestoredImage_CPU = cp.asnumpy(RestoredImage_CPU)
    cv2.imwrite(imgname + '_' + text + '.jpg', RestoredImage_CPU)

def ImgtoCparray(imgname):
    # 이미지 불러온 후 그레이스케일로 변환, 변환된 이미지를 저장
    image = cv2.imread(imgname + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(imgname + '_grayscale.jpg', gray)

    # transpose 후 gpu로 옮김
    gray_resized = np.transpose(gray)
    print(gray_resized.shape)
    gray_cupy = cp.asarray(gray_resized)

    return gray_cupy, gray_resized

imgname = '8k'
cupy_img, numpy_img = ImgtoCparray(imgname)

# Full SVD
print(benchmark(np.linalg.svd, (numpy_img, True), n_repeat=20))
print(benchmark(cp.linalg.svd, (cupy_img, True), n_repeat=20))
Umat_full, Smat_full, Vmat_T_full = cp.linalg.svd(cupy_img, full_matrices=True)
PrintSVDResultSize(Umat_full, Smat_full, Vmat_T_full, 'full')
RestoreImage(Umat_full, Smat_full, Vmat_T_full, 'full', imgname)

# Economy SVD
print(benchmark(np.linalg.svd, (numpy_img, False), n_repeat=20))
print(benchmark(cp.linalg.svd, (cupy_img, False), n_repeat=20))
Umat_economy, Smat_economy, Vmat_T_economy = cp.linalg.svd(cupy_img, full_matrices=False)
PrintSVDResultSize(Umat_economy, Smat_economy, Vmat_T_economy, 'economy')
RestoreImage(Umat_economy, Smat_economy, Vmat_T_economy, 'economy', imgname)

# Truncated SVD (k=3)
# real 또는 complex array만 지원하기 때문에, 이미지 데이터를 float32로 변환
numpy_img = numpy_img.astype(np.float32)
cupy_img = cupy_img.astype(cp.float32)
# 실행시간 측정
print(benchmark(scipy.sparse.linalg.svds, (numpy_img, 3), n_repeat=20))
print(benchmark(cupyx.scipy.sparse.linalg.svds, (cupy_img, 3), n_repeat=20))
# U, S, Vt 구함
Umat_trunc, Smat_trunc, Vmat_T_trunc = cupyx.scipy.sparse.linalg.svds(cupy_img, k=3)
PrintSVDResultSize(Umat_trunc, Smat_trunc, Vmat_T_trunc, 'trunc')
RestoreImage(Umat_trunc, Smat_trunc, Vmat_T_trunc, 'trunc', imgname)