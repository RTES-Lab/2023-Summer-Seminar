#include<stdio.h>
#include<omp.h>

#define IDX2C(i, j, len) (((i) * (len)) + (j))

float kernel[5][5] = {{0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f},
                      {0.015625f, 0.0625f, 0.09375f, 0.0625f, 0.015625f},
                      {0.0234375f, 0.09375f, 0.140625f, 0.09375f, 0.0234375f},
                      {0.015625f, 0.0625f, 0.09375f, 0.0625f, 0.015625f},
                      {0.00390625f, 0.015625f, 0.0234375f, 0.015625f, 0.00390625f}};

void pyrconv2d(float* src, float* dst, int h, int w) {
    size_t i = 0;
    size_t j = 0;

    for(i=0; i<h-4; i++) {
        for(j=0; j<w-4; j++) {
            dst[IDX2C(i, j, w-4)] = src[IDX2C(i, j, w)] * kernel[0][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+1, w)] * kernel[0][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+2, w)] * kernel[0][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+3, w)] * kernel[0][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+4, w)] * kernel[0][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j, w)] * kernel[1][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+1, w)] * kernel[1][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+2, w)] * kernel[1][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+3, w)] * kernel[1][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+4, w)] * kernel[1][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j, w)] * kernel[2][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+1, w)] * kernel[2][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+2, w)] * kernel[2][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+3, w)] * kernel[2][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+4, w)] * kernel[2][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j, w)] * kernel[3][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+1, w)] * kernel[3][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+2, w)] * kernel[3][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+3, w)] * kernel[3][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+4, w)] * kernel[3][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j, w)] * kernel[4][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+1, w)] * kernel[4][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+2, w)] * kernel[4][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+3, w)] * kernel[4][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+4, w)] * kernel[4][4];
        }
    }
    
    return;
}

void pyrconv2d_omp(float* src, float* dst, int h, int w) {
    size_t i = 0;
    size_t j = 0;

    #pragma omp parallel for default(none) collapse(2) \
        private(i, j) shared(src, kernel, dst, w, h)
    for(i=0; i<h-4; i++) {
        for(j=0; j<w-4; j++) {
            dst[IDX2C(i, j, w-4)] = src[IDX2C(i, j, w)] * kernel[0][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+1, w)] * kernel[0][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+2, w)] * kernel[0][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+3, w)] * kernel[0][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i, j+4, w)] * kernel[0][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j, w)] * kernel[1][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+1, w)] * kernel[1][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+2, w)] * kernel[1][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+3, w)] * kernel[1][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+1, j+4, w)] * kernel[1][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j, w)] * kernel[2][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+1, w)] * kernel[2][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+2, w)] * kernel[2][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+3, w)] * kernel[2][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+2, j+4, w)] * kernel[2][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j, w)] * kernel[3][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+1, w)] * kernel[3][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+2, w)] * kernel[3][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+3, w)] * kernel[3][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+3, j+4, w)] * kernel[3][4];

            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j, w)] * kernel[4][0];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+1, w)] * kernel[4][1];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+2, w)] * kernel[4][2];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+3, w)] * kernel[4][3];
            dst[IDX2C(i, j, w-4)] += src[IDX2C(i+4, j+4, w)] * kernel[4][4];
        }
    }
    
    return;
}