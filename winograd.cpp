/*
author: niejiadong
date: 2020/04/22
*/
/*
x: (n, n)
w: (k, k)
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>


void normal_conv2d(float* x, float* w, float* y, int n)
{
    int k = 3;
    int m = n - 2;

    memset(y, 0, sizeof(float) * m * m);

    for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) {
            for (int p=0; p<3; p++) {
                for (int q=0; q<3; q++) {
                    y[i * m + j] += x[(i + p) * n + j + q] * w[p * k + q];
                }
            }
        }
    }
    
    return;
}


void im2col_conv2d(float* x, float* w, float* y, int n)
{
    int k = 3;
    int m = n - 2;
    int m2 = m * m;

    memset(y, 0, sizeof(float) * m2);

    float *im2col = (float*)malloc(m2 * 9 * sizeof(float));
    for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) {
            for (int p=0; p<3; p++) {
                for (int q=0; q<3; q++) {
                    im2col[(i * m + j) * 9 + p * k + q] = x[(i + p) * n + j + q];
                }
            }
        }
    }

    // can use gemm to accelerate, just multiply one by one now.
    for (int i=0; i<m2; i++) {
        for (int j=0; j<9; j++) {
            y[i] += im2col[i * 9 + j] * w[j];
        }
    }

    free(im2col);

    return;
}

void winograd_f2_3_conv2d(float* x, float* w, float* y, int n)
{
    int k = 3;
    int m = n - 2;
    int hf_m = m / 2;

    if (n < 4 && n % 2 != 0) {
        printf("Not support!");
        return;
    }
    
    float *B0 = (float*)malloc(n * hf_m * 4 * sizeof(float));
    float *B = (float*)malloc(hf_m * hf_m * 16 * sizeof(float));
    float *G = (float*)malloc(16 * sizeof(float));

    int i4;
    for (int i=0; i<3; i++) {
        i4 = i << 2;
        G[i4] = w[i * 3];
        G[i4 + 1] = (w[i * 3] + w[i * 3 + 1] + w[i * 3 + 2]) / 2;
        G[i4 + 2] = (w[i * 3] - w[i * 3 + 1] + w[i * 3 + 2]) / 2;
        G[i4 + 3] = w[i * 3 + 2];
    }

    for (int i=0; i<4; i++) {
        G[12 + i] = G[8 + i];
        G[8 + i] = (G[i] - G[4 + i] + G[12 + i]) / 2;
        G[4 + i] = (G[i] + G[4 + i] + G[12 + i]) / 2;
    }

    int offset, j4, j2, offset2;
    for (int i=0; i<n; i++) {
        offset = (i * hf_m) << 2;
        offset2 = i * n;
        for (int j=0; j<hf_m; j++) {
            j4 = (j << 2) + offset;
            j2 = (j << 1) + offset2;
            B0[j4] = x[j2] - x[j2 + 2];
            B0[j4 + 1] = x[j2 + 1] + x[j2 + 2];
            B0[j4 + 2] = x[j2 + 2] - x[j2 + 1];
            B0[j4 + 3] = x[j2 + 1] - x[j2 + 3];
        }
    }

    // 这一步花了比较长时间，内存不连续操作
    int j16, stride, i2;
    float *k0, *k1, *k2, *k3;
    for (int i=0; i<hf_m; i++) {
        offset = (i * hf_m) << 4;
        stride = hf_m << 2;
        offset2 = offset >> 1;
        k0 = B0 + offset2;
        k1 = k0 + stride;
        k2 = k1 + stride;
        k3 = k2 + stride;
        for (int j=0; j<hf_m; j++) {
            j16 = (j << 4) + offset;
            j4 = j << 2;
            for (int k=0; k<4; k++) {
                B[j16 + k] = k0[j4 + k] - k2[j4 + k];
                B[j16 + 4 + k] = k1[j4 + k] + k2[j4 + k];
                B[j16 + 8 + k] = k2[j4 + k] - k1[j4 + k];
                B[j16 + 12 + k] = k1[j4 + k] - k3[j4 + k];
            }
        }
    }
    
    int hf_m_2 = hf_m * hf_m;
    for (int i=0; i<hf_m_2; i++) {
        offset = i << 4;
        for (int j=0; j<16; j++) {
            B[offset + j] *= G[j];
        }
    }

    for (int i=0; i<hf_m_2; i++) {
        offset = i << 4;
        for (int j=0; j<4; j++) {
            j4 = (j << 2) + offset;
            B[j4] += B[j4 + 1] + B[j4 + 2];
            B[j4 + 1] -= B[j4 + 2] + B[j4 + 3];
        }
    }

    for (int i=0; i<hf_m_2; i++) {
        offset = i << 4;
        B[offset] += B[offset + 4] + B[offset + 8];
        B[offset + 4] -= B[offset + 8] + B[offset + 12];
        B[offset + 1] += B[offset + 5] + B[offset + 9];
        B[offset + 5] -= B[offset + 9] + B[offset + 13];
    }

    // reshape
    for (int i=0; i<hf_m; i++) {
        offset = (i << 1) * m;
        offset2 = (i * hf_m) << 4;
        for (int j=0; j<hf_m; j++) {
            y[offset + (j << 1)] = B[offset2 + (j << 4)];
            y[offset + (j << 1) + 1] = B[offset2 + (j << 4) + 1];
            y[offset + m + (j << 1)] = B[offset2 + (j << 4) + 4];
            y[offset + m + (j << 1) + 1] = B[offset2 + (j << 4) + 5];
        }
    }

    free(B0);
    free(B);
    free(G);

    return;
}


void print(float* y, int size)
{
    int max = 20;
    max = (max > size ? size : max);
    for (int i=0; i<max; i++) {
        printf("%.0f, ", y[i]);
    }
    printf("\n...\n");
    for (int i=max; i>0; i--) {
        printf("%.0f, ", y[size-i]);
    }
    printf("\n\n");
}


int main(void) {
    int n = 1000;
    int k = 3;
    int m = n - 2;
    float* x = (float*)malloc(sizeof(float) * n * n);
    float* w = (float*)malloc(sizeof(float) * k * k);
    float* y = (float*)malloc(sizeof(float) * m * m);

    for (int i=0; i<9; i++) {
        w[i] = i;
        // w[i] = 1;
    }
    for (int i=0; i<n*n; i++) {
        x[i] = i;
    }

    struct timeval tv1, tv2;

    gettimeofday(&tv1, NULL);
    normal_conv2d(x, w, y, n);
    gettimeofday(&tv2, NULL);
    printf("spend %ld us\n", (long)(tv2.tv_usec + 1000000 * tv2.tv_sec) - (tv1.tv_usec + 1000000 * tv1.tv_sec));
    print(y, m*m);

    // gettimeofday(&tv1, NULL);
    // im2col_conv2d(x, w, y, n);
    // gettimeofday(&tv2, NULL);
    // printf("spend %ld us\n", (long)(tv2.tv_usec + 1000000 * tv2.tv_sec) - (tv1.tv_usec + 1000000 * tv1.tv_sec));
    // print(y, m*m);

    gettimeofday(&tv1, NULL);
    winograd_f2_3_conv2d(x, w, y, n);
    gettimeofday(&tv2, NULL);
    printf("spend %ld us\n", (long)(tv2.tv_usec + 1000000 * tv2.tv_sec) - (tv1.tv_usec + 1000000 * tv1.tv_sec));
    print(y, m*m);

    free(x);
    free(w);
    free(y);

    return 0;
}