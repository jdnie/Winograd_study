'''
author: niejiadong
date: 2020/04/22
'''
'''
x: (n, n)
w: (k, k)
'''
import numpy as np
import time

def timing(f):
    def inner(x, w):
        start = time.time()
        time.sleep(0.12)
        y = f(x, w)
        end = time.time()
        print('time：%s' % (end - start))
        return y

    return inner


@timing
def normal_conv2d(x, w):
    k, _ = w.shape
    n, _ = x.shape
    m = n - k + 1

    y = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            for p in range(k):
                for q in range(k):
                    y[i][j] += x[i+p][j+q] * w[p][q]
    
    return y


@timing
def im2col_conv2d(x, w):
    k, _ = w.shape
    n, _ = x.shape
    m = n - k + 1

    m2 = m*m
    k2 = k*k

    im2col = np.zeros((m2, k2))
    w = np.reshape(w, (k2,))

    for i in range(m):
        for j in range(m):
            for p in range(k):
                for q in range(k):
                    im2col[i*m+j][p*k+q] = x[i+p][j+q]
    
    y = im2col * w
    y = np.sum(y, axis=1)
    
    y = np.reshape(y, (m, m))
    return y


@timing
def winograd_f2_3_conv2d(x, w):
    k, _ = w.shape
    n, _ = x.shape
    m = n - k + 1
    assert(k == 3)
    assert(n % 2 == 0)
    assert(n >= 4)

    hf_m = int(m // 2)
    
    K0 = x[0:-2:2, :].reshape((hf_m, 1, n))
    K1 = x[1:-2:2, :].reshape((hf_m, 1, n))
    K2 = x[2::2, :].reshape((hf_m, 1, n))
    K3 = x[3::2, :].reshape((hf_m, 1, n))

    x = np.concatenate([K0 - K2, K1 + K2, K2 - K1, K1 - K3], axis=1)

    K0 = x[:, :, 0:-2:2].reshape((hf_m, 4, hf_m, 1))
    K1 = x[:, :, 1:-2:2].reshape((hf_m, 4, hf_m, 1))
    K2 = x[:, :, 2::2].reshape((hf_m, 4, hf_m, 1))
    K3 = x[:, :, 3::2].reshape((hf_m, 4, hf_m, 1))

    x = np.concatenate([K0 - K2, K1 + K2, K2 - K1, K1 - K3], axis=3)
    x = np.swapaxes(x, 1, 2)

    w = np.stack([w[0, :], (w[0, :] + w[1, :] + w[2, :]) / 2, (w[0, :] - w[1, :] + w[2, :]) / 2, w[2, :]], axis=0)
    w = np.stack([w[:, 0], (w[:, 0] + w[:, 1] + w[:, 2]) / 2, (w[:, 0] - w[:, 1] + w[:, 2]) / 2, w[:, 2]], axis=1)
    
    y = x * w  # 对于(4,4) vs (3,3)的卷积，原始要36次乘法，这里只要16次乘法
    y = np.stack((y[:,:,:,0] + y[:,:,:,1] + y[:,:,:,2], y[:,:,:,1] - y[:,:,:,2] - y[:,:,:,3]), axis=-1)
    y = np.stack((y[:,:,0,:] + y[:,:,1,:] + y[:,:,2,:], y[:,:,1,:] - y[:,:,2,:] - y[:,:,3,:]), axis=-1)

    y = y.reshape(hf_m, m, 2)
    y = np.swapaxes(y, 1, 2)
    y = y.reshape(m, m)
    
    return y


@timing
def winograd_f4_3_conv2d(x, w):
    k, _ = w.shape
    n, _ = x.shape
    m = n - k + 1
    assert(k == 3)
    assert((n - 6) % 4 == 0)
    assert(n >= 6)

    h_m = int((n - 6) / 4 + 1)  # int(m // 4)
    
    K0 = x[0:-2:4, :].reshape((h_m, 1, n))
    K1 = x[1:-2:4, :].reshape((h_m, 1, n))
    K2 = x[2::4, :].reshape((h_m, 1, n))
    K3 = x[3::4, :].reshape((h_m, 1, n))
    K4 = x[4::4, :].reshape((h_m, 1, n))
    K5 = x[5::4, :].reshape((h_m, 1, n))

    x = np.concatenate([
        4 * K0 - 5 * K2 + K4,
        -4 * K1 - 4 * K2 + K3 + K4,
        4 * K1 - 4 * K2 - K3 + K4,
        -2 * K1 - K2 + 2 * K3 + K4,
        2 * K1 - K2 - 2 * K3 + K4,
        4 * K1 - 5 * K3 + K5], axis=1)

    K0 = x[:, :, 0:-2:4]
    K1 = x[:, :, 1:-2:4]
    K2 = x[:, :, 2::4]
    K3 = x[:, :, 3::4]
    K4 = x[:, :, 4::4]
    K5 = x[:, :, 5::4]

    x = np.stack([
        4 * K0 - 5 * K2 + K4,
        -4 * K1 - 4 * K2 + K3 + K4,
        4 * K1 - 4 * K2 - K3 + K4,
        -2 * K1 - K2 + 2 * K3 + K4,
        2 * K1 - K2 - 2 * K3 + K4,
        4 * K1 - 5 * K3 + K5], axis=-1)
    x = np.swapaxes(x, 1, 2)

    w = np.stack([
        w[0, :] * 0.25,
        -(w[0, :] + w[1, :] + w[2, :]) / 6,
        (- w[0, :] + w[1, :] - w[2, :]) / 6,
        (w[0, :] + w[1, :] * 2 + w[2, :] * 4) / 24,
        (w[0, :] - w[1, :] * 2 + w[2, :] * 4) / 24,
        w[2, :]], axis=0)
    w = np.stack([
        w[:, 0] * 0.25,
        -(w[:, 0] + w[:, 1] + w[:, 2]) / 6,
        (- w[:, 0] + w[:, 1] - w[:, 2]) / 6,
        (w[:, 0] + w[:, 1] * 2 + w[:, 2] * 4) / 24,
        (w[:, 0] - w[:, 1] * 2 + w[:, 2] * 4) / 24,
        w[:, 2]], axis=1)
    
    y = x * w  # 对于(6,6) vs (3,3)的卷积，原始要144次乘法，这里只要36次乘法
    y = np.stack([
        y[:,:,:,0] + y[:,:,:,1] + y[:,:,:,2] + y[:,:,:,3] + y[:,:,:,4],
        y[:,:,:,1] - y[:,:,:,2] + y[:,:,:,3] * 2 - y[:,:,:,4] * 2,
        y[:,:,:,1] + y[:,:,:,2] + y[:,:,:,3] * 4 + y[:,:,:,4] * 4,
        y[:,:,:,1] - y[:,:,:,2] + y[:,:,:,3] * 8 - y[:,:,:,4] * 8 + y[:,:,:,5]
        ], axis=-1)
    y = np.stack([
        y[:,:,0,:] + y[:,:,1,:] + y[:,:,2,:] + y[:,:,3,:] + y[:,:,4,:],
        y[:,:,1,:] - y[:,:,2,:] + y[:,:,3,:] * 2 - y[:,:,4,:] * 2,
        y[:,:,1,:] + y[:,:,2,:] + y[:,:,3,:] * 4 + y[:,:,4,:] * 4,
        y[:,:,1,:] - y[:,:,2,:] + y[:,:,3,:] * 8 - y[:,:,4,:] * 8 + y[:,:,5,:]
        ], axis=-1)

    y = y.reshape(h_m, m, 4)
    y = np.swapaxes(y, 1, 2)
    y = y.reshape(m, m)
    
    return y


if __name__=='__main__':
    x = np.array(range(161604)).reshape((402, 402))
    # x = np.array(range(100)).reshape((10, 10))
    w = np.array(range(9)).reshape((3, 3))
    # w = np.ones((3,3))

    # y1 = normal_conv2d(x, w)
    # print(y1)

    # y2 = im2col_conv2d(x, w)
    # print(y2)

    y3 = winograd_f2_3_conv2d(x, w)
    print(y3)

    y4 = winograd_f4_3_conv2d(x, w)
    print(y4)




