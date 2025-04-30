import numpy as np
def backward(image, activated1, pooled1, pool_indices1, activated2, pooled2, pool_indices2, flattened, predictions,
             label, conv_kernel1, conv_kernel2, fc_weights, fc_bias):
    one_hot = np.zeros((10,))
    one_hot[label] = 1
    y_true = one_hot
    d_fc_output = predictions - y_true
    d_fc_weights = np.outer(flattened, d_fc_output)
    d_fc_bias = d_fc_output
    d_flattened = np.dot(d_fc_output, fc_weights.T)
    d_pooled2 = d_flattened.reshape(pooled2.shape)
    d_activated2 = np.zeros_like(activated2)
    for c in range(pooled2.shape[-1]):
        for i in range(pooled2.shape[0]):
            for j in range(pooled2.shape[1]):
                idx = tuple(pool_indices2[i, j, c])
                if 0 <= idx[0] < activated2.shape[0] and 0 <= idx[1] < activated2.shape[1]:
                    d_activated2[idx[0], idx[1], c] += d_pooled2[i, j, c]
    d_convolved2 = d_activated2 * (activated2 > 0).astype(float)  # ReLU 导数
    d_kernel2 = np.zeros_like(conv_kernel2)
    padded_pooled1 = np.pad(pooled1, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    for c_out in range(d_convolved2.shape[-1]):
        for c_in in range(padded_pooled1.shape[-1]):
            for i in range(d_convolved2.shape[0]):
                for j in range(d_convolved2.shape[1]):
                    d_kernel2[:, :, c_in, c_out] += d_convolved2[i, j, c_out] * padded_pooled1[i:i + 3, j:j + 3, c_in]
    d_pooled1 = np.zeros_like(pooled1)
    for c in range(pooled1.shape[-1]):
        for i in range(pooled1.shape[0]):
            for j in range(pooled1.shape[1]):
                idx = tuple(pool_indices1[i, j, c])
                if 0 <= idx[0] < d_convolved2.shape[0] and 0 <= idx[1] < d_convolved2.shape[1]:
                    d_pooled1[i, j, c] = d_convolved2[idx[0], idx[1], c]
    d_activated1 = np.zeros_like(activated1)
    for c in range(pooled1.shape[-1]):
        for i in range(pooled1.shape[0]):
            for j in range(pooled1.shape[1]):
                idx = tuple(pool_indices1[i, j, c])
                if 0 <= idx[0] < activated1.shape[0] and 0 <= idx[1] < activated1.shape[1]:
                    d_activated1[idx[0], idx[1], c] += d_pooled1[i, j, c]
    d_convolved1 = d_activated1 * (activated1 > 0).astype(float)  # ReLU 导数
    d_kernel1 = np.zeros_like(conv_kernel1)
    padded_image = np.pad(image[:, :, np.newaxis], ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    for c_out in range(d_convolved1.shape[-1]):
        for c_in in range(padded_image.shape[-1]):
            for i in range(d_convolved1.shape[0]):
                for j in range(d_convolved1.shape[1]):
                    d_kernel1[:, :, c_in, c_out] += d_convolved1[i, j, c_out] * padded_image[i:i + 3, j:j + 3, c_in]
    return d_kernel1, d_kernel2, d_fc_weights, d_fc_bias