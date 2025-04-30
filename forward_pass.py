import numpy as np
def convolve_2D(input_matrix, kernel):
    input_height, input_width, input_channels = input_matrix.shape
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    padding_size = kernel_height // 2
    matrix_padding = np.pad(input_matrix, ((padding_size, padding_size),
                                           (padding_size, padding_size),
                                           (0, 0)), mode='constant', constant_values=0)
    output_height, output_width = input_height, input_width
    output_matrix = np.zeros((output_height, output_width, out_channels))
    for c in range(out_channels):
        for i in range(output_height):
            for j in range(output_width):
                window = matrix_padding[i:i + kernel_height, j:j + kernel_width, :]
                output_matrix[i, j, c] = np.sum(window * kernel[:, :, :, c])
    return output_matrix
def max_pooling(input_matrix):
    input_height, input_width, input_channels = input_matrix.shape
    pooled_rows, pooled_cols = input_height // 2, input_width // 2
    output_matrix = np.zeros((pooled_rows, pooled_cols, input_channels))
    pool_indices = np.zeros((pooled_rows, pooled_cols, input_channels, 2), dtype=int)
    for c in range(input_channels):
        for i in range(pooled_rows):
            for j in range(pooled_cols):
                window = input_matrix[2 * i:2 * i + 2, 2 * j:2 * j + 2, c]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                output_matrix[i, j, c] = window[max_idx]
                pool_indices[i, j, c] = [2 * i + max_idx[0], 2 * j + max_idx[1]]
    return output_matrix, pool_indices
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)
def forward(image, conv_kernel1, conv_kernel2, fc_weights, fc_bias):
    image = np.expand_dims(image, axis=-1)
    convolved1 = convolve_2D(image, conv_kernel1)
    activated1 = relu(convolved1)
    pooled1, pool_indices1 = max_pooling(activated1)
    convolved2 = convolve_2D(pooled1, conv_kernel2)
    activated2 = relu(convolved2)
    pooled2, pool_indices2 = max_pooling(activated2)
    flattened = pooled2.flatten()
    fc_output = np.dot(flattened, fc_weights) + fc_bias
    predictions = softmax(fc_output)
    return convolved1, activated1, pooled1, pool_indices1, convolved2, activated2, pooled2, pool_indices2, flattened, fc_output, predictions
'''
#----------forward_pass.py test----------#
img = Image.open('./train/0/1.png')
img = img.resize((16, 16))
img = np.array(img)
print(img)
conv_kernel1, conv_kernel2, fc_weights, fc_bias = init.init()
convolved1, activated1, pooled1, pool_indices1, convolved2, activated2, pooled2, pool_indices2, flattened, fc_output, predictions = forward(img, conv_kernel1, conv_kernel2, fc_weights, fc_bias)
for u in range(16):
    for v in range(16):
        print(f"{convolved1[u][v][0]:.1f}", end=' ')
    print("")
print("")
for u in range(16):
    for v in range(16):
        print(f"{activated1[u][v][0]:.1f}", end=' ')
    print("")
print("")
for u in range(8):
    for v in range(8):
        print(f"{pooled1[u][v][0]:.1f}", end=' ')
    print("")
'''