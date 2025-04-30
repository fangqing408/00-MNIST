import numpy as np
def init():
    fan_in_conv1 = 3 * 3 * 1
    std_dev_conv1 = np.sqrt(2.0 / fan_in_conv1)
    conv_kernel1 = np.random.randn(3, 3, 1, 16) * std_dev_conv1
    fan_in_conv2 = 3 * 3 * 16
    std_dev_conv2 = np.sqrt(2.0 / fan_in_conv2)
    conv_kernel2 = np.random.randn(3, 3, 16, 32) * std_dev_conv2
    fc_input_size = (28 // 4) ** 2 * 32
    fan_in_fc = fc_input_size
    std_dev_fc = np.sqrt(2.0 / fan_in_fc)
    fc_weights = np.random.randn(fc_input_size, 10) * std_dev_fc
    fc_bias = np.zeros(10)
    return conv_kernel1, conv_kernel2, fc_weights, fc_bias
'''
# ----------init.py test----------#
conv_kernel1, conv_kernel2, fc_weights, fc_bias = init()
print("conv_kernel1 shape:", conv_kernel1.shape)
print("conv_kernel2 shape:", conv_kernel2.shape)
print("fc_weights shape:", fc_weights.shape)
print("fc_bias shape:", fc_bias.shape)
'''