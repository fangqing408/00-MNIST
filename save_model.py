import json
def save_model_to_json(file_path, conv_kernel1, conv_kernel2, fc_weights, fc_bias):
    model_params = {
        'conv_kernel1': conv_kernel1.tolist(),
        'conv_kernel2': conv_kernel2.tolist(),
        'fc_weights': fc_weights.tolist(),
        'fc_bias': fc_bias.tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(model_params, f)
    print(f"Model parameters saved to {file_path}")