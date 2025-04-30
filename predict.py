import json
import numpy as np
from PIL import Image
import forward_pass
import matplotlib.pyplot as plt
import load_images
def predict(image_path, conv_kernel1, conv_kernel2, fc_weights, fc_bias):
    img = np.array(Image.open(image_path))
    (convolved1, activated1, pooled1, pool_indices1, convolved2, activated2,
     pooled2, pool_indices2, flattened, fc_output, predictions) = forward_pass.forward(
        img, conv_kernel1, conv_kernel2, fc_weights, fc_bias
    )
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    return predicted_class, confidence
with open("./models/model_19", 'r') as f:
    model_params = json.load(f)
    conv_kernel1 = np.array(model_params['conv_kernel1'])
    conv_kernel2 = np.array(model_params['conv_kernel2'])
    fc_weights = np.array(model_params['fc_weights'])
    fc_bias = np.array(model_params['fc_bias'])
paths, images, labels = load_images.load_data("./test", 10, 28, 10)
fig, axes = plt.subplots(2, 5, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    predicted_class, confidence = predict(paths[i], conv_kernel1, conv_kernel2, fc_weights, fc_bias)
    ax.set_title(f'Act: {labels[i]}, Pre: {predicted_class}\nConfidence: {confidence * 100:.1f}%')
    ax.axis('off')
plt.tight_layout()
plt.show()