import numpy as np
import os
import load_images
import init
import forward_pass
import backward_pass
import save_model
from tqdm import tqdm
def train(images, labels, epochs, batch_size, decay, learning_rate, images_test, labels_test):
    conv_kernel1, conv_kernel2, fc_weights, fc_bias = init.init()
    for epoch in range(epochs):
        lr = learning_rate * (decay ** epoch)
        print(f"Epoch [{epoch + 1}/{epochs}]")
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        total_loss = 0.0
        correct_predictions = 0
        for i in tqdm(range(0, len(images), batch_size), desc="Training"):
            batch_indices = indices[i:i + batch_size]
            d_kernel1_sum = np.zeros_like(conv_kernel1)
            d_kernel2_sum = np.zeros_like(conv_kernel2)
            d_fc_weights_sum = np.zeros_like(fc_weights)
            d_fc_bias_sum = np.zeros_like(fc_bias)
            for idx in batch_indices:
                image = images[idx]
                label = labels[idx]
                (convolved1, activated1, pooled1, pool_indices1, convolved2, activated2,
                 pooled2, pool_indices2, flattened, fc_output, predictions) = forward_pass.forward(
                    image, conv_kernel1, conv_kernel2, fc_weights, fc_bias
                )
                loss = -np.log(predictions[label])
                total_loss += loss
                predicted_class = np.argmax(predictions)
                if predicted_class == label:
                    correct_predictions += 1
                d_kernel1, d_kernel2, d_fc_weights, d_fc_bias = backward_pass.backward(
                    image, activated1, pooled1, pool_indices1, activated2,
                    pooled2, pool_indices2, flattened, predictions, label,
                    conv_kernel1, conv_kernel2, fc_weights, fc_bias
                )
                d_kernel1_sum += d_kernel1
                d_kernel2_sum += d_kernel2
                d_fc_weights_sum += d_fc_weights
                d_fc_bias_sum += d_fc_bias
            conv_kernel1 -= lr * d_kernel1_sum / len(batch_indices)
            conv_kernel2 -= lr * d_kernel2_sum / len(batch_indices)
            fc_weights -= lr * d_fc_weights_sum / len(batch_indices)
            fc_bias -= lr * d_fc_bias_sum / len(batch_indices)
        avg_loss = total_loss / len(images)
        accuracy_train = correct_predictions / len(images)
        correct_predictions = 0
        for i in tqdm(range(len(images_test)), desc="Testing"):
            image_test = images_test[i]
            label_test = labels_test[i]
            (convolved1, activated1, pooled1, pool_indices1, convolved2, activated2,
             pooled2, pool_indices2, flattened, fc_output, predictions) = forward_pass.forward(
                image_test, conv_kernel1, conv_kernel2, fc_weights, fc_bias
            )
            predicted_class = np.argmax(predictions)
            if predicted_class == label_test:
                correct_predictions += 1
        accuracy_test = correct_predictions / len(images_test)
        print(f"Average Loss: {avg_loss:.4f}, Accuracy for train: {accuracy_train:.4f}, Accuracy for test: {accuracy_test:.4f}")
        save_model.save_model_to_json(os.path.join('./models', f"model_{epoch}"), conv_kernel1, conv_kernel2, fc_weights, fc_bias)
if __name__ == "__main__":
    classes = 10
    samples = 2000
    image_size = 28
    paths, images, labels = load_images.load_data('./train', classes, image_size, samples)
    paths_test, images_test, labels_test = load_images.load_data('./test', classes, image_size, samples // 10)
    epochs = 20
    batch_size = 10
    decay = 0.95
    learning_rate = 0.003
    train(images, labels, epochs, batch_size, decay, learning_rate, images_test, labels_test)
