import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))

    # Convert PIL image to array
    image_np = np.array(image)
    
    # Ensure image_np is RGB (OpenCV uses BGR by default for some ops, but here we just add)
    # Actually cv2.applyColorMap returns BGR. We should convert it to RGB.
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap on original image
    overlayed = heatmap * alpha + image_np * (1 - alpha) # Weighted add
    # Or just simple add as in user request: overlayed = heatmap_color * alpha + image_np
    # Let's stick to user request logic but ensure types match
    
    # User logic:
    # overlayed = heatmap_color * alpha + image_np
    # overlayed = np.uint8(overlayed)
    
    overlayed = heatmap * alpha + image_np
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

    return overlayed
