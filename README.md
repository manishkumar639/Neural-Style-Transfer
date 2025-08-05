# Neural-Style-Transfer
A PyTorch/TensorFlow/Keras implementation of "A Neural Algorithm of Artistic Style" (Leon Gatys et al.) — blending the content of one image with the style of another to produce artistic visuals.

🖼️ Overview
Neural Style Transfer (NST) utilizes a deep convolutional neural network—often VGG-19—to separate image content and style, and combine them into a new image. Starting from a content image p and a style image a, optimization creates an output image x by minimizing a content loss and a style loss, typically with L-BFGS or Adam optimizers. Variants may include total variation loss for smoothing, faster feedforward models (e.g., using perceptual loss), color preservation, masked or multi-style transfer 
arXiv
+15
Wikipedia
+15
GitHub
+15
