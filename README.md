# Anomaly_Dectection
A school project of applying `Convolutional Autoencoder` to perform anomaly detection on video frames.

## Project Description
1. Built a function `convert_video_to_images` to convert the video to `jpeg` images.
2. Built a function `load_images` to resize and normalize the extracted image frames and return a `Numpy` array of flattened images.
3. Split the flattened image array in training and test set.
4. Built a convolutional autoencoder on the training set.
5. Calculated the loss based on the whole image array, plotted the loss and identified the estimated threshold when the loss starts increasing significantly.
6. Built a function `predict` to pass a video frame at a time and use the found threshold to identify if the frame is an anomaly or not, and return a boolean.
