# Natural-Image-Classification

Run ANN_train.py to classify the images using HoG and SURF features and run ANN_train_daubechies.py to classify the images using Daubechied D4 Wavelet Transform features. Use the following terminal command to run both the codes:

"python -W ignore ANN_train.py"

OR

"python -W ignore ANN_train_daubechies.py"

(Note: The code "ANN_train.py" gives a path to the database I have used for this project. You will need to give the database path as it is on your system).

Both the codes first train the back propagation network using the training data and later test it with the images not used in the training dataset. Since, we are using a total of 480 images as our database (with 160 images belonging to each category), the code will take approximately 10-12 minutes to display the classification results (i.e. the annotated test image).

We could have taken the entire dataset (i.e. a total of 2300 images) for training and testing purposes, however, that would have taken a huge amount of running time for the codes. Hence, for display purposes, we have reduced the size of our dataset to a total of 480 images.

