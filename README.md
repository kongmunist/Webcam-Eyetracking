# WARNING - DEPRECATED
I no longer remember how this code all works, as such I am recommending for you not to use it because I will not be able to answer any questions.

# Webcam-Eyetracking

Originally a repo for doing facial recognition and matching an emoji to it, I've since hijacked it to hold my models and data collection and raw data for my eye tracking mouse. I use hough circles to perform pupil detection, and use a CNN on a bunch of data I collected. 

MLtracking.py contains the real-time mouse, run it to let the webcam control the mouse!

andyCNN.py loads all the data from the eyes/ and testeyes/ dirs to train the models

eyetrack.py collects the pupil data by moving your cursor around the screen and having you follow its tip around.

testEyes.py evaluates the models I've trained. 
