# MaskAPI
A free web API for detecting whether a user is wearing a mask

## Endpoint: /api/detect_mask
POST Request Body (JSON):
```
{
  img: string value; base 64 encoded image,
  find_face: boolean value; set this to true if image is not already cropped to face, default: false
}
```
  
Response (JSON):
```
{
  is_wearing_mask: boolean value; whether the person in the image is wearing a mask,
  face_found: boolean value; only present if find_face was true, indicates whether a face was detected,
  error: string value; only present if an error occurred, provides error details
}
```
 
For best results, crop image to face first and then call the API endpoint with find_face = false. There are many existing face detection systems in existence already, so this system does not seek to improve upon those. If find_face is set to true, Haar cascade classifiers from OpenCV will be used for face detection.

The mask detection model is a convolutional neural network which was trained for 1.3 million passes on a dataset of 10,000 examples (before data augmentation). The model achieved 99%+ accuracy and F1 score on the test set (and train and dev sets).

A demo can be found at /frontend/detectmask and documentation can be found at the root.

Note that this API is no longer being hosted. However, the code is provided here. The model is available as a saved model ([/mask_model](mask_model)) in HDF5 format which can be loaded with TensorFlow/Keras.
