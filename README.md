# UnityCvPlugin
This is an opencv plugin built for the Unity game engine. It is built on MacOS with XCode and OpenCV.
The accompanying repository for the Unity side is [here](https://github.com/sonnyky/UnityCv).

# What's inside
### Save an image in black and white
Takes a Unity Texture2D object and saves it as a black and white jpg.

### Image transformation
Takes a Unity Texture2D and rotates, scales and translates its ROI (Region of Interest) on top of a new image.

### Detect outer hull
Detects the outer points of an image, useful to determine ROI and other computer vision pipelines.

### Compare image similarity with shape matching
Compares two Texture2Ds and returns the similarity value, the lower the better (0 for exact same images).

### Compare image similarity with image features
Compares two Texture2Ds and returns the number of matched points with the SURF algorithm.

# Usage
Build the project with XCode and copy the generated bundle into Unity. Please check the [accompanying repository](https://github.com/sonnyky/UnityCv) for details.
