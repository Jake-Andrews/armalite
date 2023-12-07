# Perceputal Image Hashing

Perceptual image hashing typically operates by taking an arbitrarily sized image, creating a greyscale representation of said image, then resizing the image to a smaller fixed size and then performing an operation on the image. In constrast to cryptographic hashes, the resulting hash is not random and can be compared to other hashes in order to identify images.

# Average hashing (aHash)
This algorithm works by:
1. Convert the image to greyscale 
2. Resize the image to a smaller size
3. Calculate the average pixel value from the grayscale and shrunken image.
4. Now compare each pixel's value to the average pixel value and assign a new binary value based on the comparison (0, 1) (True or False), etc...

# Difference hashing (dHash)
This algorithm works by: 
1. Convert the image to greyscale
2. Resize the image to a smaller size
3. Go through image's pixels horizontally and compute the difference between adjacent pixels (gradient). Note: (reduces row size by 1)

# Hamming Distance
The hamming distance is calculated by counting the number of positions in which symbols differ between two strings of equal length.

a: "abc" b: "abc" Hamming Distance = 0

a: "abc" b: "cba" Hamming Distance = 3


