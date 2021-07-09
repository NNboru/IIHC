1. Download code and data of Holstein cattle from "https://github.com/ameybhole77/IIHC"

2. Upload data to matlab workspace. (matlab online in my case)

3. Run "BatchPreprocessing.m" program in matlab
	- replace ';' with ':' in the code
	- Sample folders will be created with 4 images - 2 segMasks and 2 segRGB images

4. Download this segmented data from matlab workspace into folder "Images_segmented".

5. Run "mypaint.py" program -
	- This will read images from folder "Images_segmented".
	- Then it will do inpainting and then resizing to (256*256)
	- Next it will create a new folder "Images_inpaint_resized" contaning all output images.

6. Run "my_image_to_numpy.py" program -
	- This will read images from folder "Images_inpaint_resized".
	- Then it will split data into training and testing
	- Then it will create a new folder "Images_numpy" storing data in numpy format.
	- It will output shape of the numpy arrays. See screenshot - "ss_shape-of-training-testing.png" 

