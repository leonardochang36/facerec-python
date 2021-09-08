# FACE RECOGNITION FRAMEWORK

A simple Python framework and example for efficient and accurate face recognition. 
Authors: Miguel Gonzalez / Leonardo Chang.

![Alt Text](misc/demo.gif)

## ABOUT

The [FaceRec class](facerec/facerec.py) provides several tools for the complete face recognition process.
1. Face detection
2. Landmark detection
3. Embedding computation
4. Embeddings comparison

## INSTALLATION GUIDE

The provided requirements.txt file can be used to install all the required packages 
(not too many ;) just OpenCV and NumPy). Use the command `pip install –r requirements.txt`

## RUN THE PROJECT

Run our very basic example `simplest_example.py`. Try entering different images in args -s1 and -s2.

```
python simplest_example.py -s1 celebs/Brad_Pitt/brad_1.jpg -s2 celebs/Brad_Pitt/brad_2.jpg
```

## BASIC FUNCTIONALITIES

The script `simplest_example.py` has a basic flow for face verification.

```
# create instance of FaceRec class with resources folder
fr = facerec.FaceRec('./resources')

# load images
img1 = cv.imread(args.subject1)
img2 = cv.imread(args.subject2)

# OpenCV loads image as BGR, but the input for the embedding
# should be in RGB, so for consistency we convert from BGR2RGB
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# compute 128-D embeddings for each image
e1 = fr.face_embedding(img1)
e2 = fr.face_embedding(img2)

# compute euclidean distance
d = fr.face_distance(e1, e2)

# compare distance with threshold
if d < 0.5:
    print('SAME PERSON THERE!!! with distance:', d)
else:
    print('HEYYY ARE YOU TRYING TO FOOL ME???!!! NOT the same person, with distance', d)
```

## MORE DETAILS

Run the `fr_demo.py` for more complete examples.

## WANT TO TRY WITH YOUR FACE?

Add a folder with your name containing some pics of you to the `./celebs/` folder.

![Alt Text](misc/celebs_dataset.jpg)


