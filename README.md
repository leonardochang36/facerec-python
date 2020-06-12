![Alt Text](misc/demo.gif)

# FACE RECOGNITION 

A Python framework for efficient and accurate face recognition. 
This project is part of the cooperation between QuantumLabs and Miguel Gonzalez / Leonardo Chang.

## ABOUT

The [FaceRec class](facerec.py) provides several tools for the complete face recognition process.
1. Face detection
2. Landmark detection
3. Embedding computation
4. Embeddings comparison

## USAGE

The script `simplest_example.py` provides a getting started to the code.

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
e1, _ = fr.face_embedding(img1)
e2, _ = fr.face_embedding(img2)

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


