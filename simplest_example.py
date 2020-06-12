import sys
import argparse
import logging
import cv2 as cv
import facerec


def main(args):

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

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0])

    # Optional arguments
    parser.add_argument("-s1", "--subject1", default=argparse.SUPPRESS,
                        help="Enter Subject 1 image url")
    parser.add_argument("-s2", "--subject2", default=argparse.SUPPRESS,
                        help="Enter Subject 2 image url")

    args_ = parser.parse_args()

    try:
        sys.exit(main(args_))
    except Exception as exc:
        logging.error(" Oops... something went wrong :(", exc_info=True)
        sys.exit(-1)