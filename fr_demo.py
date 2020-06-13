import sys
import argparse
import numpy as np 
import logging
import time
import os
import cv2 as cv
import re
from facerec import facerec


def main(args):

    # create instance of FaceRec class with resources folder
    fr = facerec.FaceRec('./facerec/resources')

    # ===============================================================================
    # ========--- DO FACE VERIFICATION (1 vs. 1) ---=================================
    # ===============================================================================
    if 'subject1' in args and 'subject2' in args:
        print('\n=============================================================================')
        print('Hello Human! It looks like you are trying to compare two faces:')
        print(args.subject1, 'vs.', args.subject2)
        print('=============================================================================\n')

        img1 = cv.imread(args.subject1)
        if img1 is None:
            print('ERROR: loading image', args.subject1)
            return
        img2 = cv.imread(args.subject2)
        if img2 is None:
            print('ERROR: loading image', args.subject2)
            return
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

        print('Computing embedding for subject 1 .......................... ', end='', flush=True)
        t0 = time.time()
        e1 = fr.face_embedding(img1)
        if len(e1) == 0:
            print('ERROR: no faces found')
            return
        print('DONE in', int((time.time() - t0) * 1000), 'ms')

        print('Computing embedding for subject 2 .......................... ', end='', flush=True)
        t0 = time.time()
        e2 = fr.face_embedding(img2)
        if len(e2) == 0:
            print('ERROR: no faces found')
            return
        print('DONE in', int((time.time() - t0) * 1000), 'ms')

        print('Computing faces distance ................................... ', end='', flush=True)
        t0 = time.time()
        d = fr.face_distance(e1, e2)
        print('DONE in', int((time.time() - t0) * 1000), 'ms')
        if d < 0.5:
            print('SAME PERSON THERE!!! with distance:', d)
        else:
            print('HEYYY ARE YOU TRYING TO FOOL ME???!!! NOT the same person, with distance', d)
    
    # ===============================================================================
    # ========--- DO FACE IDENTIFICATION (1 vs. MANY) ---============================
    # ===============================================================================
    else:
        print('\n=============================================================================')
        print('=== Hello Human! It looks like you are trying to identify a face')
        
        # Do face identification from image file
        if 'subject1' in args:
            print('===', args.subject1, 'vs. every subject in', args.gallery)
            print('=============================================================================\n')
            img1 = cv.imread(args.subject1)
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)                    

        # Do identification from webcam
        elif args.use_webcam is True:
            print('=== using a camera frame vs. every subject in', args.gallery)
            print('=============================================================================\n')
            video_capture = cv.VideoCapture(0)
            while True:
                _, frame = video_capture.read()
                cv.imshow('Video', frame)

                # Hit 'space' on the keyboard to capture frame!
                if cv.waitKey(5) == 32:
                    img1 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    break
        else:
            print('Input ERROR')
            return

        print('Loading gallery dataset .................................... ', end='', flush=True)
        t0 = time.time()
        gallery_ids, gallery_names, gallery_embbedings, gallery_imgs = load_gallery(args.gallery, fr)
        print('DONE in', int((time.time() - t0) * 1000), 'ms')
        print('\t', len(gallery_names), 'faces found.')

        print('Computing embedding for input face ......................... ', end='', flush=True)
        t0 = time.time()
        e1 = fr.face_embedding(img1)
        if len(e1) == 0:
            print('ERROR: no faces found')
            return
        print('DONE in', int((time.time() - t0) * 1000), 'ms')

        print('Computing distances with gallery ........................... ', end='', flush=True)
        t0 = time.time()
        distances = fr.face_distance(e1, gallery_embbedings)
        print('DONE in', int((time.time() - t0) * 1000), 'ms')

        print('Finding top matches below threshold ........................ ', end='', flush=True)
        t0 = time.time()
        threshold = 0.5
        idx = np.argsort(distances, axis=0)
        matches = [(gallery_ids[x], gallery_names[x], distances[x], gallery_imgs[x])
                   for x in idx if distances[x] < threshold]
        print('DONE in', int((time.time() - t0) * 1000), 'ms')
        if matches:
            print('\t Hey! we found', len(matches), 'matches for the input face!!!')
            [print('\t', (m[0], m[1], m[2])) for m in matches]
            imgv = cv.resize(img1, (300, int(img1.shape[0] * 300 / img1.shape[1])))
            vfeed = visual_feedback_for_matches(cv.cvtColor(imgv, cv.COLOR_RGB2BGR),
                                                [m[3] for m in matches])
            cv.rectangle(vfeed, (0, 0), (imgv.shape[1], imgv.shape[0]), (0, 255, 0), 8)
            cv.imshow('MATCHES', vfeed)
            cv.waitKey()

        else:
            print('\t Sorry, no matches found for the input face!!!')

    return


def load_gallery(folder, fr):
    gallery_ids = []
    gallery_names = []
    gallery_embbedings = []
    gallery_imgs = []

    for subfolder in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, subfolder)):
            continue
        for file in image_files_in_folder(os.path.join(folder, subfolder)):
            basename = os.path.basename(file)
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            descr, aligned = fr.face_embedding(img, return_aligned=True)

            if len(descr) == 0:
                print("WARNING: No faces found in {}. Ignoring file.".format(file))
            else:
                gallery_ids.append(subfolder)
                gallery_names.append(basename)
                gallery_embbedings.append(descr)
                gallery_imgs.append(cv.cvtColor(aligned, cv.COLOR_RGB2BGR))

    return gallery_ids, gallery_names, np.concatenate(gallery_embbedings, axis=0), gallery_imgs


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) 
            if re.match(r'.*\.(jpg|jpeg|png|JPG|PNG)', f, flags=re.I)]


def visual_feedback_for_matches(query, matches):
    h = max(query.shape[0], max(matches, key=lambda x: x.shape[0]).shape[0])
    w = query.shape[1] + sum([x.shape[1] for x in matches])

    img = np.zeros((h, w, 3), np.uint8)
    img[:query.shape[0], :query.shape[1]] = query
    col_offset = query.shape[1]
    for m in matches:
        img[0:m.shape[0], col_offset:col_offset+m.shape[1]] = m
        col_offset += m.shape[1]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0])

    # Optional arguments
    parser.add_argument("-s1", "--subject1", default=argparse.SUPPRESS,
                        help="Enter Subject 1 image url")
    parser.add_argument("-s2", "--subject2", default=argparse.SUPPRESS,
                        help="Enter Subject 2 image url")
    parser.add_argument("-webcam", "--use_webcam", default=True,
                        help="Use webcam as input")
    parser.add_argument("-g", "--gallery", default='./celebs/',
                        help="Path to the gallery folder")

    args_ = parser.parse_args()

    try:
        sys.exit(main(args_))
    except Exception as exc:
        logging.error(" Oops... something went wrong :(", exc_info=True)
        sys.exit(-1)