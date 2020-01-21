#!/usr/bin/python

import os
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS


orbmax_val = 8
orbmax_pt = -1
orbmax_kp = 0

siftmax_val = 8
siftmax_pt = -1
siftmax_kp = 0

surfmax_val = 8
surfmax_pt = -1
surfmax_kp = 0 

window_name = 'matched image'

orb = cv2.ORB_create()
test_img = cv2.imread('/home/ankush/opencv/123.jpg')

(kp1, des1) = orb.detectAndCompute(test_img, None)

path="/home/ankush/opencv/Dataset/"
for i in os.listdir(path):
	# train image
    print(i)
   # print("hi")
    ip=os.path.join(path,i)
    print(ip)
    train_img=cv2.imread(ip)
        
    (kp2, des2)=orb.detectAndCompute(train_img, None)
        
	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > orbmax_val:
        orbmax_val = len(good)
        orbmax_pt = ip
        orbmax_kp = kp2

    print(i, ' ',ip, ' ', len(good))
    
if orbmax_val != 8:
    print(orbmax_pt)
    print('good orb matches ', orbmax_val)
    
else:
    print('No ORB Matches')
    
sift = cv2.xfeatures2d.SIFT_create()

(kp1, des1) = sift.detectAndCompute(test_img, None)

for i in os.listdir(path):
    # train image
    print(i)
   # print("hi")
    ip=os.path.join(path,i)
    print(ip)
    train_img=cv2.imread(ip)
        
    (kp2, des2)=sift.detectAndCompute(train_img, None)
        
    # brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > siftmax_val:
        siftmax_val = len(good)
        siftmax_pt = ip
        siftmax_kp = kp2

    print(i, ' ',ip, ' ', len(good))
    
if siftmax_val != 8:
    print(siftmax_pt)
    print('good sift matches ', siftmax_val)

else:
    print('No SIFT Matches')
    
surf = cv2.xfeatures2d.SURF_create()

(kp1, des1) = surf.detectAndCompute(test_img, None)

for i in os.listdir(path):
    # train image
    print(i)
   # print("hi")
    ip=os.path.join(path,i)
    print(ip)
    train_img=cv2.imread(ip)
        
    (kp2, des2)=surf.detectAndCompute(train_img, None)
        
    # brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)

    good = []
 
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > surfmax_val:
        surfmax_val = len(good)
        surfmax_pt = ip
        surfmax_kp = kp2

    print(i, ' ',ip, ' ', len(good))
    
if surfmax_val != 8:
    print(surfmax_pt)
    print('good surf matches ', surfmax_val)

else:
    print('No Matches')
    
if orbmax_val >=  siftmax_val && orbmax_val >= surfmax_val:
    cv2.imshow(window_name,orbmax_pt)
elif siftmax_val >=  orbmax_val && siftmax_val >= surfmax_val:
    cv2.imshow(window_name,siftmax_pt)
else:
    cv2.imshow(window_name,surfmax_pt)

