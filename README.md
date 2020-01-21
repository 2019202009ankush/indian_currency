# indian_currency
## Algorithm
   Read the test image
   Create ORB/SIFT/SURF object of the test image
   Fin the key_points,descriptor of test image(des1)
   for each image in train directory
   ---create ORB/SIFT/SURF object of current image
   ---find the key_points,descriptor of current image(des2)
   ---now create a BF matcher object
   ---match des1 and des2 using knn-Match with 2 neighbour point
   ---store it in output_list
   ---for(point1,point2) in output_list
   -----if (point1.distance<0.789*point2.distance)
   -------append the pair <point1,point2> in good_pair_list
   
   -----Now if good_pair_list contains atlast theshold no. of pair
   -------then update old thresold and store image path in a varible
   
   Finally play the suitable audio by compairing the results of feature descriptor of 
   ORB,SIFT,SURF.
