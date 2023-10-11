+ ------------------------------------------------- +
|						    |
| 	    OIL SPILL DETECTION DATASET             |
|						    |
+ ------------------------------------------------- +

Data are split into a training and a testing set.

Training set: 1002 samples
Testing set:  110 samples

Every set contains three folders, namely:
	- images: SAR images (.jpg)
	- labels: ground truth segmentation RGB masks (.png)
	- labels_1D: ground truth segmentation labels (.png)


RGB Masks:
Black - Sea Surface
Cyan  - Oil Spill
Red   - Look-alike
Brown - Ship
Green - Land


RGB Values:
Black - (0, 0, 0)
Cyan  - (0, 255, 255)
Red   - (255, 0, 0)
Brown - (153, 76, 0)
Green - (0, 153, 0)


1-D Labels:
0 - Sea Surface
1 - Oil Spill
2 - Look-alike
3 - Ship
4 - Land




