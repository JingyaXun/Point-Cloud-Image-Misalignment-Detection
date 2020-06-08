from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import os

data_path = "../final_project_data/"
EQUATORIAL_R = 6378137.0
POLAR_R = 6356752.3

# define transformation helper functions
#LLA -> ECEF -> ENU -> Camera Coordinate -> Image Coordinate

def sin_phi(coord):
    return np.sin(coord * math.pi/180)

def cos_phi(coord):
    return np.cos(coord * math.pi/180)

def sin_lambda(coord):
    return np.sin(coord * math.pi/180)

def cos_lambda(coord):
    return np.cos(coord * math.pi/180)

def lla2ecef(lat, lon, alt):
    ellipsoid_flattening = (EQUATORIAL_R - POLAR_R) / EQUATORIAL_R
    eccentricity = math.sqrt(ellipsoid_flattening * (2-ellipsoid_flattening))
    dist_N = EQUATORIAL_R/(math.sqrt(1-(eccentricity**2)*(sin_phi(lat)**2)))

    # 𝑋 = (ℎ+𝑁(𝛷)) * cos⁡(𝜆) * cos⁡(𝛷)
    X = (alt + dist_N) * cos_lambda(lon) * cos_phi(lat)
    # 𝑌 = (ℎ+𝑁(𝛷)) * cos⁡(𝜆) * sin⁡(𝛷)
    Y = (alt + dist_N) * cos_phi(lat) * sin_lambda(lon)
    # 𝑍 = (ℎ+(1−𝑒^2)𝑁(𝛷)) * sin⁡(𝜆)
    Z = (alt + (1-eccentricity**2) * dist_N) * sin_phi(lat)

    return X, Y, Z

def ecef2enu(x, y, z, cam_lat, cam_lon, cam_alt):
    x_0, y_0, z_0 = lla2ecef(cam_lat, cam_lon, cam_alt)
    diff_x, diff_y, diff_z = x - x_0, y - y_0, z - z_0

    sin_l = sin_lambda(cam_lon)
    sin_p = sin_phi(cam_lat)
    cos_l = cos_lambda(cam_lon)
    cos_p = cos_phi(cam_lat)

    coord_east = -sin_l * diff_x + cos_l * diff_y
    coord_north = -cos_l * sin_p * diff_x - sin_p * sin_l * diff_y + cos_p * diff_z
    coord_up = cos_p * cos_l * diff_x  + cos_p * sin_l * diff_y + sin_p * diff_z

    return coord_east, coord_north, coord_up


def enu2cam(coord_east, coord_north, coord_up, cam_qs, cam_qx, cam_qy, cam_qz):
    a = 1-2*cam_qy**2-2*cam_qz**2
    b = 2*cam_qx*cam_qy+2*cam_qs*cam_qz
    c = 2*cam_qx*cam_qz-2*cam_qs*cam_qy
    d = 2*cam_qx*cam_qy-2*cam_qs*cam_qz
    e = 1-2*cam_qx**2-2*cam_qz**2
    f = 2*cam_qy*cam_qz+2*cam_qs*cam_qx
    g = 2*cam_qx*cam_qz+2*cam_qs*cam_qy
    h = 2*cam_qy*cam_qz-2*cam_qs*cam_qx
    i = 1-2*cam_qx**2-2*cam_qy**2
    Rq = [[a, b, c], [d, e, f], [g, h, i]]
    cam_x = np.dot(Rq,[coord_north,coord_east,-coord_up])[0]
    cam_y = np.dot(Rq,[coord_north,coord_east,-coord_up])[1]
    cam_z = np.dot(Rq,[coord_north,coord_east,-coord_up])[2]

    return cam_x, cam_y, cam_z

def cam2img(cam_x, cam_y, cam_z):
    x_i, y_i, direction = 0, 0, 0
    a = (resolution - 1)/2
    b = (resolution + 1)/2
    # "front" face
    if cam_z > 0 and cam_z > abs(cam_x) and cam_z > abs(cam_y):
        x_i = int(cam_y/cam_z*a+b)
        y_i = int(cam_x/cam_z*a+b)
        direction = "front"
    # "back" face
    if cam_z < 0 and cam_z < -abs(cam_x) and cam_z < -abs(cam_y):
        x_i = int(-cam_y/cam_z*a+b)
        y_i = int(cam_x/cam_z*a+b)
        direction = "back"
    # "left" face
    if cam_x < 0 and cam_x < -abs(cam_z) and cam_x < -abs(cam_y):
        x_i = int(-cam_y/cam_x*a+b)
        y_i = int(-cam_z/cam_x*a+b)
        direction = "left"
    # "right" face
    if cam_x > 0 and cam_x > abs(cam_y) and cam_x > abs(cam_z):
        x_i = int(cam_y/cam_x*a+b)
        y_i = int(-cam_z/cam_x*a+b)
        direction = "right"
    return x_i, y_i, direction

def calculate_angle(kp1, kp2, matches):
    angles = []
    for m in matches:
        img1_id = m.queryIdx
        img2_id = m.trainIdx
        point1 = tuple(np.round(kp1[img1_id].pt).astype(int))
        point2 = tuple(np.round(kp2[img2_id].pt).astype(int) + np.array([2048, 0]))
        diff = abs( math.atan((float)(point2[1] - point1[1]) / (point1[0] - point2[0])) * (180 / math.pi) )
        angles.append(diff)
    return (sum(angles) / len(angles))

def calculate_distance(kp1, kp2, matches):
    distances = []
    for m in matches:
        img1_id = m.queryIdx
        img2_id = m.trainIdx
        point1 = tuple(np.round(kp1[img1_id].pt).astype(int))
        point2 = tuple(np.round(kp2[img2_id].pt).astype(int))
        distance = math.sqrt((float)(point2[1] - point1[1])**2 + (point2[0] - point1[0])**2)
        distances.append(distance)
    return (sum(distances) / len(distances))


# create overlay of point cloud on camera images
resolution = 2048
cam_lat, cam_lon, cam_alt = 45.90414414, 11.02845385, 227.5819
cam_qs, cam_qx, cam_qy, cam_qz = 0.362114, 0.374050, 0.592222, 0.615007

img_front = np.zeros((resolution,resolution), dtype = float)
img_back = np.zeros((resolution,resolution), dtype = float)
img_left = np.zeros((resolution,resolution), dtype = float)
img_right = np.zeros((resolution,resolution), dtype = float)

img_front_with_intensity = np.zeros((resolution,resolution), dtype = float)
img_back_with_intensity = np.zeros((resolution,resolution), dtype = float)
img_left_with_intensity = np.zeros((resolution,resolution), dtype = float)
img_right_with_intensity = np.zeros((resolution,resolution), dtype = float)

with open(os.path.join(data_path, 'final_project_point_cloud.fuse'), 'rb') as f:
    for entry in f:
        entry = entry.decode('utf8').strip().split(' ')
        p_lat, p_lon, p_alt, p_intensity = float(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])
        x, y, z = lla2ecef(p_lat, p_lon, p_alt)
        e, n, u = ecef2enu(x, y, z, cam_lat, cam_lon, cam_alt)
        x_c, y_c, z_c = enu2cam(e, n, u, cam_qs, cam_qx, cam_qy, cam_qz)
        x_i, y_i, direction = cam2img(x_c, y_c, z_c)
        if direction == "front":
            img_front[x_i][y_i] = 255
            img_front_with_intensity[x_i][y_i] = p_intensity
        if direction == "back":
            img_back[x_i][y_i] = 255
            img_back_with_intensity[x_i][y_i] = p_intensity
        if direction == "left":
            img_left[x_i][y_i] = 255
            img_left_with_intensity[x_i][y_i] = p_intensity
        if direction == "right":
            img_right[x_i][y_i] = 255
        img_right_with_intensity[x_i][y_i] = p_intensity


cv2.imwrite('output/front.png',img_front)
cv2.imwrite('output/back.png',img_back)
cv2.imwrite('output/right.png',img_right)
cv2.imwrite('output/left.png',img_left)


# match misalignment with brute force matcher with ORB Descriptors
# reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# import camera images
cam_img_front = cv2.imread("output/front.png")
cam_img_back = cv2.imread("output/back.png")
cam_img_left = cv2.imread("output/left.png")
cam_img_right = cv2.imread("output/right.png")

# import point cloud images
pc_img_front = cv2.imread(os.path.join(data_path, "image/front.jpg"))
pc_img_back = cv2.imread(os.path.join(data_path, "image/back.jpg"))
pc_img_left = cv2.imread(os.path.join(data_path, "image/left.jpg"))
pc_img_right = cv2.imread(os.path.join(data_path, "image/right.jpg"))

orb = cv2.ORB_create(1000, 1.2)
# Detect keypoints of the original images
kp1, cam_img_front_key = orb.detectAndCompute(cam_img_front, None)
kp2, cam_img_back_key = orb.detectAndCompute(cam_img_back, None)
kp3, cam_img_left_key = orb.detectAndCompute(cam_img_left, None)
kp4, cam_img_right_key = orb.detectAndCompute(cam_img_right, None)

# Detect keypoint of the processed point cloud images
kp5, pc_img_front_key = orb.detectAndCompute(pc_img_front, None)
kp6, pc_img_back_key = orb.detectAndCompute(pc_img_back, None)
kp7, pc_img_left_key = orb.detectAndCompute(pc_img_left, None)
kp8, pc_img_right_key = orb.detectAndCompute(pc_img_right, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
print(type(cam_img_front_key))
print(type(pc_img_front_key))
print(cam_img_front_key)
print(pc_img_front_key)

#front
print("orb result")
matches = bf.match(cam_img_front_key, pc_img_front_key)
matches = sorted(matches, key = lambda x:x.distance)

front_matching = cv2.drawMatches(cam_img_front,kp1,pc_img_front,kp5,matches[:50],None, flags=2)

cv2.imwrite('front_matching.png',front_matching)


front_angle = calculate_angle(kp1, kp5, matches)
front_distance = calculate_distance(kp1, kp5, matches)
print ("The misalignment angle of front image is", front_angle)
print ("The misalignment distance of front image is", front_distance)

#back
matches = bf.match(cam_img_back_key, pc_img_back_key)
matches = sorted(matches, key = lambda x:x.distance)

back_matching = cv2.drawMatches(cam_img_back,kp2,pc_img_back,kp6,matches[:50],None, flags=2)
cv2.imwrite('back_matching.png',back_matching)

back_angle = calculate_angle(kp2, kp6, matches)
back_distance = calculate_distance(kp2, kp6, matches)
print ("The misalignment angle of back image is", back_angle)
print ("The misalignment distance of back image is", back_distance)

#left
matches = bf.match(cam_img_left_key, pc_img_left_key)
matches = sorted(matches, key = lambda x:x.distance)
left_matching = cv2.drawMatches(cam_img_left,kp3,pc_img_left,kp7,matches[:50],None, flags=2)
cv2.imwrite('back_matching.png',back_matching)
left_angle = calculate_angle(kp3, kp7, matches)
left_distance = calculate_distance(kp3, kp7, matches)
print ("The misalignment angle of left image is", left_angle)
print ("The misalignment distance of left image is", left_distance)


#right

matches = bf.match(cam_img_right_key, pc_img_right_key)
matches = sorted(matches, key = lambda x:x.distance)
right_matching = cv2.drawMatches(cam_img_right,kp4,pc_img_right,kp8,matches[:50],None, flags=2)
cv2.imwrite('right_matching.png',right_matching)



right_angle = calculate_angle(kp4, kp8, matches)
right_distance = calculate_distance(kp4, kp8, matches)
print ("The misalignment angle of right image is", right_angle)
print ("The misalignment distance of right image is", right_distance)

#SIFT method
print("SIFT result")
def calculate_angle_sift(kp1, kp2, matches):
    angles = []
    for m, n in matches:
        img1_id = m.queryIdx
        img2_id = m.trainIdx
        (x1, y1) = kp1[img1_id].pt
        (x2, y2) = kp2[img2_id].pt
        x2 = x2 + 2048
        diff= abs( math.atan((float)(y2 - y1) / (x2 - x1)) * (180 / math.pi) )
        angles.append(diff)
    return (sum(angles) / len(angles))


def calculate_distance_sift(kp1, kp2, matches):
    distances = []
    for m, n in matches:
        img1_id = m.queryIdx
        img2_id = m.trainIdx
        (x1, y1) = kp1[img1_id].pt
        (x2, y2) = kp2[img2_id].pt
        distance = math.sqrt((float)(y2 - y1)**2 + (x2 - x1)**2)
        distances.append(distance)
    return (sum(distances) / len(distances))



sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

# Front matching
kp1_sift, cam_img_front_key_sift = sift.detectAndCompute(cam_img_front, None)
kp2_sift, cam_img_back_key_sift = sift.detectAndCompute(cam_img_back, None)
kp3_sift, cam_img_left_key_sift = sift.detectAndCompute(cam_img_left, None)
kp4_sift, cam_img_right_key_sift = sift.detectAndCompute(cam_img_right, None)


kp5_sift, pc_img_front_key_sift = sift.detectAndCompute(pc_img_front, None)
kp6_sift, pc_img_back_key_sift = sift.detectAndCompute(pc_img_back, None)
kp7_sift, pc_img_left_key_sift = sift.detectAndCompute(pc_img_left, None)
kp8_sift, pc_img_right_key_sift = sift.detectAndCompute(pc_img_right, None)


# front
matches_sift = bf.knnMatch(cam_img_front_key_sift, pc_img_front_key_sift, k=2)

distances = []
for m,n in matches_sift:
    if m.distance < 0.85 * n.distance:
        distances.append([m])

front_matching_sift = cv2.drawMatchesKnn(cam_img_front,kp1_sift,pc_img_front,kp5_sift,distances, None, flags=2)
cv2.imwrite('front_matching_sift.png',front_matching_sift)
front_angle_sift = calculate_angle_sift(kp1_sift, kp5_sift, matches_sift)
front_distance_sift = calculate_distance_sift(kp1_sift, kp5_sift, matches_sift)
print ("The misalignment angle of right image is", front_angle_sift)
print ("The misalignment distance of right image is", front_distance_sift)

# back

matches_sift = bf.knnMatch(cam_img_back_key_sift, pc_img_back_key_sift, k=2)

distances = []
for m,n in matches_sift:
    if m.distance < 0.85 * n.distance:
        distances.append([m])

back_matching_sift = cv2.drawMatchesKnn(cam_img_back,kp2_sift,pc_img_back,kp6_sift,distances, None, flags=2)
cv2.imwrite('back_matching_sift.png',back_matching_sift)
back_angle_sift = calculate_angle_sift(kp2_sift, kp6_sift, matches_sift)
back_distance_sift = calculate_distance_sift(kp2_sift, kp6_sift, matches_sift)
print ("The misalignment angle of back image is", back_angle_sift)
print ("The misalignment distance of back image is", back_distance_sift)

#left
matches_sift = bf.knnMatch(cam_img_left_key_sift, pc_img_left_key_sift, k=2)

distances = []
for m,n in matches_sift:
    if m.distance < 0.85 * n.distance:
        distances.append([m])

left_matching_sift = cv2.drawMatchesKnn(cam_img_left,kp3_sift,pc_img_left,kp7_sift,distances, None, flags=2)
cv2.imwrite('left_matching_sift.png',left_matching_sift)
left_angle_sift = calculate_angle_sift(kp3_sift, kp7_sift, matches_sift)
left_distance_sift = calculate_distance_sift(kp3_sift, kp7_sift, matches_sift)
print ("The misalignment angle of left image is", left_angle_sift)
print ("The misalignment distance of left image is", left_distance_sift)


#right
matches_sift = bf.knnMatch(cam_img_right_key_sift, pc_img_right_key_sift, k=2)

distances = []
for m,n in matches_sift:
    if m.distance < 0.85 * n.distance:
        distances.append([m])

right_matching_sift = cv2.drawMatchesKnn(cam_img_right,kp4_sift,pc_img_right,kp8_sift,distances, None, flags=2)
cv2.imwrite('right_matching_sift.png',right_matching_sift)
right_angle_sift = calculate_angle_sift(kp4_sift, kp8_sift, matches_sift)
right_distance_sift = calculate_distance_sift(kp4_sift, kp8_sift, matches_sift)
print ("The misalignment angle of right image is", right_angle_sift)
print ("The misalignment distance of right image is", right_distance_sift)
