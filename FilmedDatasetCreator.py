
# coding: utf-8

# In[1]:

# %pylab
from matplotlib import cm
import sys
import numpy as np
import time
import os
import glob
import cv2

import OpenEXR
import Imath
import array
import pickle

from PySide import QtGui, QtCore
from PIL import Image


import triangulate

app = QtGui.QApplication(sys.argv)

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
# DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_SEQUENCE_LOCATION = "sequence_location"

DICT_FILMED_DATASET_BASE_LOC = 'filmed_dataset_base_location'

DICT_FILMED_OBJECT_NAME = 'filmed_object_name'
DICT_TRAJECTORY_POINTS = 'trajectory_points'
DICT_NEEDS_UNDISTORT = 'do_undistort_trajectory_points'
DICT_OBJECT_BILLBOARD_ORIENTATION = 'object_color_billboard_orientation_angle'
DICT_OBJECT_BILLBOARD_SCALE = 'object_color_bilboard_scale'
DICT_TRACK_LOCATION='track_points_location'

DICT_FILMED_SCENE_BASE_LOC = 'filmed_scene_base_location'
DICT_CAMERA_EXTRINSICS = 'camera_extrinsics'
DICT_CAMERA_INTRINSICS = 'camera_intrinsics'
DICT_DISTORTION_PARAMETER = 'distortion_parameter'
DICT_DISTORTION_RATIO = 'distortion_ratio'
DICT_DOWNSAMPLED_FRAMES_RATE = 'downsampled_frames_rate'
DICT_COMMENTS = "comments_and_info"
DICT_GROUND_MESH_POINTS = 'camera_ground_plane_mesh_points'
DICT_GROUND_MESH_SEGS_EXTRUDE = 'ground_plane_mesh_segments_to_extrude'
DICT_OBJECT_LENGTH = 'object_bounding_volume_length'
DICT_OBJECT_WIDTH = 'object_bounding_volume_width'
DICT_OBJECT_HEIGHT = 'object_bounding_volume_height'


# In[2]:

def line2lineIntersection(line1, line2) :
    """x1, y1, x2, y2 = line1
       x3, y3, x4, y4 = line2"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denominator != 0 :
        Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
        Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
        return np.array([Px, Py])
    else :
        raise RuntimeError("lines are parallel")

def isABetweenBandC(a, b, c):
    distAB = np.linalg.norm(a-b)
    distAC = np.linalg.norm(a-c)
    distBC = np.linalg.norm(b-c)
    return np.abs(distAB+distAC-distBC) < 1e-10

def printMatrix(mat, useQt = True) :
    if useQt :
        sys.stdout.write("QtGui.QMatrix4x4(")
    else :
        sys.stdout.write("np.array([")
        
    for i in xrange(mat.shape[0]) :
        if useQt :
            sys.stdout.write("{0}".format(mat[i, 0]))
        else :
            sys.stdout.write("[{0}".format(mat[i, 0]))
        for j in xrange(1, mat.shape[1]) :
            sys.stdout.write(", {0}".format(mat[i, j]))
        
        if i < mat.shape[0]-1 :
            if useQt :
                sys.stdout.write(",\n\t\t ")
            else :
                sys.stdout.write("],\n\t  ")
        else :
            if not useQt :
                sys.stdout.write("]")
    if useQt :
        sys.stdout.write(")")
    else :
        sys.stdout.write("])")
    print


# In[3]:

def getLineRectIntersectionPoints(line, lines) :
    ## this could probably be done using some fast line rect intersection but I cba
    intersectionPoints = []
    for segment in lines :
        try :
            point = line2lineIntersection(line, segment)
            if isABetweenBandC(point, segment[0:2], segment[2:]) :
                intersectionPoints.append(point)
        except Exception:
            pass
    return np.array(intersectionPoints)


# In[4]:

def getWorldSpacePosAndNorm(transform, normDirEnd=np.array([[0.0], [0.0], [1.0], [1.0]]), posOnly=False) :
    pos = np.dot(transform, np.array([[0.0], [0.0], [0.0], [1.0]])).T
    pos = pos[0, :3]/pos[0, 3]
    if posOnly :
        return pos
    norm = np.dot(transform, normDirEnd).T
    norm = norm[0, :3]/norm[0, 3]
    norm -= pos
    norm /= np.linalg.norm(norm)
    
    return pos, norm

def quaternionTo4x4Rotation(quaternion, inverted=False):
    x, y, z, w = quaternion
    ## quaternion rotation
    M = np.array([[1.0 - 2.0*(y**2) - 2.0*(z**2), 2*x*y + 2*w*z, 2*x*z - 2*w*y, 0.0],
                  [2*x*y - 2*w*z, 1.0 - 2.0*(x**2) - 2.0*(z**2), 2*y*z + 2*w*x, 0.0],
                  [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1.0 - 2.0*(x**2) - 2.0*(y**2), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    ## invert it
    if inverted :
        M[:-1, :-1] = M[:-1, :-1].T
        
    return M

def angleAxisToQuaternion(angle, axis) :
    return np.array([axis[0]*np.sin(angle/2.0), axis[1]*np.sin(angle/2.0), axis[2]*np.sin(angle/2.0), np.cos(angle/2.0)])

def rotateAboutPoint(matrix, quaternion, centerPoint) :
    M = quaternionTo4x4Rotation(quaternion)
    T = np.array([[1.0, 0.0, 0.0, centerPoint[0]],
                  [0.0, 1.0, 0.0, centerPoint[1]],
                  [0.0, 0.0, 1.0, centerPoint[2]],
                  [0.0, 0.0, 0.0, 1.0]])
    
    return np.dot(T, np.dot(M, np.dot(np.linalg.inv(T), matrix)))

def cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, imageShape) :
    """ return viewMat, projectionMat """
    
    viewMat = np.copy(cameraExtrinsics)
    ## flip z and y axis because of opencv vs opengl coord systems
    viewMat[2, :] *= -1
    viewMat[1, :] *= -1

    K = np.copy(cameraIntrinsics)
    ## changing signs for the same reason as above for the viewMat
    K[:, 2] *= -1
    K[:, 1] *= -1
    near = 0.1
    far = 100.0
    projectionMat = np.zeros([4, 4])
    projectionMat[:2, :-1] = K[:2, :]
    projectionMat[-1, :-1] = K[-1, :]
    projectionMat[2, 2] = near + far
    projectionMat[2, 3] = near*far

    left = 0.0
    right = float(imageShape[1])
    bottom = float(imageShape[0])
    top = 0.0

    projectionMat = np.dot(np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                                     [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                                     [0, 0, -2/(far-near), -(far+near)/(far-near)],
                                     [0, 0, 0, 1]]), np.copy(projectionMat))
    return viewMat, projectionMat

def glCameraToOpenCV(viewMat, projectionMat, imageShape) :
    """ return cameraExtrinsics, cameraIntrinsics """
    
    cameraExtrinsics = np.copy(viewMat)
    ## flip z and y axis because of opencv vs opengl coord systems
    cameraExtrinsics[2, :] *= -1
    cameraExtrinsics[1, :] *= -1

    near = 0.1
    far = 100.0
    left = 0.0
    right = float(imageShape[1])
    bottom = float(imageShape[0])
    top = 0.0

    cameraIntrinsics = np.dot(np.linalg.inv(np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                                                      [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                                                      [0, 0, -2/(far-near), -(far+near)/(far-near)],
                                                      [0, 0, 0, 1]])), np.copy(projectionMat))
    ## changing signs for the same reason as above for the cameraExtrinsics
    cameraIntrinsics[:, 2] *= -1
    cameraIntrinsics[:, 1] *= -1
    
    cameraIntrinsics = np.vstack([cameraIntrinsics[:2, :-1], cameraIntrinsics[-1, :-1][np.newaxis, :]])
    return cameraExtrinsics, cameraIntrinsics

def worldToScreenSpace(viewMat, projectionMat, worldSpacePoint, viewportWidth, viewportHeight) :
    """worldSpacePoint can be either a vector of length 3 or it can be a matrix Nx3"""
    if len(worldSpacePoint.shape) == 1 :
        worldSpacePoints = np.reshape(worldSpacePoint, [1, 3])
    else :
        worldSpacePoints = worldSpacePoint
    
    screenSpacePoints = np.dot(np.dot(projectionMat, viewMat), np.hstack([worldSpacePoints, np.ones([len(worldSpacePoints), 1])]).T)
    screenSpacePoints = screenSpacePoints[:-1, :]/screenSpacePoints[-1, :]
    screenSpacePoints = screenSpacePoints.T
    
    ## from clip space to screen space
    screenSpacePoints = np.hstack([((screenSpacePoints[:, 0]+1.0)*viewportWidth/2.0)[:, np.newaxis], 
                                   ((1.0-screenSpacePoints[:, 1])*viewportHeight/2.0)[:, np.newaxis]])
    if len(screenSpacePoints) == 1 :
        return screenSpacePoints.flatten()
    else :
        return screenSpacePoints
    
def screenToWorldPlane(screenPoints, cameraIntrinsics, cameraExtrinsics, planeZ=0.0) :
    """screenPoints is a Nx2 matrix"""
    worldPoints = np.dot(np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])), 
                         np.vstack([screenPoints.T, np.ones([1, len(screenPoints)])]))
    worldPoints /= worldPoints[-1, :]
    worldPoints[-1, :] = planeZ
    
    return worldPoints.T
    
def moveVirtualCamera(viewMat, doGoCW, doGoHigh) :
    virtualViewMat = np.copy(viewMat)
    camPos, camNorm = getWorldSpacePosAndNorm(np.linalg.inv(virtualViewMat))
    translate = camNorm*np.linalg.norm(camPos)*0.7
    virtualViewMat = np.dot(virtualViewMat, np.linalg.inv(np.array(np.array([[1, 0, 0, translate[0]],
                                                                             [0, 1, 0, translate[1]],
                                                                             [0, 0, 1, translate[2]],
                                                                             [0, 0, 0, 1]], float))))
    rotationModifier = 1.0
    if not doGoCW :
        rotationModifier = -1.0
    virtualViewMat = np.linalg.inv(rotateAboutPoint(np.linalg.inv(virtualViewMat), angleAxisToQuaternion(rotationModifier*np.pi/8, np.array([0.0, 0.0, 1.0])), np.zeros(3)))
    
    rotationModifier = 1.0
    if not doGoHigh :
        rotationModifier = -1.0
    angle = np.arccos(np.dot(camNorm, np.array([0.0, 0.0, 1.0])))
    if doGoHigh :
        angle *= 0.5
    else :
        angle = ((np.pi/2.0) - angle)*0.5
    
    _, camRightVec = getWorldSpacePosAndNorm(np.linalg.inv(virtualViewMat), np.array([[1, 0, 0, 1]], float).T)
    virtualViewMat = np.linalg.inv(rotateAboutPoint(np.linalg.inv(virtualViewMat), angleAxisToQuaternion(rotationModifier*angle, camRightVec), np.zeros(3)))
    
    return virtualViewMat

def triangulate2DPolygon(poly2D, doReturnIndices=True) :
    pts = [(point[0], point[1]) for point in poly2D]
    availableIndices = np.ones(len(pts), dtype=bool)
    tris = []
    plist = pts[::-1] if triangulate.IsClockwise(pts) else pts[:]
    while len(plist) >= 3:
        a = triangulate.GetEar(plist, np.arange(len(pts), dtype=int), availableIndices, doReturnIndices)
        if a == []:
            break
        if doReturnIndices :
            if triangulate.IsClockwise(pts) :
                tris.append([len(pts)-1-a[0], len(pts)-1-a[1], len(pts)-1-a[2]])
            else :
                tris.append(list(a))
        else :
            tris.append(a)
            
    return tris

def extrudeSegment(points, height, viewLoc, doReturnIndexedVertices=True) :
    inputVertices = np.vstack([points, points[::-1, :]+np.array([0.0, 0.0, height])])
    outputIndices = [0, 1, 3, 1, 2, 3]
    
    ## check that the triangle is front facing --> https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/single-vs-double-sided-triangle-backface-culling
    vertices = inputVertices[outputIndices[:3], :]
    N = np.cross(vertices[1, :]-vertices[0, :], vertices[2, :]-vertices[0, :])
    N /= np.linalg.norm(N)
    viewDir = vertices[0, :]-viewLoc
    viewDir /= np.linalg.norm(viewDir)
    
    ## it is back-facing so need to reverse dir
    if np.dot(viewDir, N) > 0 :
        outputIndices[0], outputIndices[1] = outputIndices[1], outputIndices[0]
        outputIndices[3], outputIndices[4] = outputIndices[4], outputIndices[3]
        
    if doReturnIndexedVertices :
        return inputVertices[outputIndices, :], outputIndices
    else :
        return inputVertices, outputIndices

def isPoint2DInTriangle2D(point2D, triangle2D) :
    ## formulas from here http://mathworld.wolfram.com/TriangleInterior.html
    v = point2D
    v0 = triangle2D[0, :]
    v1 = triangle2D[1, :]-v0
    v2 = triangle2D[2, :]-v0
    
    vv2 = np.hstack([v.reshape([2, 1]), v2.reshape([2, 1])])
    v0v2 = np.hstack([v0.reshape([2, 1]), v2.reshape([2, 1])])
    v1v2 = np.hstack([v1.reshape([2, 1]), v2.reshape([2, 1])])
    vv1 = np.hstack([v.reshape([2, 1]), v1.reshape([2, 1])])
    v0v1 = np.hstack([v0.reshape([2, 1]), v1.reshape([2, 1])])
    
    a = (np.linalg.det(vv2)-np.linalg.det(v0v2))/np.linalg.det(v1v2)
    b = -(np.linalg.det(vv1)-np.linalg.det(v0v1))/np.linalg.det(v1v2)
    
    return a > 0 and b > 0 and (a+b) < 1

def getGridPointsInPolygon2D(polygon2D, gridSpacing) :
    triangles = np.array(triangulate2DPolygon(polygon2D, False))
    minBounds = ((np.min(polygon2D, axis=0)>0)*2-1)*np.ceil(np.abs(np.min(polygon2D, axis=0)))
    maxBounds = ((np.max(polygon2D, axis=0)>0)*2-1)*np.ceil(np.abs(np.max(polygon2D, axis=0)))

    gridPoints = np.mgrid[minBounds[0]:maxBounds[0]+gridSpacing:gridSpacing, minBounds[1]:maxBounds[1]+gridSpacing:gridSpacing]
    gridPoints = gridPoints.reshape([2, gridPoints.shape[1]*gridPoints.shape[2]]).T
    
    validPoints = []
    for pointIdx, point in enumerate(gridPoints) :
        for triangle in triangles :
            if isPoint2DInTriangle2D(point, triangle) :
                validPoints.append(pointIdx)
                break
    return gridPoints[validPoints, :]


# In[5]:

def distortPoints(undistortedPoints, distortionCoeff, undistortedIntrinsics, distortedIntrinsics) :
    """ distorts points in an undistorted image space to the original image space as if they are 
    seen again through the distorting lens
    
    - as seen here: http://stackoverflow.com/a/35016615"""
    
    ### could also do this as seen in getDistortedPointsFromUndistorted()
    
    ## not sure what this does but it doesn't work without it
    tmp = cv2.undistortPoints(undistortedPoints.reshape([1, len(undistortedPoints), 2]),
                              undistortedIntrinsics, np.zeros(5))
    distortedPoints = cv2.projectPoints(np.concatenate([tmp, np.ones([1, tmp.shape[1], 1])], axis=-1), (0, 0, 0),
                                        (0, 0, 0), distortedIntrinsics, distortionCoeff)[0][:, 0, :]
    return distortedPoints

def getDistortionCoeffFromParamAndRatio(distortionParameter, distortionRatio) :
    return np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])

def undistortImage(distortionParameter, distortionRatio, image, cameraIntrinsics, doUncrop=True, interpolation=cv2.INTER_LANCZOS4, doReturnMaps=False) :
    distortionCoeff = getDistortionCoeffFromParamAndRatio(distortionParameter, distortionRatio)
    
    frameSize = np.array([image.shape[1], image.shape[0]])

    ## undistort image
    if doUncrop :
        ## here I was just making the image I project the undistorted pixels to bigger
#         sizeDelta = 0.3
#         newFrameSize = (frameSize*(1+sizeDelta)).astype(int)
#         newIntrinsics = np.copy(cameraIntrinsics)
#         newIntrinsics[0, 2] += image.shape[1]*sizeDelta/2.0
#         newIntrinsics[1, 2] += image.shape[0]*sizeDelta/2.0
        ## here I instead use opencv to figure out the best new camera matrix that includes all possible pixels
        newIntrinsics = cv2.getOptimalNewCameraMatrix(cameraIntrinsics, distortionCoeff, tuple(frameSize), 1)[0]
        ## the above tends to change the camera center in different way and giving x and y focals different values
        ## so I scale the center to match the old intrinsics and the corresponding focals which should bring them to be the same
        newIntrinsics[0, [0, 2]] *= cameraIntrinsics[0, 2]/newIntrinsics[0, 2]
        newIntrinsics[1, [1, 2]] *= cameraIntrinsics[1, 2]/newIntrinsics[1, 2]
        ## the above, changes the focal length to see the full scene, but I want to keep focal length and have a bigger image instead, so I change the intrinsics to get the original focal length but bigger image
        scale = np.average([cameraIntrinsics[0, 0]/newIntrinsics[0, 0], cameraIntrinsics[1, 1]/newIntrinsics[1, 1]])
        newFrameSize = np.ceil(np.copy(frameSize)*scale).astype(int)
        newIntrinsics[0, 0] = cameraIntrinsics[0, 0]
        newIntrinsics[1, 1] = cameraIntrinsics[1, 1]
        ## I want the camera center to be a full number and the new frame size to be divisible by two
        newIntrinsics[:-1, -1] = np.ceil(newFrameSize/2.0)
        newFrameSize = np.array(newIntrinsics[:-1, -1]*2, dtype=int)
    else :
        newIntrinsics = np.copy(cameraIntrinsics)
        newFrameSize = np.copy(frameSize)
    
    map1, map2 = cv2.initUndistortRectifyMap(cameraIntrinsics, distortionCoeff, None, newIntrinsics, tuple(newFrameSize), cv2.CV_32FC1)
    undistortedImage = cv2.remap(image, map1, map2, interpolation)
    if doReturnMaps :
        return undistortedImage, newIntrinsics, distortionCoeff, map1, map2
    else :
        return undistortedImage, newIntrinsics, distortionCoeff
    
    
def getDistortedPointsFromUndistorted(undistortedPoints, k1, k2, distortedIntrinsics, undistortedIntrinsics, doOpenCVModel=False, center=np.array([[0.0, 0.0]])) :
    if doOpenCVModel :
        distortedPoints = (undistortedPoints - undistortedIntrinsics[[0, 1], [2, 2]][np.newaxis, :])/undistortedIntrinsics[[0, 1], [0, 1]][np.newaxis, :]
        r = np.sqrt(np.sum((distortedPoints-center)**2, axis=1))[:, np.newaxis]
        distortedPoints = distortedPoints*(1+k1*(r**2)+k2*(r**4))
        distortedPoints = distortedPoints*distortedIntrinsics[[0, 1], [0, 1]][np.newaxis, :] + distortedIntrinsics[[0, 1], [2, 2]][np.newaxis, :]
    else :
        distortedPoints = (undistortedPoints - undistortedIntrinsics[[0, 1], [2, 2]][np.newaxis, :])/np.max(undistortedIntrinsics[[0, 1], [2, 2]])
        rdest = np.sqrt(np.sum((distortedPoints-center)**2, axis=1))[:, np.newaxis]
        print undistortedPoints[150*1101+1100/2+1, :], rdest[150*1101+1100/2+1]
        ## rsrc = a*rdest4 + b*rdest3 + c*rdest2 + (1-a-b-c)*rdest a = k2, b = 0, c = k1
#         a = k2; b = 0; c = k1
        a = 0.08090021; b = -0.24602611; c = -0.02017619
#         a *= 1.18189323
#         b *= 1.18189323
#         c *= 1.18189323
        rsrc = a*(rdest**4)+b*(rdest**3)+c*(rdest**2)+(1.0-a-b-c)*rdest
        moveDirs = np.copy(distortedPoints)
        moveDirsNorms = np.linalg.norm(moveDirs, axis=1)
        moveDirsNorms[moveDirsNorms == 0.0] = 1.0
        moveDirs = moveDirs/moveDirsNorms[:, np.newaxis]
        distortedPoints = distortedPoints + moveDirs*(rsrc-rdest)
        distortedPoints = distortedPoints*np.max(distortedIntrinsics[[0, 1], [2, 2]]) + distortedIntrinsics[[0, 1], [2, 2]][np.newaxis, :]
    return distortedPoints

# medianImage = np.array(Image.open("/home/ilisescu/PhD/data/havana/median.png")).astype(np.uint8)
# distortionParameter = -0.19
# distortionRatio = -0.19
# cameraIntrinsics = np.load("/home/ilisescu/PhD/data/havana/filmed_scene-havana.npy").item()[DICT_CAMERA_INTRINSICS]
# undistortedCV, undistortedIntrinsics, distortionCoeff, mapXcv, mapYcv = undistortImage(distortionParameter, distortionRatio, medianImage, cameraIntrinsics, doReturnMaps=True)
# figure(); imshow(undistortedCV)

# multiplier = 1.18189323
# undistortedSize = (np.array(undistortedCV.shape[0:2][::-1])*multiplier).astype(int) ### [x, y]
# gridPoints = np.mgrid[0:undistortedSize[0], 0:undistortedSize[1]].reshape([2, np.prod(undistortedSize)]).T-(undistortedSize-np.array(undistortedCV.shape[0:2][::-1]))/2.0
# mapXY = getDistortedPointsFromUndistorted(gridPoints, distortionParameter, distortionParameter*distortionRatio, cameraIntrinsics, undistortedIntrinsics)
# minCoords = np.floor(gridPoints[np.argmin(np.linalg.norm(mapXY, axis=1)), :]+(undistortedSize-np.array(undistortedCV.shape[0:2][::-1]))/2.0).astype(int)
# maxCoords = np.ceil(gridPoints[np.argmin(np.linalg.norm(mapXY-np.array([[cameraIntrinsics[0, 2]*2-1, cameraIntrinsics[1, 2]*2-1]]), axis=1)), :]+(undistortedSize-np.array(undistortedCV.shape[0:2][::-1]))/2.0).astype(int)+1
# mapXY = mapXY.T.reshape([2, undistortedSize[0], undistortedSize[1]]).T

# figure(); imshow(cv2.remap(medianImage, mapXY[:, :, 0].astype(np.float32), mapXY[:, :, 1].astype(np.float32), cv2.INTER_LANCZOS4)[minCoords[1]:maxCoords[1], minCoords[0]:maxCoords[0], :])
# Image.fromarray(cv2.remap(medianImage, mapXY[:, :, 0].astype(np.float32), mapXY[:, :, 1].astype(np.float32),
#                           cv2.INTER_LANCZOS4)[minCoords[1]:maxCoords[1], minCoords[0]:maxCoords[0], :].astype(np.uint8)).save("/home/ilisescu/PhD/data/havana/medianUndistortNukeTry.png")


# In[6]:

def createAndSaveFilmedSceneNukeData(filmedSceneData) :
    ## get filmed intrinsics and extrinsics from filmed scene
    cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS]
    originalIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]

    ## get a frame and undistort it to get the new intrinsics
    image = np.array(Image.open(filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"frame-00001.png"), np.uint8)
    undistortedImage, undistortedIntrinsics, _, mapXcv, mapYcv = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], image,
                                                                                filmedSceneData[DICT_CAMERA_INTRINSICS], doReturnMaps=True)
    viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, undistortedIntrinsics, undistortedImage.shape[:2])
    print "(image shape, undistorted image shape):", image.shape, undistortedImage.shape
    print undistortedIntrinsics
    
    #################################### CREATE EXR TO SAVE UNDISTORTION ST MAP ####################################
    halfSizeDiff = (np.array(undistortedImage.shape[0:2][::-1])-np.array(image.shape[0:2][::-1]))/2
    cvUndistortSTMapEXRHeaders = OpenEXR.Header(image.shape[1], image.shape[0])
    cvUndistortSTMapEXRHeaders['version'] = 1
    cvUndistortSTMapEXRHeaders['dataWindow'] = Imath.Box2i(Imath.point(-halfSizeDiff[0], -halfSizeDiff[1]), Imath.point(image.shape[1]-1+halfSizeDiff[0], image.shape[0]-1+halfSizeDiff[1]))
    cvUndistortSTMapEXRHeaders['type'] = 'scanlineimage'
    cvUndistortSTMapEXRHeaders['channels'] = {'forward.u': Imath.Channel(type=Imath.PixelType(Imath.PixelType.FLOAT)),
                                              'forward.v': Imath.Channel(type=Imath.PixelType(Imath.PixelType.FLOAT))}

    #### get the undistor maps and map them to the correct interval
    stMapImg = np.zeros([undistortedImage.shape[0], undistortedImage.shape[1], 3])
    stMapImg[:, :, 0] = mapXcv
    stMapImg[:, :, 1] = mapYcv
    ## normalize the maps so that the lower left corner is (0, 0) and the top right corner is (1, 1) as used by the Nuke STMap
    ## [A, B] --> [a, b] --> (val - A)*(b-a)/(B-A) + a
    A = stMapImg[-1, 0, 0:2][np.newaxis, np.newaxis, :]
    B = stMapImg[0, -1, 0:2][np.newaxis, np.newaxis, :]
    a = np.zeros([1, 1, 2], float)
    b = np.ones([1, 1, 2], float)
    stMapImg[:, :, 0:2] = (stMapImg[:, :, 0:2] - A)*(b-a)/(B-A) + a

    cvUndistortSTMapEXR = OpenEXR.OutputFile(filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"undistort_stmap.exr", cvUndistortSTMapEXRHeaders)
    cvUndistortSTMapEXR.writePixels({'forward.u': array.array('f', list(stMapImg[:, :, 0].flatten().astype(np.float32))).tostring(),
                                     'forward.v': array.array('f', list(stMapImg[:, :, 1].flatten().astype(np.float32))).tostring()})

    #################################### GET CAMERA DATA ####################################

    ## compute vertical field of view and from it the vertical focal length which should be the same or very similar to the one in the intrinsics
    vFov = np.arctan2(1.0, projectionMat[1, 1])*2.0*180.0/np.pi
    vFocalLen = float(undistortedImage.shape[0])/(2.0*np.tan(np.pi*vFov/360.0))
    print "(vertical field of view, vertical focal length in pixels):", vFov, vFocalLen
    ## apertures and focal length in mm as used by nuke
    hAperture = 36.0
    vAperture = hAperture*float(undistortedImage.shape[0])/float(undistortedImage.shape[1])
    ## here I could use vFocalLen but it should be the same as the value in the intrinsics and the above formula is mostly to demonstrate how to do this if I don't have the intrinsics and only have a projection mat
    vFocalLen = vAperture/float(undistortedImage.shape[0])*undistortedIntrinsics[1, 1]
    print "in mm: (V focal lenght, H aperture, V aperture)", vFocalLen, hAperture, vAperture

    ## rotate opengl model mat by -90 around x axis
    cameraMat = np.dot(np.array([[1, 0, 0, 0],
                                 [0, np.cos(-np.pi/2.0), -np.sin(-np.pi/2.0), 0],
                                 [0, np.sin(-np.pi/2.0), np.cos(-np.pi/2.0), 0],
                                 [0, 0, 0, 1]]), np.linalg.inv(viewMat))

    nukeData = {'invT': np.linalg.inv(np.dot(undistortedIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])).flatten().tolist(),
                'matrix': cameraMat.flatten().tolist(),
                'focal': vFocalLen,
                'haperture': hAperture,
                'vaperture': vAperture}
    pickle.dump(nukeData, open(filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"nukeData.p", "wb"))


# In[ ]:

def placeBoundingVolumeOnTrajectory(worldPos, worldOrientDir, length, width, height) :
    adjustAngle = np.arccos(np.clip(np.dot(np.array([1.0, 0.0, 0.0]), worldOrientDir), -1, 1))
    adjustAxis = np.cross(worldOrientDir, np.array([1.0, 0.0, 0.0]))
    adjustAxis /= np.linalg.norm(adjustAxis)
    objectTransform = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
    objectTransform[:-1, -1] = worldPos

    worldFootprintVertices = getBillboardVertices(length/width, width)
    worldFootprintVertices = np.dot(objectTransform, np.concatenate([worldFootprintVertices, np.ones([len(worldFootprintVertices), 1])], axis=1).T).T[:, :-1]

    worldBoundingVolumeVertices = np.vstack([worldFootprintVertices,
                                             worldFootprintVertices+np.array([[0, 0, 1.0]])*height])
    
    return worldBoundingVolumeVertices, objectTransform

class LongOperationThread(QtCore.QThread):
    updateOperationProgressSignal = QtCore.Signal(int, str)
    updateOperationTextSignal = QtCore.Signal(str, str)
    abortOperationSignal = QtCore.Signal()
    doneOperationSignal = QtCore.Signal(int)
    
    def __init__(self, threadIdx, parent = None):
        super(LongOperationThread, self).__init__(parent)
        
        self.threadIdx = threadIdx
        self.longOperation = None
        self.args = None
        
    def doCleanUp(self) :
        self.abortOperationSignal.disconnect()
        
        del self.longOperation
        self.longOperation = None
        
    def doQuit(self) :
        self.abortOperationSignal.emit()
        
    def doRun(self, longOperation, args) :
        self.longOperation = longOperation
        self.args = args
        
        if not self.isRunning() and self.longOperation is not None and self.args is not None :
            self.abortOperationSignal.connect(self.longOperation.doAbort)
            self.longOperation.updateOperationProgressSignal.connect(self.updateOperationProgressSignal)
            self.longOperation.updateOperationTextSignal.connect(self.updateOperationTextSignal)
            self.start()
            
    def run(self):
        if self.longOperation is not None and self.args is not None :
#             print "starting the operation"; sys.stdout.flush()
            self.longOperation.run(*self.args)
#             print "ending the operation"; sys.stdout.flush()
            self.doneOperationSignal.emit(self.threadIdx)
        return
    
class LongOperationClass(QtCore.QObject) :
    updateOperationProgressSignal = QtCore.Signal(int, str)
    updateOperationTextSignal = QtCore.Signal(str, str)
    
    def __init__(self, parent = None):
        super(LongOperationClass, self).__init__(parent)
        
        self.abortRequested = False
    
    def doAbort(self) :
        self.abortRequested = True
        
class ProjectiveTextureMeshOperation(LongOperationClass) :
    
    def __init__(self, parent = None):
        super(ProjectiveTextureMeshOperation, self).__init__(parent)
        
        self.texturedMeshImage = np.empty([0, 0], dtype=np.uint8)
        
    def __del__(self) :
        del self.texturedMeshImage
    
    def run(self, image, filmedDataset, viewMat, projectionMat) :
        """ Textures a mesh as defined in filmedDataset using projective texturing and projects it onto texturedMeshImage as seen from the current view
        assumes image.shape == filmedDataset.undistortedImage.shape
        viewMat and projectionMat are the camera matrices for the current view
        """
        
        totalNumWarps = float(1+len(filmedDataset.worldExtrudedGeometryPoints)/4)
        
        originalViewMat, originalProjectionMat = cvCameraToOpenGL(filmedDataset.cameraExtrinsics, filmedDataset.undistortedIntrinsics, image.shape[0:2])
        worldPointsOnGroundPlane = filmedDataset.getWorldSpaceRectangle()
        originalCameraPointsOnGroundPlane = worldToScreenSpace(originalViewMat, originalProjectionMat, worldPointsOnGroundPlane, image.shape[1], image.shape[0])

        if self.abortRequested :
            return
        ## find points in current view
        cameraPointsOnGroundPlane = worldToScreenSpace(viewMat, projectionMat, worldPointsOnGroundPlane, image.shape[1], image.shape[0])
        ## project the masked image onto ground plane using this homography
        M = cv2.findHomography(originalCameraPointsOnGroundPlane, cameraPointsOnGroundPlane)[0]

        imgMask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
        cv2.fillPoly(imgMask, [np.round(filmedDataset.cameraGroundPlaneGeometryPoints).astype(np.int32)], [255])

        warpedImage = cv2.warpPerspective(np.concatenate([image, imgMask[:, :, np.newaxis]], axis=-1).astype(np.uint8), M, tuple(image.shape[0:2][::-1]))

        self.texturedMeshImage = (warpedImage[:, :, :-1]*(warpedImage[:, :, -1]/255.0)[:, :, np.newaxis]).astype(np.uint8)
        if self.abortRequested :
            return
        self.updateOperationProgressSignal.emit(1.0/totalNumWarps*100.0, "")
        
        cameraExtrudedGeometryPoints = worldToScreenSpace(viewMat, projectionMat, filmedDataset.worldExtrudedGeometryPoints, image.shape[1], image.shape[0])

        extrudedFacesStartIndices = np.arange(0, len(cameraExtrudedGeometryPoints), 4)
        ## compute depth for each face
        camPos = getWorldSpacePosAndNorm(np.linalg.inv(viewMat), posOnly=True)
        extrudedFacesDepths = [np.sum(np.linalg.norm(filmedDataset.worldExtrudedGeometryPoints[i:i+4, :]-camPos[np.newaxis, :], axis=1)) for i in extrudedFacesStartIndices]
        extrudedFacesStartIndices = extrudedFacesStartIndices[np.argsort(extrudedFacesDepths)[::-1]]
        for idx, i in enumerate(extrudedFacesStartIndices) :
            imgMask = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
            cv2.fillPoly(imgMask, [np.round(filmedDataset.cameraExtrudedGeometryPoints[i:i+4, :]).astype(np.int32)], [255])
            ## use this to morph the masked image into the current face seen from current view
            M = cv2.findHomography(filmedDataset.cameraExtrudedGeometryPoints[i:i+4, :],
                                   cameraExtrudedGeometryPoints[i:i+4, :])[0]

            warpedImage = cv2.warpPerspective(np.concatenate([image, imgMask[:, :, np.newaxis]], axis=-1).astype(np.uint8), M, tuple(image.shape[0:2][::-1]))
            self.texturedMeshImage = (warpedImage[:, :, :-1]*(warpedImage[:, :, -1]/255.0)[:, :, np.newaxis] +
                                      self.texturedMeshImage*(1.0-warpedImage[:, :, -1]/255.0)[:, :, np.newaxis]).astype(np.uint8)
            
            if self.abortRequested :
                return
            self.updateOperationProgressSignal.emit(float(2+idx)/totalNumWarps*100.0, "")
            
class ComputePatchesForBillboardOperation(LongOperationClass) :
    
    def __init__(self, objectName, parent = None):
        super(ComputePatchesForBillboardOperation, self).__init__(parent)
        
        self.objectName = objectName
        self.orientationAngles = np.empty([0], dtype=float)
        self.maxBillboardHeight = 0.0
        
    def __del__(self) :
        del self.orientationAngles, self.maxBillboardHeight
    
    def run(self, patchesSaveLoc, filmedSceneData, bgImage, cameraIntrinsics, trajectoryPoints, sortedFrameKeys,
            objectLength, objectWidth, objectHeight, masksLocation, segmentationThreshold=0.8, verbose=False) :
        """
        everything is in the image space of the undistorted image apart from images loaded from disk
        sortedFrameKeys are the keys of the frames each trajectory point is found in
        bgImage and cameraIntrinsics are after undistortion
        """
        ##The tidied up methods used here can be found in their original form in 3D Video Looping ##
        if self.abortRequested :
            return
        self.updateOperationProgressSignal.emit(0.0, self.objectName)

        cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS]
        originalIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]
        distortionParameter = filmedSceneData[DICT_DISTORTION_PARAMETER]
        distortionRatio = filmedSceneData[DICT_DISTORTION_RATIO]
        baseLoc = filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]
        
        height, width = bgImage.shape[:2]
        viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, bgImage.shape)
        maxBillboardWidth = 0
        maxBillboardHeight = 0
        usedOrientationsAndSizes = {}
        worldTrajectoryPoints = screenToWorldPlane(trajectoryPoints, cameraIntrinsics, cameraExtrinsics)
        worldTrajectoryDirections = getDirectionsFromTrajectoryPoints(worldTrajectoryPoints)
        

        #################################### LOAD IMAGES ####################################
        startLoadingTime = time.time()
        images = np.zeros([bgImage.shape[0], bgImage.shape[1], 4, len(sortedFrameKeys)], dtype=np.uint8)
        patchesData = []
        if self.abortRequested :
            del images, patchesData
            return
        self.updateOperationProgressSignal.emit(2.0, self.objectName)
        for patchIdx, patchKey in enumerate(sortedFrameKeys) :
#             startTime = time.time()
            patchFrameName = "frame-{0:05}.png".format(patchKey+1)
            originalImage = np.array(Image.open(baseLoc+os.sep+patchFrameName)).astype(np.uint8)
            patchImage, _, distortionCoeff = undistortImage(distortionParameter, distortionRatio, originalImage, originalIntrinsics)
            ## load mask from disk
            if masksLocation != "" and os.path.isdir(masksLocation) :
                originalAlpha = np.array(Image.open(masksLocation+os.sep+patchFrameName)).astype(np.uint8)
                patchAlpha = undistortImage(distortionParameter, distortionRatio, originalAlpha, originalIntrinsics)[0][:, :, -1]
            else :
            ## make mask by rendering the convex hull of the bounding volume
                patchAlpha = np.zeros_like(patchImage)
                
                #0.58, 0.27, 0.18 (blue_car1)
                worldBoundingVolumeVertices, _ = placeBoundingVolumeOnTrajectory(worldTrajectoryPoints[patchIdx, :], worldTrajectoryDirections[patchIdx, :], objectLength, objectWidth, objectHeight)

                cameraBoundingVolumeVertices = worldToScreenSpace(viewMat, projectionMat, worldBoundingVolumeVertices, width, height)
                cameraBoundingVolumeVertices = cv2.convexHull(cameraBoundingVolumeVertices.astype(np.float32))[:, 0, :]

                cv2.fillConvexPoly(patchAlpha, cameraBoundingVolumeVertices.astype(np.int32), (255, 255, 255))

            ## threshold alpha
            patchAlpha, patchData = getPatchDataWithThresholdedAlpha(patchImage, patchAlpha[:, :, -1], bgImage, segmentationThreshold)
            images[:, :, :-1, patchIdx] = patchImage
            images[:, :, -1, patchIdx] = patchAlpha
            patchesData.append(patchData)

#             Image.fromarray(patchImage).save("patchImage.png")
#             Image.fromarray(patchAlpha).save("patchAlpha.png")
#             patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
#             patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]
#             Image.fromarray(patchColors).save("patchColors.png")

#             print time.time()-startTime
            if np.mod(patchIdx, 10) == 0 :
                if self.abortRequested :
                    del images, patchesData
                    return
                self.updateOperationProgressSignal.emit((1.0+patchIdx)/float(len(sortedFrameKeys))*58.0+2.0, self.objectName)

        if self.abortRequested :
            del images, patchesData
            return
        print "loaded images in", time.time()-startLoadingTime; sys.stdout.flush()
        self.updateOperationProgressSignal.emit(60.0, self.objectName)
        
        #################################### FIND MAX BILLBOARD SIZE AND ITS ORIENTATION AT EACH TRAJECTORY POINT ####################################
        for patchIdx, patchKey in enumerate(sortedFrameKeys) :
            billboardWidth, billboardHeight, usedOrientationsAndSizes[patchIdx] = findEncompassingBillboardSize(images[:, :, :-1, patchIdx], images[:, :, -1, patchIdx], patchesData[patchIdx],
                                                                                                                np.copy(worldTrajectoryPoints[patchIdx, :]),
                                                                                                                np.copy(worldTrajectoryDirections[patchIdx, :]), cameraExtrinsics,
                                                                                                                cameraIntrinsics, bgImage.shape[0:2], verbose)
            if billboardWidth > maxBillboardWidth :
                maxBillboardWidth = billboardWidth
                if verbose :
                    print("NEW MAX WIDTH", patchIdx, maxBillboardWidth)
            if billboardHeight > maxBillboardHeight :
                maxBillboardHeight = billboardHeight
                if verbose :
                    print("NEW MAX HEIGHT", patchIdx, billboardHeight)
            if verbose :
                print(patchIdx, patchKey, billboardWidth, billboardHeight, maxBillboardWidth, maxBillboardHeight)
                
            if np.mod(patchIdx, 20) == 0 :
                if self.abortRequested :
                    del images, patchesData
                    return
                self.updateOperationProgressSignal.emit((1.0+patchIdx)/float(len(sortedFrameKeys))*5.0 + 60.0, self.objectName)

        if self.abortRequested :
            del images, patchesData
            return
        print "found max billboard [", maxBillboardWidth,",", maxBillboardHeight, "]"; sys.stdout.flush()
        self.updateOperationProgressSignal.emit(65.0, self.objectName)

        #################################### UNDISTORT THE PATCHES SO THAT WHEN USED TO TEXTURE A BILLBOARD IT PROJECTS IN THE ORIGINAL VIEW PROPERLY AND SAVE ####################################
        undistortedPatches = {}
        for patchIdx, patchKey in enumerate(sortedFrameKeys) :
            if verbose :
                print patchIdx,
            
            ## check that patchIdx is in usedOrientationsAndSizes (it could not be in there if something goes wrong in findEncompassingBillboardSize)
            ## check that billboard widths and heights are non-zero
            if patchIdx in usedOrientationsAndSizes.keys() and usedOrientationsAndSizes[patchIdx][1] != 0.0 and usedOrientationsAndSizes[patchIdx][2] != 0.0 :
                undistortedPatches[patchKey] =  undistortPatch(images[:, :, :-1, patchIdx], images[:, :, -1, patchIdx], patchesData[patchIdx],
                                                               np.copy(worldTrajectoryPoints[patchIdx, :]), usedOrientationsAndSizes[patchIdx],
                                                               maxBillboardWidth, maxBillboardHeight, cameraExtrinsics, cameraIntrinsics, bgImage, worldTrajectoryPoints, verbose)
                
#             if np.mod(patchIdx, 5) == 0 :
            if self.abortRequested :
                del images, patchesData
                return
            self.updateOperationProgressSignal.emit((1.0+patchIdx)/float(len(sortedFrameKeys))*30.0 + 65.0, self.objectName)
                
        del images, patchesData
        
        if self.abortRequested :
            return
        np.save(patchesSaveLoc, undistortedPatches)
        if self.abortRequested :
            return
        self.updateOperationProgressSignal.emit(95.0, self.objectName)
        
        
    
        #################################### COMPUTE ORIENTATION ANGLES AND SCALE FOR RENDERING ####################################
        self.orientationAngles = np.zeros(len(usedOrientationsAndSizes))
        for frameIdx in np.arange(len(usedOrientationsAndSizes)) :
            if self.abortRequested :
                return
            worldPos = np.copy(worldTrajectoryPoints[frameIdx, :])
            worldMovingDir = np.copy(worldTrajectoryDirections[frameIdx, :])
            worldMoveNormalDir = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([0.0, 0.0, 1.0]))), np.concatenate([worldMovingDir, [1]]))
            worldMoveNormalDir = worldMoveNormalDir[:-1]/np.linalg.norm(worldMoveNormalDir[:-1])
            worldOrientDir, billboardWidth, billboardHeight, dirMultiplier = usedOrientationsAndSizes[frameIdx]
            ## check if the chosen orienation direction is the opposite of the moving direction in which case, the desired angle should be 0.0 (I will add the 180.0 to flip it later on when checking the adjust axis)
            if np.sum(np.sqrt((worldMovingDir+worldOrientDir)**2)) < 1e-10 :
                adjustAngle = 0.0
            else :
                adjustAngle = np.arccos(np.clip(np.dot(worldMovingDir, worldOrientDir), -1, 1))
            ## if the adjust angle is not 0, it means I'm picking the normal to the trajectory as orientation and thus I can compute the adjust axis (which will tell me if I need to flip)
            if adjustAngle != 0.0 :
                adjustAxis = np.cross(worldOrientDir, worldMovingDir)
                adjustAxis /= np.linalg.norm(adjustAxis)
            ## otherwise the adjust axis depends on the dirMultiplier which tells me whether I need to flip the billboard oriented using the tangent to the trajectory
            else :
                adjustAxis = np.array([0.0, 0.0, dirMultiplier])
            self.orientationAngles[frameIdx] = adjustAngle
            ## I can achieve the same rotation as using the flipped z axis, by adding 180 degrees to the current rotation
            if adjustAxis[-1] < 0.0 :
                self.orientationAngles[frameIdx] += np.pi
                
                
        if self.abortRequested :
            return
        
        
# filmedObjectData[DICT_PATCHES_LOCATION] = updatedPatchesLoc
        
# filmedObjectData[DICT_OBJECT_BILLBOARD_ORIENTATION] = orientationAngles
## the scale of the billboard is the same as maxBillboardHeight because the way I define billboards in my app is by keeping the height fixed to 1 and setting the width based on the aspect ratio of the 
## texture; the size of the texture is based on maxBillboardHeight and maxBillboardWidth and all patches are mapped to the exact same size texture; then I map the smallest billboard size for each frame to a rectangle
## centerd within the texture, with the same aspect ratio and scaled down to make sure that its height is the same relative to maxBillboardHeight; it's analogous to instead always mapping the max size billboard to the 
## texture rectangle (which would probably be easier in retrospect); the problem is then that I always map a billboard of height maxBillboardHeight and correct aspect ratio, to the texture rectangle with the same 
## aspect ratio, but if I just use the default GLBillboard, I artificially map the texture onto a bigger or smaller billboard than the one used when it was computed, resulting in bigger or smaller sprites; to solve this
## then, all I need to do is scale the whole GLBillboard using maxBillboardHeight which will uniformly scale the full billboard to have the correct height and aspect ratio; alternatively, I could have computed the
## max billboard size to get the aspect ratio and then used a billboard of height=1 and of width=ratio and mapped that to the texture rectangle and then I wouldn't have had to scale the GLBillboard
# filmedObjectData[DICT_OBJECT_BILLBOARD_SCALE] = maxBillboardHeight
# print(maxBillboardHeight)
        self.maxBillboardHeight = np.copy(maxBillboardHeight)
    
    
        self.updateOperationProgressSignal.emit(100.0, self.objectName)
        
        print "done computing patches"


# filmedObjectData[DICT_TRAJECTORY_POINTS] = trajectoryPoints + originalIntrinsics[:2, -1] - cameraIntrinsics[:2, -1]
# np.save(np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx], filmedObjectData)


# In[ ]:

DRAW_FIRST_FRAME = 'first_frame'
DRAW_LAST_FRAME = 'last_frame'
DRAW_COLOR = 'color'
LIST_SECTION_SIZE = 60
SLIDER_INDICATOR_WIDTH=4

class SemanticsSlider(QtGui.QSlider) :
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None) :
        super(SemanticsSlider, self).__init__(orientation, parent)
        style = "QSlider::handle:horizontal { background: #cccccc; width: 0px; border-radius: 0px; } "
        style += "QSlider::groove:horizontal { background: #dddddd; } "
        self.setStyleSheet(style)
        
        self.semanticsToDraw = []
        self.numOfFrames = 1
        self.selectedSemantics = -1
        
    def setSelectedSemantics(self, selectedSemantics) :
        self.selectedSemantics = selectedSemantics
        
    def setSemanticsToDraw(self, semanticsToDraw, numOfFrames) :
        self.semanticsToDraw = semanticsToDraw
        self.numOfFrames = float(numOfFrames)
        
        desiredHeight = np.max((42, len(self.semanticsToDraw)*7))
        self.setFixedHeight(desiredHeight)
        
        self.resize(self.width(), self.height())
        self.update()
        
    def mousePressEvent(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            self.setValue(event.pos().x()*(float(self.maximum())/self.width()))
        
    def paintEvent(self, event) :
        super(SemanticsSlider, self).paintEvent(event)
        
        painter = QtGui.QPainter(self)
        
        ## draw semantics
        
        yCoord = 0.0
        for i in xrange(len(self.semanticsToDraw)) :
            col = self.semanticsToDraw[i][DRAW_COLOR]

            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255)))
            startX =  self.semanticsToDraw[i][DRAW_FIRST_FRAME]/self.numOfFrames*self.width()
            endX =  self.semanticsToDraw[i][DRAW_LAST_FRAME]/self.numOfFrames*self.width()

            if self.selectedSemantics == i :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 127), 1, 
                                              QtCore.Qt.DashLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)

            else :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 63), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)


            yCoord += 7        
        
        ## draw slider
        ## mapping slider interval to its size
        A = 0.0
        B = float(self.maximum())
        a = SLIDER_INDICATOR_WIDTH/2.0
        b = float(self.width())-SLIDER_INDICATOR_WIDTH/2.0
        if (B-A) != 0.0 :
            ## (val - A)*(b-a)/(B-A) + a
            sliderXCoord = (float(self.value()) - A)*(b-a)/(B-A) + a
        else :
            sliderXCoord = a
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), SLIDER_INDICATOR_WIDTH,
                                          QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
        painter.drawLine(sliderXCoord, 0, sliderXCoord, self.height())
        
        painter.end()


# In[ ]:

POINT_SELECTION_RADIUS = 20

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        self.setMouseTracking(True)
        
        self.maxImageShape = np.array([720, 1280])
        self.resizeRatio = 1.0
        
        self.image = np.empty([0, 0], np.uint8)
        self.qImage = None
        self.overlayImg = np.empty([0, 0], np.uint8)
        self.overlayQImg = None
        
        self.doDrawControls = False
        self.selectedPoint = np.empty([0], dtype=float)
        
        self.doDrawLines = False
        self.doDrawLineControls = False
        self.lines = np.empty([0, 4], dtype=float)
        self.doDrawRectangle = False
        self.rectanglePoints = np.empty([0, 2], dtype=float)
        self.sceneOriginPoint = np.empty([0], dtype=float)
        self.doDrawFrustumPoints = False
        self.frustumPoints = np.empty([0, 2], dtype=float)
        
        self.doDrawMesh = False
        self.meshPoints = np.empty([0, 2], dtype=float)
        self.meshIndices = np.empty([0], dtype=int)
        self.doShowPointIdx = np.empty([0], dtype=bool)
        
        self.doDrawTrajectories = False
        self.trajectories = []
        self.boundingVolumes = []
        
    def setImage(self, image) :
        if len(image) > 0 and np.any(image.shape[0:2] > self.maxImageShape) :
            self.resizeRatio = np.min(self.maxImageShape.astype(float)/image.shape[0:2])
            self.image = cv2.resize(image, (0, 0), fx=self.resizeRatio, fy=self.resizeRatio, interpolation=cv2.INTER_CUBIC)
        else :
            self.resizeRatio = 1.0
            self.image = image
            
        if len(image) > 0 :
            self.qImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QtGui.QImage.Format_RGB888);
            self.setMinimumSize(self.qImage.size())
        else :
            self.qImage = None
        self.update()
        
    def setOverlayImg(self, overlayImg) :
        if len(overlayImg) > 0 and np.any(overlayImg.shape[0:2] > self.image.shape[0:2]) :
            resizeRatio = np.min(np.array(self.image.shape[0:2]).astype(float)/overlayImg.shape[0:2])
            self.overlayImg = cv2.resize(overlayImg, (0, 0), fx=self.resizeRatio, fy=self.resizeRatio, interpolation=cv2.INTER_CUBIC)
        else :
            self.overlayImg = overlayImg
            
        if len(self.overlayImg) > 0 :
            self.overlayQImg = QtGui.QImage(self.overlayImg.data, self.overlayImg.shape[1], self.overlayImg.shape[0], self.overlayImg.strides[0], QtGui.QImage.Format_ARGB32);
        else :
            self.overlayQImg = None
        self.update()
        
    def setSelectedPoint(self, selectedPoint) :
        if len(selectedPoint) != len(self.selectedPoint) or np.linalg.norm(selectedPoint-self.selectedPoint) != 0.0 :
            self.selectedPoint = selectedPoint
            self.update()
        
    def setLines(self, lines) :
        self.lines = lines
        self.update()
        
    def setRectanglePoints(self, rectanglePoints) :
        self.rectanglePoints = rectanglePoints
        self.update()
        
    def setSceneOriginPoint(self, sceneOriginPoint) :
        self.sceneOriginPoint = sceneOriginPoint
        self.update()
        
    def setFrustumPoints(self, frustumPoints) :
        self.frustumPoints = frustumPoints
        self.update()
        
    def setMesh(self, meshPoints, meshIndices, doShowPointIdx) :
        if len(meshPoints) == len(doShowPointIdx) :
            if len(meshPoints) > 0 and (np.min(meshIndices) < 0 or np.max(meshIndices) >= len(meshPoints)) :
                raise Exception("Setting mesh with invalid indices!")
            self.meshPoints = meshPoints
            self.meshIndices = meshIndices
            self.doShowPointIdx = doShowPointIdx
            self.update()
        
    def setTrajectories(self, trajectories, colors) :
        if len(trajectories) == len(colors) :
            self.trajectories = [(t, c) for t, c in zip(trajectories, colors)]
            self.update()
                      
    def setBoundingVolumes(self, boundingVolumes, colors) :
        if len(boundingVolumes) == len(colors) :
            self.boundingVolumes = [(v, c) for v, c in zip(boundingVolumes, colors)]
            self.update()
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        if self.qImage is not None :
            upperLeft = np.array([(self.width()-self.qImage.width())/2, (self.height()-self.qImage.height())/2])
            
            ## draw image
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.qImage)
            
            if self.doDrawTrajectories and self.overlayQImg is not None :
                painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.overlayQImg)
                        
            if self.doDrawControls and len(self.selectedPoint) == 2 :
                ## draw selected point
                selectedPoint = self.selectedPoint*self.resizeRatio + upperLeft
                
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(225, 225, 225, 128)))
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 128, 128), 1, 
                                                  QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

                painter.drawEllipse(QtCore.QPointF(selectedPoint[0], selectedPoint[1]), POINT_SELECTION_RADIUS, POINT_SELECTION_RADIUS)
                painter.setBrush(QtCore.Qt.NoBrush)
                    
            if self.doDrawMesh and len(self.meshPoints) > 0 and len(self.meshIndices) > 0 and np.min(self.meshIndices) >= 0 and np.max(self.meshIndices) < len(self.meshPoints) :
                ## draw mesh triangles
                meshPoints = self.meshPoints*self.resizeRatio + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                for i, j, k in zip(self.meshIndices[0::3], self.meshIndices[1::3], self.meshIndices[2::3]) :
                    ## draw line
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(127, 0, 0, 127), 1, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    ## could use paintPolygon but cba
                    painter.drawLine(QtCore.QPointF(meshPoints[i, 0], meshPoints[i, 1]),
                                     QtCore.QPointF(meshPoints[j, 0], meshPoints[j, 1]))
                    painter.drawLine(QtCore.QPointF(meshPoints[j, 0], meshPoints[j, 1]),
                                     QtCore.QPointF(meshPoints[k, 0], meshPoints[k, 1]))
                    painter.drawLine(QtCore.QPointF(meshPoints[k, 0], meshPoints[k, 1]),
                                     QtCore.QPointF(meshPoints[i, 0], meshPoints[i, 1]))
                
                ## draw points and indices
                for i, point in enumerate(meshPoints) :
                    if self.doShowPointIdx[i] :
                        ## draw points
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 6, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawPoint(QtCore.QPointF(point[0], point[1]))
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 4, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawPoint(QtCore.QPointF(point[0], point[1]))

                        ## draw corner idx text
                        textMoveDir = (point-meshPoints[np.mod(i+1, len(meshPoints)), :])/np.linalg.norm(point-meshPoints[np.mod(i+1, len(meshPoints)), :])
                        textMoveDir += (point-meshPoints[i-1, :])/np.linalg.norm(point-meshPoints[i-1, :])
                        textMoveDir /= np.linalg.norm(textMoveDir)
                        textPath = QtGui.QPainterPath()
                        textPath.addText(QtCore.QPointF(point[0]+textMoveDir[0]*10, point[1]+textMoveDir[1]*10), QtGui.QApplication.font(), np.string_(i+1))
                        textPath.translate(-textPath.boundingRect().width()/2.0, textPath.boundingRect().height()/2.0)

                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 5, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawPath(textPath)
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 2, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawPath(textPath)
                    
            
            if self.doDrawLines and len(self.lines) == 4 :
                ## draw lines
                lines = self.lines*self.resizeRatio + np.repeat(np.array([[upperLeft[0], upperLeft[1]]]), 2, axis=0).flatten().reshape((1, 4))
                for i in xrange(len(lines)) :
                    ## get intersection with the box of the label
                    intersectionPoints = getLineRectIntersectionPoints(lines[i, :], np.array([[upperLeft[0], upperLeft[1], upperLeft[0]+self.qImage.width(), upperLeft[1]],
                                                                                              [upperLeft[0]+self.qImage.width(), upperLeft[1], 
                                                                                               upperLeft[0]+self.qImage.width(), upperLeft[1]+self.qImage.height()],
                                                                                              [upperLeft[0]+self.qImage.width(), upperLeft[1]+self.qImage.height(), 
                                                                                               upperLeft[0], upperLeft[1]+self.qImage.height()],
                                                                                              [upperLeft[0], upperLeft[1]+self.qImage.height(), upperLeft[0], upperLeft[1]]], dtype=float))
                    if len(intersectionPoints) == 2 :
                        ## draw line
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 4, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawLine(QtCore.QPointF(intersectionPoints[0, 0], intersectionPoints[0, 1]),
                                         QtCore.QPointF(intersectionPoints[1, 0], intersectionPoints[1, 1]))
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 2, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawLine(QtCore.QPointF(intersectionPoints[0, 0], intersectionPoints[0, 1]),
                                         QtCore.QPointF(intersectionPoints[1, 0], intersectionPoints[1, 1]))

                        if self.doDrawLineControls :
                            ## draw points
                            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 10, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                            painter.drawPoint(QtCore.QPointF(lines[i, 0], lines[i, 1]))
                            painter.drawPoint(QtCore.QPointF(lines[i, 2], lines[i, 3]))
                            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 8, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                            painter.drawPoint(QtCore.QPointF(lines[i, 0], lines[i, 1]))
                            painter.drawPoint(QtCore.QPointF(lines[i, 2], lines[i, 3]))
                    
                ## draw horizon line
                try :
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(192, 0, 0, 192), 10, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    point1 = line2lineIntersection(lines[0, :], lines[1, :])
                    point2 = line2lineIntersection(lines[2, :], lines[3, :])
                    intersectionPoints = getLineRectIntersectionPoints(np.concatenate([point1, point2]), np.array([[upperLeft[0], upperLeft[1], upperLeft[0]+self.qImage.width(), upperLeft[1]],
                                                                                                                   [upperLeft[0]+self.qImage.width(), upperLeft[1], 
                                                                                                                    upperLeft[0]+self.qImage.width(), upperLeft[1]+self.qImage.height()],
                                                                                                                   [upperLeft[0]+self.qImage.width(), upperLeft[1]+self.qImage.height(), 
                                                                                                                    upperLeft[0], upperLeft[1]+self.qImage.height()],
                                                                                                                   [upperLeft[0], upperLeft[1]+self.qImage.height(), upperLeft[0], upperLeft[1]]], dtype=float))
                    if len(intersectionPoints) == 2 :
                            painter.drawLine(QtCore.QPointF(intersectionPoints[0, 0], intersectionPoints[0, 1]),
                                             QtCore.QPointF(intersectionPoints[1, 0], intersectionPoints[1, 1]))
                except Exception:
                    pass
            
            
            if self.doDrawRectangle and len(self.rectanglePoints) == 4 :
                ## draw intersection rectangle
                rectanglePoints = self.rectanglePoints*self.resizeRatio + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                for idx, (i, j) in enumerate(zip(rectanglePoints, rectanglePoints[np.roll(np.arange(len(rectanglePoints)), -1), :])) :
                    ## draw line
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 4, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 2, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
                    
                    ## draw corner idx text
                    textMoveDir = (i-rectanglePoints[np.mod(idx+1, len(rectanglePoints)), :])/np.linalg.norm(i-rectanglePoints[np.mod(idx+1, len(rectanglePoints)), :])
                    textMoveDir += (i-rectanglePoints[idx-1, :])/np.linalg.norm(i-rectanglePoints[idx-1, :])
                    textMoveDir /= np.linalg.norm(textMoveDir)
                    textPath = QtGui.QPainterPath()
                    textPath.addText(QtCore.QPointF(i[0]+textMoveDir[0]*10, i[1]+textMoveDir[1]*10), QtGui.QApplication.font(), np.string_(idx+1))
                    textPath.translate(-textPath.boundingRect().width()/2.0, textPath.boundingRect().height()/2.0)
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 5, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawPath(textPath)
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 2, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawPath(textPath)
                    
                ## draw scene origin 
                if len(self.sceneOriginPoint) == 2 :
                    sceneOriginPoint = self.sceneOriginPoint*self.resizeRatio + upperLeft
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 10, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawPoint(QtCore.QPointF(sceneOriginPoint[0], sceneOriginPoint[1]))
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 8, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawPoint(QtCore.QPointF(sceneOriginPoint[0], sceneOriginPoint[1]))
                
            if self.doDrawFrustumPoints and len(self.frustumPoints) > 0 :
                ## draw frustum 
                frustumPoints = self.frustumPoints*self.resizeRatio + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                for i, j in zip(frustumPoints[0:-1:2, :], frustumPoints[1::2, :]) :
                    ## draw line
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 2, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
                
            
            if self.doDrawTrajectories :
                for trajIdx in np.arange(len(self.trajectories)) :
                    ## draw trajectory
                    trajectory = self.trajectories[trajIdx][0]*self.resizeRatio + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))

                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(self.trajectories[trajIdx][1][0], self.trajectories[trajIdx][1][1], self.trajectories[trajIdx][1][2], 255), 2, 
                                   QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for i, j in zip(trajectory[:-1, :], trajectory[1:, :]) :
                        painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                         QtCore.QPointF(j[0], j[1]))
                        
                for volIdx in np.arange(len(self.boundingVolumes)) :
                    ## draw trajectory
                    boundingVolume = self.boundingVolumes[volIdx][0]*self.resizeRatio + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))

                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(self.boundingVolumes[volIdx][1][0], self.boundingVolumes[volIdx][1][1], self.boundingVolumes[volIdx][1][2], 255), 2, 
                                   QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for i, j in zip(boundingVolume[[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], :], boundingVolume[[1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7], :]) :
                        painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                         QtCore.QPointF(j[0], j[1]))

#             ## draw bbox rectangle
#             if self.bboxRectangle != None :
#                 bboxRectangle = self.bboxRectangle + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
#                 painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 2, 
#                               QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
#                 for i, j in zip(bboxRectangle, bboxRectangle[np.roll(np.arange(len(bboxRectangle)), -1), :]) :
                    
#                     painter.drawLine(QtCore.QPointF(i[0], i[1]),
#                                      QtCore.QPointF(j[0], j[1]))


# In[ ]:

class NewFilmedObjectDialog(QtGui.QDialog):
    def __init__(self, parent=None, title="", baseLoc=os.path.expanduser("~")) :#, image=None):
        super(NewFilmedObjectDialog, self).__init__(parent)
        
        self.baseLoc = baseLoc
        self.representativeColor = np.array([255, 255, 255])
        self.trackLocation = ""
        self.masksLocation = ""
        
        self.createGUI()
        
        self.setWindowTitle(title)
        
    def accept(self):        
        self.done(1)
    
    def reject(self):
        self.done(0)
    
    def setRepresentativeColor(self) :
        newColor = QtGui.QColorDialog.getColor(QtGui.QColor(self.representativeColor[0],
                                                            self.representativeColor[1],
                                                            self.representativeColor[2]), self, "Choose Representative Color")
        if newColor.isValid() :
            self.representativeColor = np.array([newColor.red(), newColor.green(), newColor.blue()])
            textColor = "black"
            if np.average(self.representativeColor) < 127 :
                textColor = "white"
            self.colorButton.setStyleSheet("QPushButton {border: 1px solid "+textColor+"; background-color: rgb("+", ".join(self.representativeColor.astype(np.string_))+"); color: "+textColor+"; }")
            
    def setTrackLocation(self) :
        self.trackLocation = QtGui.QFileDialog.getOpenFileName(self, "Set Track Location", self.baseLoc, "Tracks (*track*.txt)")[0]
        self.trackLocationLabel.setText(self.trackLocation)
        
    def setMasksLocation(self) :
        self.masksLocation = QtGui.QFileDialog.getExistingDirectory(self, "Set Masks Location", self.baseLoc)
        self.masksLocationLabel.setText(self.masksLocation)
    
    def createGUI(self) :
        self.nameEdit = QtGui.QLineEdit()
        self.nameEdit.setText("object_name")
        
        self.lengthSpinBox = QtGui.QDoubleSpinBox()
        self.lengthSpinBox.setRange(0.0, 100.0)
        self.lengthSpinBox.setSingleStep(0.1)
        self.lengthSpinBox.setValue(1.0)
        
        self.widthSpinBox = QtGui.QDoubleSpinBox()
        self.widthSpinBox.setRange(0.0, 100.0)
        self.widthSpinBox.setSingleStep(0.1)
        self.widthSpinBox.setValue(1.0)
        
        self.heightSpinBox = QtGui.QDoubleSpinBox()
        self.heightSpinBox.setRange(0.0, 100.0)
        self.heightSpinBox.setSingleStep(0.1)
        self.heightSpinBox.setValue(1.0)
        
        self.trackLocationLabel = QtGui.QLabel("<b>Please choose a track.</b> (Required)")
        self.setTrackLocationButton = QtGui.QPushButton("Find Track")
        
        self.masksLocationLabel = QtGui.QLabel("Please choose masks directory. (Optional)")
        self.setMasksLocationButton = QtGui.QPushButton("Find Masks")
        
        self.colorButton = QtGui.QPushButton("Choose Color")
        self.colorButton.setStyleSheet("QPushButton {border: 1px solid black; background-color: white;}")
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel);
         
        ## SIGNALS ##
        
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.setTrackLocationButton.clicked.connect(self.setTrackLocation)
        self.setMasksLocationButton.clicked.connect(self.setMasksLocation)
        self.colorButton.clicked.connect(self.setRepresentativeColor)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QGridLayout()
        idx = 0
        mainLayout.addWidget(QtGui.QLabel("Name:"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(self.nameEdit, idx, 1, 1, 3); idx+=1
        mainLayout.addWidget(QtGui.QLabel("Length"), idx, 1, 1, 1, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(QtGui.QLabel("Width"), idx, 2, 1, 1, QtCore.Qt.AlignCenter)
        mainLayout.addWidget(QtGui.QLabel("Height"), idx, 3, 1, 1, QtCore.Qt.AlignCenter); idx+=1
        mainLayout.addWidget(QtGui.QLabel("Volume Box Size [m]:"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(self.lengthSpinBox, idx, 1, 1, 1)
        mainLayout.addWidget(self.widthSpinBox, idx, 2, 1, 1)
        mainLayout.addWidget(self.heightSpinBox, idx, 3, 1, 1); idx+=1
        mainLayout.addWidget(self.setTrackLocationButton, idx, 0, 1, 1)
        mainLayout.addWidget(self.trackLocationLabel, idx, 1, 1, 3, QtCore.Qt.AlignLeft); idx+=1
        mainLayout.addWidget(self.setMasksLocationButton, idx, 0, 1, 1)
        mainLayout.addWidget(self.masksLocationLabel, idx, 1, 1, 3, QtCore.Qt.AlignLeft); idx+=1
        mainLayout.addWidget(self.colorButton, idx, 0, 1, 4); idx+=1
        mainLayout.addWidget(self.buttonBox, idx, 0, 1, 4, QtCore.Qt.AlignCenter); idx+=1
        
        self.setLayout(mainLayout)

def createNewFilmedObject(parent=None, title="Dialog", baseLoc=os.path.expanduser("~")) :#, image= None) :
    newFilmedObjectDialog = NewFilmedObjectDialog(parent, title, baseLoc)#, image)
    exitCode = newFilmedObjectDialog.exec_()
    
    return (newFilmedObjectDialog.nameEdit.text(), newFilmedObjectDialog.trackLocation,
            newFilmedObjectDialog.representativeColor.astype(np.int), newFilmedObjectDialog.lengthSpinBox.value(),
            newFilmedObjectDialog.widthSpinBox.value(), newFilmedObjectDialog.heightSpinBox.value(),
            newFilmedObjectDialog.masksLocation, exitCode)


# In[ ]:

def findClosestPoint(points, point, threshold) :
    closePoints = np.argwhere(np.sqrt(np.sum((points-point)**2, axis=1)) < threshold)
    if len(closePoints) > 0 :
#         print np.sqrt(np.sum((points-point)**2, axis=1))[closePoints.flatten()[0]]
        return closePoints.flatten()[-1]
    else :
        return -1
    
DICT_FILMED_DATASET_BASE_LOC = 'filmed_dataset_base_location'
DICT_FIRST_EDGE_SIZE = 'intersection_rectangle_first_edge_size'
DICT_SECOND_EDGE_SIZE = 'intersection_rectangle_second_edge_size'
DICT_SCENE_ORIGIN = 'camera_scene_origin_point'
DICT_GROUND_PLANE_LINES = 'camera_parallel_lines_on_ground_plane'
DICT_EXTRUSION_HEIGHT = 'extruded_segments_world_height'
class FilmedDataset() :
    def __init__(self, filmedDatasetLoc) :
        ## bookkeeping
        self.isEmptyDataset = True
        self.isOriginPointUserDefined = False
        
        if os.path.isfile(filmedDatasetLoc) :
            self.loadFilmedDataset(filmedDatasetLoc)
        elif os.path.isdir(filmedDatasetLoc) :
            self.newFilmedDataset(filmedDatasetLoc)
        else :
            self.emptyDataset()
    
    def emptyDataset(self) :
        ## general
        self.isEmptyDataset = True
        self.filmedDatasetData = {}
        self.filmedSceneData = {}
        self.filmedObjectsData = {}
        
        ##images
        self.image = np.empty([0, 0], np.uint8)
        self.undistortedImage = np.empty([0, 0], np.uint8)
        
        ## intrinsics
        self.originalIntrinsics = np.eye(3, dtype=float)
        self.undistortedIntrinsics = np.eye(3, dtype=float)
        self.distortionParameter = 0.0
        self.distortionRatio = 0.0
        self.distortionCoeff = np.zeros(5, dtype=float)
        
        ## extrinsics
        self.cameraExtrinsics = np.eye(4, dtype=float)
        self.cameraLines = np.empty([0, 4], dtype=float)
        self.cameraSpaceRectangle = np.empty([0, 2], dtype=float)
        self.cameraSceneOriginPoint = np.empty([0], dtype=float)
        self.firstEdgeSize = 1.0
        self.secondEdgeSize = 1.0
        self.currentReprojectionError = 0.0
        
        ## scene
        self.cameraGroundPlaneGeometryPoints = np.empty([0, 2], dtype=float)
        self.groundPlaneGeometryPointsIndices = np.empty([0], dtype=int)
        self.extrusionHeight = 1.0
        self.segmentsToExtrude = np.empty([0], dtype=int)
        self.cameraExtrudedGeometryPoints = np.empty([0, 2], dtype=float)
        self.worldExtrudedGeometryPoints = np.empty([0, 3], dtype=float)
        self.extrudedGeometryPointsIndices = np.empty([0], dtype=int)
            
    def newFilmedDataset(self, baseLoc) :
        self.emptyDataset()
        if os.path.isdir(baseLoc) :
            self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC] = baseLoc
            if os.path.isfile(baseLoc+os.sep+"median.png") :
                self.image = np.array(Image.open(baseLoc+os.sep+"median.png"))
            else :
                frameLocs = np.sort(glob.glob(baseLoc+os.sep+"frame-*.png"))
                if len(frameLocs) > 0 :
                    self.image = np.array(Image.open(frameLocs[-1]))
            
            self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC] = baseLoc
            self.filmedSceneData[DICT_DOWNSAMPLED_FRAMES_RATE] = 4
            self.filmedSceneData[DICT_COMMENTS] = ""
            
            self.isEmptyDataset = False            
        
    def loadFilmedDataset(self, filmedDatasetLoc) :
        self.emptyDataset()
        if os.path.isfile(filmedDatasetLoc) :
            self.filmedDatasetData = np.load(filmedDatasetLoc).item()
            if DICT_FILMED_DATASET_BASE_LOC in self.filmedDatasetData.keys() :
                ## init image
                datasetName = [tmp for tmp in self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC].split(os.sep) if tmp != ""][-1]
                if os.path.isfile(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"median.png") :
                    self.image = np.array(Image.open(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"median.png"))
                else :
                    frameLocs = np.sort(glob.glob(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"frame-*.png"))
                    if len(frameLocs) > 0 :
                        self.image = np.array(Image.open(frameLocs[-1]))
                    else :
                        return
                if self.image.shape[-1] == 4 :
                    self.image = np.ascontiguousarray(self.image[:, :, :-1])
                
                ## load scene data if it exists, otherwise, just init with the default params
                if os.path.isfile(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"filmed_scene-{0}.npy".format(datasetName)) :
                    self.filmedSceneData = np.load(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"filmed_scene-{0}.npy".format(datasetName)).item()
                else :
                    self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC] = self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]
                    self.filmedSceneData[DICT_DOWNSAMPLED_FRAMES_RATE] = 4
                    self.filmedSceneData[DICT_COMMENTS] = ""
                self.isEmptyDataset = False
                    
                ## init intrinsics
                if DICT_CAMERA_INTRINSICS in self.filmedSceneData.keys() and DICT_DISTORTION_PARAMETER in self.filmedSceneData.keys() and DICT_DISTORTION_RATIO in self.filmedSceneData.keys() :
                    self.updateIntrinsics(self.filmedSceneData[DICT_CAMERA_INTRINSICS][0, 0],
                                          self.filmedSceneData[DICT_CAMERA_INTRINSICS][1, 1],
                                          self.filmedSceneData[DICT_CAMERA_INTRINSICS][0, 1],
                                          self.filmedSceneData[DICT_CAMERA_INTRINSICS][0, 2],
                                          self.filmedSceneData[DICT_CAMERA_INTRINSICS][1, 2],
                                          self.filmedSceneData[DICT_DISTORTION_PARAMETER],
                                          self.filmedSceneData[DICT_DISTORTION_RATIO])
                    
                if self.hasIntrinsics() :
                    ## init extrinsics
                    if DICT_GROUND_PLANE_LINES in self.filmedDatasetData.keys() :
                        self.cameraLines = self.filmedDatasetData[DICT_GROUND_PLANE_LINES]
                        if DICT_SCENE_ORIGIN in self.filmedDatasetData.keys() :
                            self.cameraSceneOriginPoint = self.filmedDatasetData[DICT_SCENE_ORIGIN]
                            self.isOriginPointUserDefined = True
                        else :
                            self.isOriginPointUserDefined = False

                        self.updateCameraSpaceRectangle()

                        if DICT_FIRST_EDGE_SIZE in self.filmedDatasetData.keys() :
                            self.firstEdgeSize = self.filmedDatasetData[DICT_FIRST_EDGE_SIZE]
                        if DICT_SECOND_EDGE_SIZE in self.filmedDatasetData.keys() :
                            self.secondEdgeSize = self.filmedDatasetData[DICT_SECOND_EDGE_SIZE]

                        self.updateExtrinsics(self.firstEdgeSize, self.secondEdgeSize)
                    
                if self.hasIntrinsics() and self.hasExtrinsics() :
                    ## init scene geometry
                    if DICT_GROUND_MESH_POINTS in self.filmedSceneData.keys() :
                        self.cameraGroundPlaneGeometryPoints = self.filmedSceneData[DICT_GROUND_MESH_POINTS]

                        if DICT_GROUND_MESH_SEGS_EXTRUDE in self.filmedSceneData.keys() :
                            self.segmentsToExtrude = np.array(self.filmedSceneData[DICT_GROUND_MESH_SEGS_EXTRUDE], dtype=int)
                        if DICT_EXTRUSION_HEIGHT in self.filmedDatasetData.keys() :
                            self.extrusionHeight = self.filmedDatasetData[DICT_EXTRUSION_HEIGHT]

                        self.updateSceneGeometry(self.segmentsToExtrude, self.extrusionHeight)
                
                if self.hasIntrinsics() and self.hasExtrinsics() and self.hasSceneGeometry() :
                    ## init objects
                    for objectLoc in np.sort(glob.glob(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"filmed_object-*.npy")) :
                        tmp = np.load(objectLoc).item()
                        self.filmedObjectsData[tmp[DICT_FILMED_OBJECT_NAME]] = tmp

#                 self.cameraLines = np.array([[894.222732595, 451.109855279, 496.881410009, 347.117983808],
#                                              [943.7, 400.7, 502.818572618, 312.803525516],
#                                              [245.45, 534.383333333, 750.505555556, 350.3],
#                                              [506.416666667, 621.616666666, 943.898308377, 355.533817445]], dtype=float)
#                 self.updateCameraSpaceRectangle()

#                 self.image = np.array(Image.open("/home/ilisescu/PhD/data/havana/median.png"))

#                 self.updateIntrinsics(702.736053, 702.736053, 0.0, 640.0, 360, -0.19, -0.19)
#                 self.updateExtrinsics(0.6, 1.0)

#                 self.cameraGroundPlaneGeometryPoints = np.array([[17.30645161290323, 928.4677419354839],
#                                                                  [295.85349462365593, 783.5981182795698],
#                                                                  [125.68212365591393, 345.3790322580645],
#                                                                  [230.23185483870964, 332.0322580645161],
#                                                                  [342.5672043010752, 414.33736559139777],
#                                                                  [520.5241935483871, 365.3991935483871],
#                                                                  [479.3716397849462, 344.26680107526875],
#                                                                  [544.9932795698925, 325.35887096774195],
#                                                                  [887.5604838709678, 384.307123655914],
#                                                                  [1106.6700268817206, 263.07392473118273],
#                                                                  [1364.7076612903227, 289.76747311827955],
#                                                                  [1531.5423387096776, 457.7143817204301],
#                                                                  [1648.3266129032259, 468.8366935483871],
#                                                                  [1652.7755376344087, 928.1881720430106]], dtype=float)

#                 self.updateSceneGeometry(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=int), 1.5)

    def hasIntrinsics(self) :
        return DICT_CAMERA_INTRINSICS in self.filmedSceneData.keys() and DICT_DISTORTION_PARAMETER in self.filmedSceneData.keys() and DICT_DISTORTION_RATIO in self.filmedSceneData.keys()
    
    def hasExtrinsics(self) :
        return (DICT_FIRST_EDGE_SIZE in self.filmedDatasetData.keys() and DICT_SECOND_EDGE_SIZE in self.filmedDatasetData.keys() and
                DICT_GROUND_PLANE_LINES in self.filmedDatasetData.keys() and DICT_CAMERA_EXTRINSICS in self.filmedSceneData.keys())
    
    def hasSceneGeometry(self) :
        return DICT_GROUND_MESH_POINTS in self.filmedSceneData.keys() and DICT_GROUND_MESH_SEGS_EXTRUDE in self.filmedSceneData.keys() and DICT_EXTRUSION_HEIGHT in self.filmedDatasetData.keys()
        
    def saveFilmedDatasetToDisk(self) :
        if not self.isEmptyDataset :
            datasetName = [tmp for tmp in self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC].split(os.sep) if tmp != ""][-1]
            
            np.save(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"filmed_dataset-{0}.npy".format(datasetName),
                    self.filmedDatasetData)
            np.save(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"filmed_scene-{0}.npy".format(datasetName),
                    self.filmedSceneData)
            
            for objectKey in self.filmedObjectsData.keys() :
                np.save(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+os.sep+"filmed_object-{0}.npy".format(self.filmedObjectsData[objectKey][DICT_FILMED_OBJECT_NAME]),
                        self.filmedObjectsData[objectKey])
                
            createAndSaveFilmedSceneNukeData(self.filmedSceneData)
                
    def saveIntrinsics(self) :
        self.filmedSceneData[DICT_CAMERA_INTRINSICS] = self.originalIntrinsics
        self.filmedSceneData[DICT_DISTORTION_PARAMETER] = self.distortionParameter
        self.filmedSceneData[DICT_DISTORTION_RATIO] = self.distortionRatio
        
    def saveExtrinsics(self) :
        self.filmedDatasetData[DICT_FIRST_EDGE_SIZE] = self.firstEdgeSize
        self.filmedDatasetData[DICT_SECOND_EDGE_SIZE] = self.secondEdgeSize
        self.filmedDatasetData[DICT_GROUND_PLANE_LINES] = self.cameraLines
        self.filmedSceneData[DICT_CAMERA_EXTRINSICS] = self.cameraExtrinsics
        if self.isOriginPointUserDefined :
            self.filmedDatasetData[DICT_SCENE_ORIGIN] = self.cameraSceneOriginPoint
        
    def saveSceneGeometry(self) :
        self.filmedDatasetData[DICT_EXTRUSION_HEIGHT] = self.extrusionHeight
        self.filmedSceneData[DICT_GROUND_MESH_POINTS] = self.cameraGroundPlaneGeometryPoints
        self.filmedSceneData[DICT_GROUND_MESH_SEGS_EXTRUDE] = self.segmentsToExtrude
        
    def saveSceneComments(self, sceneComments) :
        self.filmedSceneData[DICT_COMMENTS] = sceneComments
        
    def saveObjectRenderData(self, objectKey, orientationAngles, billboardHeight) :
        self.filmedObjectsData[objectKey][DICT_OBJECT_BILLBOARD_ORIENTATION] = orientationAngles
        self.filmedObjectsData[objectKey][DICT_OBJECT_BILLBOARD_SCALE] = billboardHeight
        
    def saveObjects(self, objectsLengths, objectsWidths, objectsHeights, objectsComments) :
        sortedObjectsKeys = np.sort(self.filmedObjectsData.keys())
        if (len(objectsLengths) == len(sortedObjectsKeys) and len(objectsWidths) == len(sortedObjectsKeys) and
            len(objectsHeights) == len(sortedObjectsKeys) and len(objectsComments) == len(sortedObjectsKeys)) :
            for idx, (objectLength, objectWidth, objectHeight, comment) in enumerate(zip(objectsLengths, objectsWidths, objectsHeights, objectsComments)) :
                self.filmedObjectsData[sortedObjectsKeys[idx]][DICT_OBJECT_LENGTH] = objectLength
                self.filmedObjectsData[sortedObjectsKeys[idx]][DICT_OBJECT_WIDTH] = objectWidth
                self.filmedObjectsData[sortedObjectsKeys[idx]][DICT_OBJECT_HEIGHT] = objectHeight
                self.filmedObjectsData[sortedObjectsKeys[idx]][DICT_COMMENTS] = comment
            return True
        else :
            return False
    
    def updateIntrinsics(self, fx, fy, s, x0, y0, distortionParameter, distortionRatio) :
        if not self.isEmptyDataset :
            self.originalIntrinsics = np.array([[fx, s, x0],
                                                [0, fy, y0],
                                                [0, 0, 1.0]])

            self.distortionParameter = distortionParameter
            self.distortionRatio = distortionRatio

            if len(self.image) != 0 :
                self.undistortedImage, self.undistortedIntrinsics, self.distortionCoeff = undistortImage(self.distortionParameter, self.distortionRatio, self.image, self.originalIntrinsics)        
            
    def getWorldSpaceRectangle(self) :
        return np.array([[-self.secondEdgeSize/2.0, -self.secondEdgeSize/2.0, self.secondEdgeSize/2.0, self.secondEdgeSize/2.0],
                         [-self.firstEdgeSize/2.0, self.firstEdgeSize/2.0, self.firstEdgeSize/2.0, -self.firstEdgeSize/2.0],
                         [0.0, 0.0, 0.0, 0.0]]).T
            
        
    def updateExtrinsics(self, firstEdgeSize, secondEdgeSize) :
        if not self.isEmptyDataset :
            startTime = time.time()
            self.firstEdgeSize = firstEdgeSize
            self.secondEdgeSize = secondEdgeSize

            if len(self.cameraSpaceRectangle) == 4 :
                ## compute the camera extrinsics using PnP
                points3D = self.getWorldSpaceRectangle()
                objectPoints = points3D.astype(np.float64).reshape([len(points3D), 1, 3])
                imagePoints = self.cameraSpaceRectangle.astype(np.float64).reshape([len(self.cameraSpaceRectangle), 1, 2])
                cameraMatrix = self.undistortedIntrinsics.astype(np.float64)
                distCoeffs = np.zeros([5, 1], dtype=np.float64)

    #             _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv2.CV_P3P)
                _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv2.CV_ITERATIVE)
                rotMat, _ = cv2.Rodrigues(rvec)

                self.cameraExtrinsics = np.vstack([np.concatenate([rotMat, tvec], axis=1), np.array([[0, 0, 0, 1]])])

                self.currentReprojectionError = np.sum(np.sqrt(np.sum((imagePoints[:, 0, :].T-cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)[0][:, 0, :].T)**2, axis=0)))
    #             print "ITER", currentError,

                ## compute the projection of the 3d points moved so that their center is at the user defined center
                worldSceneOriginPoint = screenToWorldPlane(self.cameraSceneOriginPoint[np.newaxis, :], self.undistortedIntrinsics, self.cameraExtrinsics)
                imagePoints = cv2.projectPoints(objectPoints+worldSceneOriginPoint.reshape([1, 1, 3]), rvec, tvec, cameraMatrix, distCoeffs)[0]

                ## recompute the extrinsics
                _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=cv2.CV_ITERATIVE)
                rotMat, _ = cv2.Rodrigues(rvec)

                self.cameraExtrinsics = np.vstack([np.hstack([rotMat, tvec]), np.array([[0, 0, 0, 1]])])

    #             print time.time()-startTime
        
    def updateCameraSpaceRectangle(self) :
        if len(self.cameraLines) == 4 :
            intersectionPoints = []
            for line1 in self.cameraLines[0:2, :] :
                for line2 in self.cameraLines[2:, :] :
                    intersectionPoints.append(line2lineIntersection(line1, line2))

            self.cameraSpaceRectangle = np.array(intersectionPoints)[[0, 2, 3, 1], :]
            
            if not self.isOriginPointUserDefined :
                try :
                    self.cameraSceneOriginPoint = line2lineIntersection(np.concatenate(self.cameraSpaceRectangle[[0, 2], :]),
                                                                        np.concatenate(self.cameraSpaceRectangle[[1, 3], :]))
                except Exception:
                    pass
        else :
            self.cameraSpaceRectangle = np.empty([0, 2], dtype=float)
            if not self.isOriginPointUserDefined :
                self.cameraSceneOriginPoint = np.empty([0], dtype=float)
        
    def moveCameraLinesPoint(self, pointToMove, deltaX, deltaY, minX, minY, maxX, maxY) :
        pointRow = int(np.floor(pointToMove/2.0))
        pointCol = np.mod(pointToMove, 2)*2
        if pointRow >= 0 and pointRow < len(self.cameraLines) and pointCol >=0 and pointCol < 4 :
            self.cameraLines[pointRow, pointCol] = np.min([np.max([self.cameraLines[pointRow, pointCol]+deltaX, minX]), maxX])
            self.cameraLines[pointRow, pointCol+1] = np.min([np.max([self.cameraLines[pointRow, pointCol+1]+deltaY, minY]), maxY])

            self.updateCameraSpaceRectangle()

            return self.cameraLines[pointRow, pointCol:pointCol+2]
        else :
            return np.empty([0], dtype=float)
    
    def moveCameraSceneOriginPoint(self, deltaX, deltaY, minX, minY, maxX, maxY) :
        self.cameraSceneOriginPoint[0] = np.min([np.max([self.cameraSceneOriginPoint[0]+deltaX, minX]), maxX])
        self.cameraSceneOriginPoint[1] = np.min([np.max([self.cameraSceneOriginPoint[1]+deltaY, minY]), maxY])

        return self.cameraSceneOriginPoint
    
    def updateSceneGeometry(self, segmentsToExtrude, extrusionHeight) :
        if not self.isEmptyDataset :
            self.segmentsToExtrude = segmentsToExtrude
            self.extrusionHeight = extrusionHeight
            if len(self.cameraGroundPlaneGeometryPoints) > 0 :
                viewMat, projectionMat = cvCameraToOpenGL(self.cameraExtrinsics, self.undistortedIntrinsics, self.undistortedImage.shape[0:2])

                worldGroundPlaneGeometryPoints = screenToWorldPlane(self.cameraGroundPlaneGeometryPoints, self.undistortedIntrinsics, self.cameraExtrinsics)
                self.groundPlaneGeometryPointsIndices = np.array(triangulate2DPolygon(worldGroundPlaneGeometryPoints[:, :-1])).flatten()

                self.cameraExtrudedGeometryPoints = np.empty([0, 2], dtype=float)
                self.worldExtrudedGeometryPoints = np.empty([0, 3], dtype=float)
                self.extrudedGeometryPointsIndices = np.empty([0], dtype=int)
                if len(self.segmentsToExtrude) > 0 :
                    for segmentIndices in zip(self.segmentsToExtrude, np.mod(self.segmentsToExtrude+1, len(worldGroundPlaneGeometryPoints))) :
                        newVertices, newIndices = extrudeSegment(worldGroundPlaneGeometryPoints[segmentIndices, :], self.extrusionHeight, np.linalg.inv(self.cameraExtrinsics)[:-1, -1], False)
                        self.extrudedGeometryPointsIndices = np.concatenate([self.extrudedGeometryPointsIndices, np.array(newIndices, dtype=int)+len(self.cameraExtrudedGeometryPoints)])
                        self.cameraExtrudedGeometryPoints = np.vstack([self.cameraExtrudedGeometryPoints, 
                                                                       worldToScreenSpace(viewMat, projectionMat, newVertices, self.undistortedImage.shape[1], self.undistortedImage.shape[0])])
                        self.worldExtrudedGeometryPoints = np.vstack([self.worldExtrudedGeometryPoints, newVertices])
        
        
    def moveCameraGroundPlaneGeometryPoints(self, pointToMove, deltaX, deltaY, minX, minY, maxX, maxY) :
        if pointToMove >= 0 and pointToMove < len(self.cameraGroundPlaneGeometryPoints) :
            self.cameraGroundPlaneGeometryPoints[pointToMove, 0] = np.min([np.max([self.cameraGroundPlaneGeometryPoints[pointToMove, 0]+deltaX, minX]), maxX])
            self.cameraGroundPlaneGeometryPoints[pointToMove, 1] = np.min([np.max([self.cameraGroundPlaneGeometryPoints[pointToMove, 1]+deltaY, minY]), maxY])

            return self.cameraGroundPlaneGeometryPoints[pointToMove, :]
        else :
            return np.empty([0], dtype=float)
        
    def doAddMeshPoint(self) :
        self.cameraGroundPlaneGeometryPoints = np.vstack([self.cameraGroundPlaneGeometryPoints, np.array([self.undistortedImage.shape[1], self.undistortedImage.shape[0]], dtype=float)/2.0])
        
    def doDeleteMeshPoint(self, pointIdx) :
        if pointIdx >= 0 and pointIdx < len(self.cameraGroundPlaneGeometryPoints) :
            self.cameraGroundPlaneGeometryPoints = np.delete(self.cameraGroundPlaneGeometryPoints, pointIdx, 0)
    
    def doDeleteFilmedObjects(self, objectsName) :
        for objectName in objectsName :
            if objectName not in self.filmedObjectsData.keys() :
                raise Exception("Bad Delete Object!")
            else :
                os.remove(self.filmedObjectsData[objectName][DICT_PATCHES_LOCATION])
                os.remove(self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"filmed_object-{0}.npy".format(self.filmedObjectsData[objectName][DICT_FILMED_OBJECT_NAME]))
                del self.filmedObjectsData[objectName]
    
    def addNewFilmedObject(self, objectName, trackLocation, representativeColor, length, width, height, masksLocation) :
        if self.hasIntrinsics() and self.hasExtrinsics() :
            trajectoryPoints, sortedFrameKeys = readNukeTrack(trackLocation)
        
            if len(sortedFrameKeys) > 0 :
                self.filmedObjectsData[objectName] = {DICT_FILMED_OBJECT_NAME : objectName,
                                                      DICT_TRACK_LOCATION : trackLocation,
                                                      DICT_TRAJECTORY_POINTS : trajectoryPoints,
                                                      DICT_NEEDS_UNDISTORT : False,
                                                      DICT_CAMERA_INTRINSICS : self.originalIntrinsics,
                                                      DICT_PATCHES_LOCATION : self.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"camera_adjusted-homography_billboard-preloaded_patches-{0}.npy".format(objectName),
                                                      DICT_REPRESENTATIVE_COLOR : representativeColor,
                                                      DICT_OBJECT_LENGTH : length, 
                                                      DICT_OBJECT_WIDTH : width,
                                                      DICT_OBJECT_HEIGHT : height,
                                                      DICT_MASK_LOCATION : masksLocation
                                                     }

            return sortedFrameKeys
        else :
            return np.empty([0], dtype=int)
        
def readNukeTrack(trackLocation) :
    """returns trajectoryPoints, sortedFrameKeys"""
    f = open(trackLocation, 'r')
    lines = f.readlines()
    try :
        vals = [np.array(i.split(" ")).astype(float) for i in lines]
    except ValueError :
        return np.empty([0], dtype=int), np.empty([0, 2], dtype=float)
    if np.array(vals).shape[1] != 3 :
        return np.empty([0], dtype=int), np.empty([0, 2], dtype=float)

    vals = [(int(i[-1]), i[0:2]) for i in vals]
    tmp = dict(vals)
    sortedFrameKeys = np.sort(tmp.keys())
    trajectoryPoints = np.array([tmp[key] for key in sortedFrameKeys])
    
    return trajectoryPoints, sortedFrameKeys


# In[ ]:

def searchWorldSpacePoint(viewMat, projectionMat, worldTargetPoint, worldStartPoint, cameraStartPoint, worldDir, width, height, verbose=False) :
    cameraClosestPointToIntersection = np.zeros([1, 2])
    worldClosestPointToIntersection = np.zeros([1, 3])
    ## init ##
    pointDist = 1600.0
    dirRatio = 1.0
    increment = np.copy(dirRatio)
    worldCurrentPoint = worldStartPoint+worldDir*dirRatio
    foundInside = False
    iterNum = 0 
    while pointDist > 0.1 and iterNum < 100 :
        iterNum += 1
        cameraCurrentPoint = worldToScreenSpace(viewMat, projectionMat, worldCurrentPoint, width, height)
        cameraClosestPointToIntersection[0, :] = cameraCurrentPoint
        worldClosestPointToIntersection[0, :] = worldCurrentPoint

        dotProduct = np.dot(cameraCurrentPoint-cameraStartPoint, worldTargetPoint-cameraStartPoint)
        squaredDist = np.linalg.norm(cameraCurrentPoint-cameraStartPoint)**2

        if verbose :
            print(dotProduct, squaredDist, np.linalg.norm(cameraCurrentPoint-worldTargetPoint))
        ## flip dirRatio direction if the worldTargetPoint is outside of the line segment (cameraCurrentPoint-cameraStartPoint) in the direction of cameraStartPoint
        if dotProduct < 0 :
            if verbose :
                print("FLIPPING")
            dirRatio *= -1
            increment *= -1
            worldCurrentPoint = worldStartPoint+worldDir*dirRatio
            continue

        ## if worldTargetPoint is within the line segment then set the increment to half the current length and set currentPoint to the middle of the half segment closest to cameraStartPoint
        if dotProduct < squaredDist :
            increment *= 0.5
            foundInside = True
            dirRatio -= increment
        ## if the worldTargetPoint is outside the line segment
        else :
            ## set the increment to half the current length only if the worldTargetPoint has been within the line segment (otherwise don't split but keep increasing the length of the line segment I'm looking within)
            if foundInside :
                increment *= 0.5
            ## if foundInside == True this sets currentPoint to the middle of the half segment furthest from cameraStartPoint, otherwise it doubles the length of the line segment
            dirRatio += increment

        if verbose :
            print("DIR RATIO", dirRatio, increment, foundInside)

        worldCurrentPoint = worldStartPoint+worldDir*dirRatio

        pointDist = np.linalg.norm(cameraCurrentPoint-worldTargetPoint)
    if iterNum >= 100 :
        print("...REACHED MAXIMUM ITER COUNT")
#     else :
#         print("...DONE")

    return worldClosestPointToIntersection, cameraClosestPointToIntersection

def findBillboardSize(worldPos, worldOrientDir, worldUpDir, projectionMat, viewMat, worldToCameraHomography, patchData, width, height, verbose=False, doReturnExtraInfo=False) :    
    worldOrientDirPos = worldPos + worldOrientDir

    ## find projections of world coords into camera space
    cameraPos = worldToScreenSpace(viewMat, projectionMat, worldPos, width, height)
    cameraOrientDirPos = worldToScreenSpace(viewMat, projectionMat, worldOrientDirPos, width, height)
    
    worldUpDirPos = worldPos+worldUpDir
    cameraUpDirPos = worldToScreenSpace(viewMat, projectionMat, worldUpDirPos, width, height)
    
    ########### FIND BILLBOARD WIDTH BASED ON HOW IT PROJECTS INTO SCREEN SPACE AND HOW IT THEN RELATES WITH THE SEGMENTED PATCH ###########
    cameraDirLeftIntersection = line2lineIntersection(np.array([cameraPos, cameraOrientDirPos]).flatten(),
                                                      np.array([patchData['top_left_pos'][::-1], np.array([patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][1]])[::-1]]).flatten())

    cameraDirRightIntersection = line2lineIntersection(np.array([cameraPos, cameraOrientDirPos]).flatten(),
                                                       np.array([(patchData['top_left_pos']+patchData['patch_size'])[::-1],
                                                                 np.array([patchData['top_left_pos'][0], patchData['top_left_pos'][1]+patchData['patch_size'][1]])[::-1]]).flatten())

    worldDirLeftIntersection = np.dot(np.linalg.inv(worldToCameraHomography), np.concatenate([cameraDirLeftIntersection, [1]]).reshape([3, 1])).flatten()
    worldDirLeftIntersection /= worldDirLeftIntersection[-1]
    worldDirLeftIntersection[-1] = 0

    worldDirRightIntersection = np.dot(np.linalg.inv(worldToCameraHomography), np.concatenate([cameraDirRightIntersection, [1]]).reshape([3, 1])).flatten()
    worldDirRightIntersection /= worldDirRightIntersection[-1]
    worldDirRightIntersection[-1] = 0

    billboardWidth = np.max([np.linalg.norm(worldPos-worldDirLeftIntersection), np.linalg.norm(worldPos-worldDirRightIntersection)])*2

    ########### FIND BILLBOARD HEIGHT IN A SIMILAR MANNER TO ITS WIDTH ###########
    cameraUpDirTopIntersection = line2lineIntersection(np.array([cameraPos, cameraUpDirPos]).flatten(),
                                                       np.array([patchData['top_left_pos'][::-1], np.array([patchData['top_left_pos'][0], patchData['top_left_pos'][1]+patchData['patch_size'][1]])[::-1]]).flatten())

    cameraUpDirBottomIntersection = line2lineIntersection(np.array([cameraPos, cameraUpDirPos]).flatten(),
                                                          np.array([(patchData['top_left_pos']+patchData['patch_size'])[::-1],
                                                                    np.array([patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][1]])[::-1]]).flatten())
    ## to find the height I can't do the same thing as I did for the width because I can't project screen space points back into the world (previously I could because I knew the points were on the ground plane)
    ## instead, do a binary search type thing along the worldUpDir to find the world space points that project closest to the screen space points found above (i.e cameraUpDirTopIntersection and cameraUpDirBottomIntersection)

    worldClosestPointsToIntersection = np.empty([0, 3])
    cameraClosestPointsToIntersection = np.empty([0, 2])
    for i, point in enumerate([cameraUpDirTopIntersection, cameraUpDirBottomIntersection]) :
        worldClosestPointToIntersection, cameraClosestPointToIntersection = searchWorldSpacePoint(viewMat, projectionMat, point, worldPos, cameraPos, worldUpDir, width, height, verbose)
        worldClosestPointsToIntersection = np.concatenate([worldClosestPointsToIntersection, worldClosestPointToIntersection], axis=0)
        cameraClosestPointsToIntersection = np.concatenate([cameraClosestPointsToIntersection, cameraClosestPointToIntersection], axis=0)


    billboardHeight = np.max([np.linalg.norm(worldPos-worldClosestPointsToIntersection[0, :]), np.linalg.norm(worldPos-worldClosestPointsToIntersection[1, :])])*2
    
    if doReturnExtraInfo :
        return (billboardWidth, billboardHeight, cameraPos, cameraOrientDirPos, cameraUpDirPos, cameraDirLeftIntersection,
                cameraDirRightIntersection, cameraUpDirTopIntersection, cameraUpDirBottomIntersection, cameraClosestPointsToIntersection)
    else :
        return billboardWidth, billboardHeight
    
    
def getPatchDataWithThresholdedAlpha(patchImage, patchAlpha, bgImage, threshold=0.8) :
    """patchImage.shape = [H, W, 3], patchAlpha.shape = [H, W]"""
    
    ## threshold the alpha based on bg diff 
    if True :
        visiblePixels = np.argwhere(patchAlpha != 0)
        if len(visiblePixels > 0) :
            diffs = np.sqrt(np.sum((patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :] - bgImage[visiblePixels[:, 0], visiblePixels[:, 1], :])**2, axis=1))
            tmp = np.zeros([bgImage.shape[0], bgImage.shape[1]], np.uint8)
            tmp[visiblePixels[:, 0], visiblePixels[:, 1]] = diffs
            tmp = cv2.medianBlur(tmp, 7).astype(float)
            if float(np.max(tmp)) != 0.0 :
                tmp /= float(np.max(tmp))
            med = np.median(tmp[visiblePixels[:, 0], visiblePixels[:, 1]])
            tmp[tmp<med*threshold] = 0
            tmp[tmp>0] = np.max(tmp)
            if float(np.max(tmp)) != 0.0 :
                tmp /= float(np.max(tmp))
            patchAlpha = (tmp*255).astype(np.uint8)
        

    visiblePixels = np.argwhere(patchAlpha != 0)
    if len(visiblePixels) > 0 :
        topLeft = np.min(visiblePixels, axis=0)
        patchSize = np.max(visiblePixels, axis=0) - topLeft + 1
        colors = np.concatenate([patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)
    else :
        topLeft = np.zeros(2)
        patchSize = np.zeros(2)
        colors = np.empty([0, 4], dtype=np.uint8)

    patchData = {'top_left_pos':topLeft, 'sprite_colors':colors[:, [2, 1, 0, 3]],
                 'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}
    
    return patchAlpha, patchData


# In[15]:

def getDirectionsFromTrajectoryPoints(trajectoryPoints) :
    trajectoryDirections = np.array([trajectoryPoints[i, :]-trajectoryPoints[j, :] for i, j in zip(np.arange(1, len(trajectoryPoints)), np.arange(0, len(trajectoryPoints)-1))])
    
    trajectoryDirections = np.vstack([trajectoryDirections, trajectoryDirections[-1, :][np.newaxis, :]])
    trajectoryDirections /= np.linalg.norm(trajectoryDirections, axis=1)[:, np.newaxis]
    for i in xrange(len(trajectoryDirections)) :
        if np.linalg.norm(trajectoryDirections[i, :]) != 1.0 and i > 0 :
            trajectoryDirections[i, :] = trajectoryDirections[i-1, :]
            
    return trajectoryDirections

def findEncompassingBillboardSize(patchImage, patchAlpha, patchData, worldPos, worldMovingDir, cameraExtrinsics, cameraIntrinsics, imageShape, verbose=False) :
    """find the billboard size that when placed and oriented given the worldTrajectoryPoints, it always encompasses the segmented patches when projected into the view"""
    height, width = imageShape
    viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, imageShape)
    
    
    worldMoveNormalDir = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([0.0, 0.0, 1.0]))), np.concatenate([worldMovingDir, [1]]))
    worldMoveNormalDir = worldMoveNormalDir[:-1]/np.linalg.norm(worldMoveNormalDir[:-1])
    currentBillboardArea = 10000000000.0
    
    ## sensible init values
    billboardWidth, billboardHeight, worldOrientDir, dirMultiplier = 0.0, 0.0, worldMovingDir, 1.0
    
    for currentWorldOrientDir in [worldMoveNormalDir, worldMovingDir] :
        ## check if the projection of the orientation direction into image space has a positive x coordinate; if it doesn't, I need to flip the orientation direction, which will produce the same exact billboard, but
        ## it will ensure it projects into image space with the front face visible
        currentDirMultiplier = 1.0
        if (worldToScreenSpace(viewMat, projectionMat, worldPos + currentWorldOrientDir, width, height)-worldToScreenSpace(viewMat, projectionMat, worldPos, width, height))[0] < 0 :
            currentWorldOrientDir *= -1.0
            currentDirMultiplier = -1.0
        worldUpDir = np.array([0.0, 0.0, 1.0])
        
        try :
            currentBillboardWidth, currentBillboardHeight = findBillboardSize(worldPos, currentWorldOrientDir, worldUpDir, projectionMat, viewMat,
                                                                              np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]), patchData,
                                                                              width, height, verbose)
        except RuntimeError as e:
            print "RuntimeError : {0} --- {1}".format(e, len(patchData['sprite_colors']))
            continue

        if currentBillboardWidth*currentBillboardHeight < currentBillboardArea :
#             print("NEW AREA", currentBillboardWidth*currentBillboardHeight, currentBillboardWidth, currentBillboardHeight, currentWorldOrientDir)
            currentBillboardArea = currentBillboardWidth*currentBillboardHeight
            billboardWidth = currentBillboardWidth
            billboardHeight = currentBillboardHeight
            worldOrientDir = currentWorldOrientDir
        ## I'm really only interested in saving the dirMultiplier for the tangent direction, worldMovingDir
        dirMultiplier = currentDirMultiplier
            
    return billboardWidth, billboardHeight, [worldOrientDir, billboardWidth, billboardHeight, dirMultiplier]


# In[16]:

def getBillboardVertices(aspectRatio, scale) :
    return np.dot(np.array([[scale, 0, 0],
                            [0, scale, 0],
                            [0, 0, 1.0]], float),
                  np.array([[aspectRatio/2.0, -0.5, 0.0], [-aspectRatio/2.0, -0.5, 0.0], [-aspectRatio/2.0, 0.5, 0.0], [aspectRatio/2.0, 0.5, 0.0]], float).T).T

def undistortPatch(patchImage, patchAlpha, patchData, worldPos, orientationAndSize, maxBillboardWidth, maxBillboardHeight, cameraExtrinsics, cameraIntrinsics,
                   bgImage, worldTrajectoryPoints, verbose=False, doVisualize=False) :
    """undistorts a patch so that when it's placed on a billboard at the right place on the object's trajectory, it projects into image space properly"""
    
    height, width = bgImage.shape[:2]
    viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, bgImage.shape)
    worldOrientDir, billboardWidth, billboardHeight, dirMultiplier = orientationAndSize


    ########### ROTATE BILLBOARD TO ALIGN WITH MOVING DIRECTION AND PLACE AT POINT ON TRAJECTORY ###########
    billboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2.0, np.array([1.0, 0.0, 0.0]))) ## rotate billboard ccw along x axis to put it up
    adjustAngle = np.arccos(np.clip(np.dot(np.array([1.0, 0.0, 0.0]), worldOrientDir), -1, 1))
    adjustAxis = np.cross(worldOrientDir, np.array([1.0, 0.0, 0.0]))
    adjustAxis /= np.linalg.norm(adjustAxis)
    billboardModelMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), billboardModelMat)
    billboardModelMat[:-1, -1] = worldPos

    worldBillboardVertices = getBillboardVertices(billboardWidth/billboardHeight, billboardHeight)
    worldBillboardVertices = np.dot(billboardModelMat, np.concatenate([worldBillboardVertices, np.ones([len(worldBillboardVertices), 1])], axis=1).T).T[:, :-1]
    screenBillboardVertices = worldToScreenSpace(viewMat, projectionMat, worldBillboardVertices, width, height)

    footprintBillboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
    footprintBillboardModelMat[:-1, -1] = worldPos
    ## HACK : this is for the havana cars
    worldFootprintBillboardVertices = getBillboardVertices(0.18/0.081, 0.265)
    worldFootprintBillboardVertices = np.dot(footprintBillboardModelMat, np.concatenate([worldFootprintBillboardVertices, np.ones([len(worldFootprintBillboardVertices), 1])], axis=1).T).T[:, :-1]
    screenFootprintBillboardVertices = worldToScreenSpace(viewMat, projectionMat, worldFootprintBillboardVertices, width, height)

    texHeight = 0
    texWidth = 0
    if maxBillboardWidth > maxBillboardHeight :
        texWidth = 512
        texHeight = texWidth*maxBillboardHeight/maxBillboardWidth
    else :
        texHeight = 512
        texWidth = texHeight*maxBillboardWidth/maxBillboardHeight

    ######################### SCALE COMPENSATION BASED ON BILLBOARD WIDTH IN WORLD SPACE #########################
    widthScale = billboardWidth/maxBillboardWidth
    heightScale = billboardHeight/maxBillboardHeight
    scaledTexHeight = texHeight*heightScale
    scaledTexWidth = scaledTexHeight*billboardWidth/billboardHeight
    if verbose :
        print scaledTexWidth, scaledTexHeight

    ## when defining the rectangle in texture space, need to make sure that it uses the same conventions as screenBillboardVertices, which in this case, it means y goes down and x goes left
    billboardHomography = cv2.findHomography(screenBillboardVertices, np.array([[texWidth-(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0],
                                                                                [(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0],
                                                                                [(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0],
                                                                                [texWidth-(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0]], dtype=float))[0]

    tmp = cv2.warpPerspective(np.concatenate([patchImage, patchAlpha.reshape([patchAlpha.shape[0], patchAlpha.shape[1], 1])], axis=-1), billboardHomography, (int(np.ceil(texWidth)), int(np.ceil(texHeight))))

    visiblePixels = np.argwhere(tmp[:, :, -1] != 0)

    colors = np.concatenate([tmp[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)
    
    if doVisualize :
        ### VISUALIZE BILLBOARD AND STUFF ###
        (billboardWidth, billboardHeight, cameraPos, cameraOrientDirPos, cameraUpDirPos, cameraDirLeftIntersection,
         cameraDirRightIntersection, cameraUpDirTopIntersection, cameraUpDirBottomIntersection,
         cameraClosestPointsToIntersection) = findBillboardSize(worldPos, worldOrientDir, np.array([0.0, 0.0, 1.0]), projectionMat,
                                                                viewMat, np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]), patchData, width, height, verbose, True)
        compositedImage = np.copy(bgImage)
        patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
        patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]

        compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
                        patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :] = (compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
                                                                                                                                    patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :]*
                                                                                                                   (1.0-patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]) +
                                                                                                                   patchColors[:, :, :-1]*(patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]))


        figure(); imshow(compositedImage)
        xlim([0, bgImage.shape[1]])
        ylim([bgImage.shape[0], 0])
        # xlim([(bgImage.shape[1]-1280)/2, 1280+(bgImage.shape[1]-1280)/2])
        # ylim([720+(bgImage.shape[0]-720)/2, (bgImage.shape[0]-720)/2])

        cameraTrajectoryPoints = worldToScreenSpace(viewMat, projectionMat, worldTrajectoryPoints, width, height)

        #     scatter(cameraTrajectoryPoints[:, 0], cameraTrajectoryPoints[:, 1], color=tuple(trajectory.drawColor/255.0), marker='o', facecolors='none', s=90)
        scatter(cameraTrajectoryPoints[:, 0], cameraTrajectoryPoints[:, 1], color='red', marker='x', s=90)
        plot([cameraPos[0], cameraOrientDirPos[0]], [cameraPos[1], cameraOrientDirPos[1]], color='yellow', linewidth=2)
        plot([cameraPos[0], cameraUpDirPos[0]], [cameraPos[1], cameraUpDirPos[1]], color='yellow', linewidth=2)
        plot([patchData['top_left_pos'][1], patchData['top_left_pos'][1], patchData['top_left_pos'][1]+patchData['patch_size'][1],
              patchData['top_left_pos'][1]+patchData['patch_size'][1], patchData['top_left_pos'][1]], [patchData['top_left_pos'][0], patchData['top_left_pos'][0]+patchData['patch_size'][0],
                                                                                                       patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][0],
                                                                                                       patchData['top_left_pos'][0]], color='red', linewidth=2)

        scatter([cameraDirLeftIntersection[0], cameraDirRightIntersection[0], cameraUpDirTopIntersection[0], cameraUpDirBottomIntersection[0]],
                [cameraDirLeftIntersection[1], cameraDirRightIntersection[1], cameraUpDirTopIntersection[1], cameraUpDirBottomIntersection[1]], color='blue', s=90)
        scatter([cameraClosestPointsToIntersection[:, 0]], [cameraClosestPointsToIntersection[:, 1]], color="yellow", marker="x", s=90)
        plot(screenBillboardVertices[[0, 3, 2, 1, 0], 0], screenBillboardVertices[[0, 3, 2, 1, 0], 1], color='magenta', linewidth=2)
        plot(screenFootprintBillboardVertices[[0, 3, 2, 1, 0], 0], screenFootprintBillboardVertices[[0, 3, 2, 1, 0], 1], color='cyan', linewidth=2)
    
    return {'top_left_pos':np.zeros(2, int), 'sprite_colors':colors[:, [2, 1, 0, 3]], 'visible_indices': visiblePixels, 'patch_size': np.array([int(np.ceil(texHeight)), int(np.ceil(texWidth))], int)}


# In[18]:

def getUndistortedTrajectoryPoints(filmedObjectData, filmedSceneData, undistortedIntrinsics) :
    trajectoryPoints = filmedObjectData[DICT_TRAJECTORY_POINTS]
    if filmedObjectData[DICT_NEEDS_UNDISTORT] :
        ## for the trajectory points to be valid, I need to undistort them (as I'm working in the undistorted space and the objects were tracked in the original image space) 
        trajectoryPoints = cv2.undistortPoints(trajectoryPoints.reshape((1, len(trajectoryPoints), 2)), filmedObjectData[DICT_CAMERA_INTRINSICS],
                                               getDistortionCoeffFromParamAndRatio(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO]),
                                               P=undistortedIntrinsics)[0, :, :]
    else :
        ## or in the case of nuke tracks they were tracked in the cropped undistorted image so I need to account for difference in camera intrinsics
        ## THE ABOVE IS NOT TRUE ANYMORE
        ## trajectoryPoints = trajectoryPoints + filmedDataset.undistortedIntrinsics[:2, -1] - filmedObjectData[DICT_CAMERA_INTRINSICS][:2, -1]
        pass
        
    return trajectoryPoints


# In[19]:

class HorizontalLine(QtGui.QFrame) :
    def __init__(self):
        super(HorizontalLine, self).__init__()
        
        self.setFrameStyle(QtGui.QFrame.HLine)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)
        self.setStyleSheet("QFrame{color: rgba(0, 0, 0, 127);}")

CREATION_STAGE_INTRINSICS = 0
CREATION_STAGE_EXTRINSICS = 1
CREATION_STAGE_SCENE = 2
CREATION_STAGE_OBJECTS = 3
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.currentCreationStage = -1
        self.datasetReady = False
        self.operationThreads = []
        self.unsavedIntrinsics = False
        self.unsavedExtrinsics = False
        self.unsavedSceneGeometry = False
        self.unsavedObjects = False
        self.unsavedToDisk = False
        
        frustumScale = 0.15
        frustumAspectRatio = 1280.0/720.0
        frustumZDir = 0.2
        self.frustumPoints = np.dot(np.array([[frustumScale, 0, 0],
                                              [0, frustumScale, 0],
                                              [0, 0, 1.0]], float),
                                    np.array([[-frustumAspectRatio/2.0, -0.5, frustumZDir], [-frustumAspectRatio/2.0, 0.5, frustumZDir],
                                              [frustumAspectRatio/2.0, 0.5, frustumZDir], [frustumAspectRatio/2.0, -0.5, frustumZDir], [0.0, 0.0, 0.0],
                                              [0.0, 0.0, frustumZDir], [0.0, -1.0, 0.0]], float).T).T
        self.frustumPoints = self.frustumPoints[[0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 4, 2, 4, 3, 4, 4, 5, 4, 6], :]
        self.resetScene3DData()
        
        self.filmedObjectsSortedFrameKeys = {}
        self.filmedObjectsPatches = {}
        
        self.createGUI()
        self.setCurrentCreationStage(CREATION_STAGE_INTRINSICS)
        
        self.setWindowTitle("Filmed Scene Creator")
        self.resize(1850, 720)
        
        ## UI bookeeping
        self.pointToMove = -1
        self.doMovePoint = False
        self.prevMousePosition = QtCore.QPointF(0, 0)
        
        self.setFocus()
        
        self.filmedDataset = FilmedDataset("/home/ilisescu/PhD/data/havana/filmed_dataset-havana.npy")
        self.datasetReady = True
        self.updateUIForDataset()
        
    def resetScene3DData(self) :
        self.scene3DGroundPlaneCameraFrustumPoints = np.empty([0, 2], dtype=float)
        self.scene3DGroundPlaneImage = np.empty([0, 0], dtype=np.uint8)
        self.scene3DGroundPlaneViewMat = np.eye(4, dtype=float)
        self.scene3DGroundPlaneProjectionMat = np.eye(4, dtype=float)
        self.scene3DGroundPlaneCameraLines = np.empty([0, 4], dtype=float)
        self.scene3DGroundPlaneCameraRectanglePoints = np.empty([0, 2], dtype=float)
        self.scene3DGroundPlaneCameraSceneOriginPoint = np.empty([0], dtype=float)
        
        self.scene3DGeometryCameraFrustumPoints = np.empty([0, 2], dtype=float)
        self.scene3DGeometryImage = np.empty([0, 0], dtype=np.uint8)
        self.scene3DGeometryViewMat = np.eye(4, dtype=float)
        self.scene3DGeometryProjectionMat = np.eye(4, dtype=float)
        self.scene3DGeometryCameraGroundPlaneGeometryPoints = np.empty([0, 2], dtype=float)
        self.scene3DGeometryCameraExtrudedGeometryPoints = np.empty([0, 2], dtype=float)
        
    
    def loadFilmedDatasetButtonPressed(self) :
        if self.changesDealtWith("Are you sure you want to load a new dataset?") :
            filmedDatasetLoc = QtGui.QFileDialog.getOpenFileName(self, "Load Filmed Dataset", os.path.expanduser("~")+"/PhD/data/", "Filmed Datasets (filmed_dataset-*.npy)")[0]
            if filmedDatasetLoc != "" :
                
                ## the method connected to the done signal of each thread should deal with cleaning up
                for operationThread in self.operationThreads :
                    operationThread.doQuit()
                    
                self.datasetReady = False
                self.updateUIForDataset()

                del self.filmedDataset
                self.filmedDataset = FilmedDataset(filmedDatasetLoc)
                self.datasetReady = not self.filmedDataset.isEmptyDataset

                if not self.datasetReady :
                    QtGui.QMessageBox.warning(self, "Invalid Dataset", "<center>The selected dataset is invalid.\nIf you need more details, complain to the dev.</center>")

                self.updateUIForDataset()

                if self.currentCreationStage != CREATION_STAGE_INTRINSICS :
                    self.tabWidget.setCurrentIndex(0)
                else :
                    self.setCurrentCreationStage(-1)
                    self.setCurrentCreationStage(CREATION_STAGE_INTRINSICS)

        self.setFocus()
    
    def newFilmedDatasetButtonPressed(self) :
        filmedDatasetBaseLoc = QtGui.QFileDialog.getExistingDirectory(self, "New Filmed Dataset", os.path.expanduser("~")+"/PhD/data/")
        if filmedDatasetBaseLoc != "" :
            self.datasetReady = False
            self.updateUIForDataset()
            
            del self.filmedDataset
            self.filmedDataset = FilmedDataset(filmedDatasetBaseLoc)
            self.datasetReady = not self.filmedDataset.isEmptyDataset
            
            if not self.datasetReady :
                QtGui.QMessageBox.warning(self, "Invalid Dataset", "<center>The selected dataset is invalid.\nIf you need more details, complain to the dev.</center>")
            
            self.updateUIForDataset()
            
            if self.currentCreationStage != CREATION_STAGE_INTRINSICS :
                self.tabWidget.setCurrentIndex(0)
            else :
                self.setCurrentCreationStage(-1)
                self.setCurrentCreationStage(CREATION_STAGE_INTRINSICS)

        self.setFocus()
        
    def updateUIForDataset(self) :
        if self.datasetReady :
            ## here I should check if these things even exist (I think the only thing that I can assume is always there even for a new dataset is image)
            ## maybe move this to its own method
            self.imageLabel.setImage(self.filmedDataset.image)
            
            ## no need to check if the filmed dataset has any of the things as it should have sensible initial values
            ## but if it doesn't have the required stuff then I should set self.datasetReady to false or else the slots connected to the widgets would update stuff using invalid data
            ## intrinsics
            self.datasetReady = self.filmedDataset.hasIntrinsics()
            self.fxParameterEdit.setText(np.string_(self.filmedDataset.originalIntrinsics[0, 0]))
            self.fyParameterEdit.setText(np.string_(self.filmedDataset.originalIntrinsics[1, 1]))
            self.sParameterEdit.setText(np.string_(self.filmedDataset.originalIntrinsics[0, 1]))
            self.x0ParameterEdit.setText(np.string_(self.filmedDataset.originalIntrinsics[0, 2]))
            self.y0ParameterEdit.setText(np.string_(self.filmedDataset.originalIntrinsics[1, 2]))

            self.distortionParameterSpinBox.setValue(self.filmedDataset.distortionParameter)
            self.distortionRatioSpinBox.setValue(self.filmedDataset.distortionRatio)
            
            self.doShowUndistortedCheckBox.setChecked(self.filmedDataset.hasIntrinsics())
            
            ## extrinsics
            if self.filmedDataset.hasIntrinsics() and not self.filmedDataset.hasExtrinsics() :
                self.initExtrinsics()
            self.datasetReady = self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics()
            self.firstEdgeSizeSpinBox.setValue(self.filmedDataset.firstEdgeSize)
            self.secondEdgeSizeSpinBox.setValue(self.filmedDataset.secondEdgeSize)
            self.reprojectionErrorLabel.setText("{0:.2f}".format(self.filmedDataset.currentReprojectionError))
            
            ## scene geometry
            if self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics() and not self.filmedDataset.hasSceneGeometry() :
                self.initSceneGeometry()
            self.datasetReady = self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics() and self.filmedDataset.hasSceneGeometry()
            self.deleteMeshPointIdxSpinBox.setRange(1, len(self.filmedDataset.cameraGroundPlaneGeometryPoints))
            self.segmentsToExtrudeEdit.setText("-".join((self.filmedDataset.segmentsToExtrude+1).astype(np.string_)))
            self.extrusionHeightSpinBox.setValue(self.filmedDataset.extrusionHeight)
                
            del self.filmedObjectsPatches
            self.filmedObjectsPatches = {}
            for objectKey in self.filmedDataset.filmedObjectsData.keys() :
                self.filmedObjectsPatches[objectKey] = np.load(self.filmedDataset.filmedObjectsData[objectKey][DICT_PATCHES_LOCATION]).item()
            self.setObjectsVis()
                
            self.commentsEdit.setText(self.filmedDataset.filmedSceneData[DICT_COMMENTS])

            self.intrinsicsInfoLabel.setText("")
            self.tabWidget.setTabText(0, "Intrinsics")
            self.extrinsicsInfoLabel.setText("")
            self.tabWidget.setTabText(1, "Extrinsics")
            self.sceneInfoLabel.setText("")
            self.tabWidget.setTabText(2, "Scene")
            self.objectsInfoLabel.setText("")
            self.tabWidget.setTabText(3, "Objects")
            
            self.datasetInfoLabel.setText("")
            self.unsavedIntrinsics = False
            self.unsavedExtrinsics = False
            self.unsavedSceneGeometry = False
            self.unsavedObjects = False
            self.unsavedToDisk = False
            
            self.tabWidget.setTabEnabled(0, True)
            self.tabWidget.setTabEnabled(1, self.filmedDataset.hasIntrinsics())
            self.tabWidget.setTabEnabled(2, self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics())
            self.tabWidget.setTabEnabled(3, self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics())
            
            self.datasetReady = True
        else :
            ## reset the image label
            self.imageLabel.setImage(np.empty([0, 0], np.uint8))
            self.imageLabel.setLines(np.empty([0, 4], float))
            self.imageLabel.setRectanglePoints(np.empty([0, 2], float))
            self.imageLabel.setSceneOriginPoint(np.empty([0], dtype=float))
            self.imageLabel.setFrustumPoints(np.empty([0, 2], dtype=float))
            self.imageLabel.setMesh(np.empty([0, 2], dtype=float), np.empty([0], dtype=int), np.empty([0], dtype=bool))
            self.imageLabel.setSelectedPoint(np.empty([0], dtype=float))
        self.resetScene3DData()
            
    def eventFilter(self, obj, event) :
        if obj == self.imageLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            self.mouseMoved(event)
            return True
        elif obj == self.imageLabel and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.mousePressed(event)
            return True
        elif obj == self.imageLabel and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
            self.mouseReleased(event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
                
    def pointAroundLoc(self, loc, threshold) :
        """returns point found around the loc"""
        if self.datasetReady :
            allPoints = np.empty([0, 2], dtype=float)
            if self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
                if self.doShowScene3DGroundPlaneCheckBox.isChecked() :
                    if len(self.scene3DGroundPlaneCameraLines) > 0 :
                        allPoints = np.vstack([allPoints, self.scene3DGroundPlaneCameraLines.reshape([len(self.scene3DGroundPlaneCameraLines)*2, 2])])

                        ## this is only possible if I have enough lines
                        if len(self.scene3DGroundPlaneCameraSceneOriginPoint) == 2 :
                            allPoints = np.vstack([allPoints, self.scene3DGroundPlaneCameraSceneOriginPoint.reshape([1, 2])])

                else :
                    if len(self.filmedDataset.cameraLines) > 0 :
                        allPoints = np.vstack([allPoints, self.filmedDataset.cameraLines.reshape([len(self.filmedDataset.cameraLines)*2, 2])])

                        ## this is only possible if I have enough lines
                        if len(self.filmedDataset.cameraSceneOriginPoint) == 2 :
                            allPoints = np.vstack([allPoints, self.filmedDataset.cameraSceneOriginPoint.reshape([1, 2])])
            elif self.currentCreationStage == CREATION_STAGE_SCENE :
                if len(self.filmedDataset.cameraGroundPlaneGeometryPoints) > 0 :
                    allPoints = np.vstack([allPoints, self.filmedDataset.cameraGroundPlaneGeometryPoints])
            
            if len(allPoints) > 0 :
                self.pointToMove = findClosestPoint(allPoints, np.array([[loc[0], loc[1]]]), threshold)
            else :
                self.pointToMove = -1
            return self.movePoint(0, 0)
        
    def movePoint(self, deltaX, deltaY, minX=-np.inf, minY=-np.inf, maxX=np.inf, maxY=np.inf) :
        """returns the moved point"""
        if self.pointToMove != -1 :
            if self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
                if self.doShowScene3DGroundPlaneCheckBox.isChecked() :
                    pass
                if self.pointToMove < 8 :
                    ## move point that defines lines
                    movedPoint = self.filmedDataset.moveCameraLinesPoint(self.pointToMove, deltaX, deltaY, minX, minY, maxX, maxY)
                elif self.pointToMove == 8 :
                    ## move scene origin point
                    movedPoint = self.filmedDataset.moveCameraSceneOriginPoint(deltaX, deltaY, minX, minY, maxX, maxY)
                    if self.doMovePoint :
                        self.filmedDataset.isOriginPointUserDefined = True

                if self.doShowScene3DGroundPlaneCheckBox.isChecked() :
                    movedPoint = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat,
                                                    screenToWorldPlane(movedPoint[np.newaxis, :], self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics).flatten(),
                                                    self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
            elif self.currentCreationStage == CREATION_STAGE_SCENE :
                ## move geometry ground plane points
                movedPoint = self.filmedDataset.moveCameraGroundPlaneGeometryPoints(self.pointToMove, deltaX, deltaY, minX, minY, maxX, maxY)
                
            return movedPoint
        else :
            return np.empty([0], dtype=float)
        
    def canMovePoints(self) :
        return (self.datasetReady and self.imageLabel.qImage != None and (self.currentCreationStage == CREATION_STAGE_EXTRINSICS or
                                                                          (self.currentCreationStage == CREATION_STAGE_SCENE and not self.doShowScene3DGeometryCheckBox.isChecked())))
        
    def mousePressed(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            if self.canMovePoints() :
                sizeDiff = (self.imageLabel.size() - self.imageLabel.qImage.size())/2
                mousePos = (event.posF() - QtCore.QPointF(sizeDiff.width(), sizeDiff.height()))/self.imageLabel.resizeRatio
                
                self.pointAroundLoc([mousePos.x(), mousePos.y()], POINT_SELECTION_RADIUS)
                self.doMovePoint = self.pointToMove != -1

            self.prevMousePosition = event.posF()
        
    def mouseReleased(self, event) :
        self.doMovePoint = False
        self.prevMousePosition = QtCore.QPointF(0, 0)
        
        self.imageLabel.setSelectedPoint(np.empty([0], dtype=float))
        
    def mouseMoved(self, event) :
        if self.canMovePoints() :
            sizeDiff = (self.imageLabel.size() - self.imageLabel.qImage.size())/2
            mousePos = (event.posF() - QtCore.QPointF(sizeDiff.width(), sizeDiff.height()))/self.imageLabel.resizeRatio
            if self.doMovePoint :
                if (event.x() >= 0 and event.y() >= 0 and 
                        event.x() < self.imageLabel.width() and 
                        event.y() < self.imageLabel.height()) :
                    
                    deltaMove = (event.posF() - self.prevMousePosition)/self.imageLabel.resizeRatio
                    deltaMove = np.array([deltaMove.x(), deltaMove.y()])
                    
                    if self.currentCreationStage == CREATION_STAGE_EXTRINSICS and self.doShowScene3DGroundPlaneCheckBox.isChecked() :
                        mousePos = np.array([mousePos.x(), mousePos.y()])
                        prevMousePos = mousePos-deltaMove
                        T, K = glCameraToOpenCV(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat, self.filmedDataset.undistortedImage.shape[0:2])
                        viewMat, projectionMat = cvCameraToOpenGL(self.filmedDataset.cameraExtrinsics, self.filmedDataset.undistortedIntrinsics, self.filmedDataset.undistortedImage.shape[0:2])
                        
                        worldMousePos = screenToWorldPlane(mousePos[np.newaxis, :], K, T)
                        mousePos = worldToScreenSpace(viewMat, projectionMat, worldMousePos, self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0]).flatten()
                        
                        worldPrevMousePos = screenToWorldPlane(prevMousePos[np.newaxis, :], K, T)
                        prevMousePos = worldToScreenSpace(viewMat, projectionMat, worldPrevMousePos, self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0]).flatten()
                        
                        deltaMove = mousePos-prevMousePos
                        moveNorm = np.linalg.norm(deltaMove)
                        deltaMove = deltaMove/moveNorm*np.min([moveNorm, 50.0])


                    point = self.movePoint(deltaMove[0], deltaMove[1], -sizeDiff.width(), -sizeDiff.height(),
                                           (self.imageLabel.qImage.width()+sizeDiff.width())/self.imageLabel.resizeRatio,
                                           (self.imageLabel.qImage.height()+sizeDiff.height())/self.imageLabel.resizeRatio)

                    self.imageLabel.setSelectedPoint(point)

                    if self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
                        self.extrinsicsChanged(self.filmedDataset.firstEdgeSize, self.filmedDataset.secondEdgeSize)
                    elif self.currentCreationStage == CREATION_STAGE_SCENE :
                        self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, self.filmedDataset.extrusionHeight)
                            
                self.prevMousePosition = event.pos()
            else :
                point = self.pointAroundLoc(np.array([mousePos.x(), mousePos.y()]), POINT_SELECTION_RADIUS/self.imageLabel.resizeRatio)
                self.imageLabel.setSelectedPoint(point)
        
    def setCurrentCreationStage(self, currentCreationStage) :
        if self.currentCreationStage != currentCreationStage :
            self.currentCreationStage = currentCreationStage

            if self.currentCreationStage == CREATION_STAGE_INTRINSICS :
                self.setShowUndistorted(self.doShowUndistortedCheckBox.isChecked())
            elif self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
                self.setShowScene3DGroundPlane(self.doShowScene3DGroundPlaneCheckBox.isChecked())
            elif self.currentCreationStage == CREATION_STAGE_SCENE :
                self.doShowScene3DGeometryCheckBox.setChecked(False)
                self.setShowScene3DGeometry(False)
            elif self.currentCreationStage == CREATION_STAGE_OBJECTS :
                self.setShowScene3DGeometry(False)
                self.showFrame(self.frameIdxSlider.value())
            self.setProperViz()
                
            self.imageLabel.update()
            
    def setProperViz(self) :
        if self.currentCreationStage == CREATION_STAGE_INTRINSICS :
            self.imageLabel.doDrawLines = self.doShowUndistortedCheckBox.isChecked()
            self.imageLabel.doDrawLineControls = False
            self.imageLabel.doDrawRectangle = True
            self.imageLabel.doDrawControls = False
            self.imageLabel.doDrawFrustumPoints = False
            self.imageLabel.doDrawMesh = False
            self.imageLabel.doDrawTrajectories = False
            
            self.frameIdxSlider.setVisible(False)

        elif self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
            self.imageLabel.doDrawLines = True
            self.imageLabel.doDrawLineControls = True
            self.imageLabel.doDrawRectangle = True
            self.imageLabel.doDrawControls = True
            self.imageLabel.doDrawFrustumPoints = self.doShowScene3DGroundPlaneCheckBox.isChecked()
            self.imageLabel.doDrawMesh = False
            self.imageLabel.doDrawTrajectories = False
            
            self.frameIdxSlider.setVisible(False)

        elif self.currentCreationStage == CREATION_STAGE_SCENE :
            self.imageLabel.doDrawLines = False
            self.imageLabel.doDrawLineControls = False
            self.imageLabel.doDrawRectangle = False
            self.imageLabel.doDrawControls = True
            self.imageLabel.doDrawFrustumPoints = self.doShowScene3DGeometryCheckBox.isChecked()
            self.imageLabel.doDrawMesh = True
            self.imageLabel.doDrawTrajectories = False
            
            self.frameIdxSlider.setVisible(False)

        elif self.currentCreationStage == CREATION_STAGE_OBJECTS :
            self.imageLabel.doDrawLines = False
            self.imageLabel.doDrawLineControls = False
            self.imageLabel.doDrawRectangle = False
            self.imageLabel.doDrawControls = False
            self.imageLabel.doDrawFrustumPoints = False
            self.imageLabel.doDrawMesh = True
            self.imageLabel.doDrawTrajectories = True
            
            self.frameIdxSlider.setVisible(True)
        
            
    def setShowUndistorted(self, doShow) :
        if self.datasetReady :
            if doShow :
#                 print "show undistorted"
                self.imageLabel.setImage(self.filmedDataset.undistortedImage)
                self.imageLabel.setLines(self.filmedDataset.cameraLines)
                self.imageLabel.setRectanglePoints(self.filmedDataset.cameraSpaceRectangle)
                self.imageLabel.setSceneOriginPoint(self.filmedDataset.cameraSceneOriginPoint)
                
                self.setProperViz()
            else :
#                 print "show distorted"
                self.imageLabel.setImage(self.filmedDataset.image)
                if len(self.filmedDataset.cameraSpaceRectangle) > 0 :
                    self.imageLabel.setRectanglePoints(distortPoints(self.filmedDataset.cameraSpaceRectangle, self.filmedDataset.distortionCoeff,
                                                                     self.filmedDataset.undistortedIntrinsics, self.filmedDataset.originalIntrinsics))
                if len(self.filmedDataset.cameraSceneOriginPoint) == 2 :
                    self.imageLabel.setSceneOriginPoint(distortPoints(self.filmedDataset.cameraSceneOriginPoint.reshape([1, 2]), self.filmedDataset.distortionCoeff,
                                                                      self.filmedDataset.undistortedIntrinsics, self.filmedDataset.originalIntrinsics).flatten())
                    
                self.setProperViz()
                
    def setShowScene3DGroundPlane(self, doShow) :
        if self.datasetReady :
            if doShow :                
                if self.updateScene3DGroundPlane() :
                    self.imageLabel.setImage(self.scene3DGroundPlaneImage)
                    self.imageLabel.setFrustumPoints(self.scene3DGroundPlaneCameraFrustumPoints)
                    self.imageLabel.setLines(self.scene3DGroundPlaneCameraLines)
                    self.imageLabel.setRectanglePoints(self.scene3DGroundPlaneCameraRectanglePoints)
                    self.imageLabel.setSceneOriginPoint(self.scene3DGroundPlaneCameraSceneOriginPoint)

                    self.setProperViz()
                else :
                    self.doShowScene3DGroundPlaneCheckBox.setCheckState(QtCore.Qt.Unchecked)
            else :
                self.setShowUndistorted(True)
                
    def enableSceneGeometryUI(self, doEnable) :
        self.doDeleteMeshPointButton.setEnabled(doEnable)
        self.doAddMeshPointButton.setEnabled(doEnable)
        self.acceptSegmentsToExtrudeButton.setEnabled(doEnable)
        self.extrusionHeightSpinBox.setEnabled(doEnable)
        self.doShowScene3DGeometryCheckBox.setEnabled(doEnable)
        self.doMoveScene3DGeometryCameraCWCheckBox.setEnabled(doEnable)
        self.doMoveScene3DGeometryCameraHigherCheckBox.setEnabled(doEnable)
        
                
    def setShowScene3DGeometry(self, doShow) :
        if self.datasetReady :
            if doShow :
#                 print "show scene geometry"
                ## disable shit that could trigger this again as updateScene3DGeometry can take a while
                self.enableSceneGeometryUI(False)
                if self.updateScene3DGeometry() :
#                     self.enableSceneGeometryUI(True)

                    self.setProperViz()
                else :
                    self.doShowScene3DGeometryCheckBox.setCheckState(QtCore.Qt.Unchecked)
            else :
                self.imageLabel.setMesh(np.vstack([self.filmedDataset.cameraGroundPlaneGeometryPoints, self.filmedDataset.cameraExtrudedGeometryPoints]),
                                        np.concatenate([self.filmedDataset.groundPlaneGeometryPointsIndices,
                                                        self.filmedDataset.extrudedGeometryPointsIndices+len(self.filmedDataset.cameraGroundPlaneGeometryPoints)]),
                                        np.concatenate([np.ones(len(self.filmedDataset.cameraGroundPlaneGeometryPoints), dtype=bool),
                                                        np.zeros(len(self.filmedDataset.cameraExtrudedGeometryPoints), dtype=bool)]))
                
                self.setShowUndistorted(True)
                
    def setObjectsVis(self) :
        self.setObjectsTable()
        self.setSliderDrawables()
        trajectories = []
        colors = []
        for objectKey in self.filmedDataset.filmedObjectsData.keys() :
            if DICT_REPRESENTATIVE_COLOR in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                col = np.array(self.filmedDataset.filmedObjectsData[objectKey][DICT_REPRESENTATIVE_COLOR]).flatten()
            else :
                col = np.array([0, 0, 0])
            colors.append(col)
            
            trajectories.append(getUndistortedTrajectoryPoints(self.filmedDataset.filmedObjectsData[objectKey], self.filmedDataset.filmedSceneData, self.filmedDataset.undistortedIntrinsics))
            
        self.imageLabel.setTrajectories(trajectories, colors)
                
    def updateScene3DGroundPlane(self) :
        if self.datasetReady :
            try :
                self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat = cvCameraToOpenGL(self.filmedDataset.cameraExtrinsics, self.filmedDataset.undistortedIntrinsics,
                                                                                                        self.filmedDataset.undistortedImage.shape[0:2])
                points3D = self.filmedDataset.getWorldSpaceRectangle()
                points2D = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat, points3D,
                                              self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                worldFrustumPoints = np.dot(np.linalg.inv(self.filmedDataset.cameraExtrinsics), np.vstack([self.frustumPoints.T, np.ones([1, len(self.frustumPoints)])]))

                ## move virtual camera
                self.scene3DGroundPlaneViewMat = moveVirtualCamera(self.scene3DGroundPlaneViewMat, self.doMoveScene3DGroundPlaneCameraCWCheckBox.isChecked(),
                                                                   self.doMoveScene3DGroundPlaneCameraHigherCheckBox.isChecked())

                ## update the projection of the image projected onto the grond plane
                ## find points in virtual camera
                virtualCameraPoints2D = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat, points3D,
                                                           self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                ## find homography between the points the points3D project to from the original view to the ones they project to in the virtual view
                M = cv2.findHomography(points2D, virtualCameraPoints2D)[0]
                self.scene3DGroundPlaneImage = cv2.warpPerspective(self.filmedDataset.undistortedImage, M, tuple(self.filmedDataset.undistortedImage.shape[0:2][::-1]))

                ## update visualization stuff
                self.scene3DGroundPlaneCameraFrustumPoints = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat, worldFrustumPoints.T[:, :-1],
                                                                                self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                self.scene3DGroundPlaneCameraLines = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat,
                                                                        screenToWorldPlane(np.reshape(self.filmedDataset.cameraLines, [len(self.filmedDataset.cameraLines)*2, 2]),
                                                                                           self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics),
                                                                        self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0]).reshape(self.filmedDataset.cameraLines.shape)
                self.scene3DGroundPlaneCameraRectanglePoints = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat,
                                                                                  screenToWorldPlane(self.filmedDataset.cameraSpaceRectangle,
                                                                                                     self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics),
                                                                                  self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                self.scene3DGroundPlaneCameraSceneOriginPoint = worldToScreenSpace(self.scene3DGroundPlaneViewMat, self.scene3DGroundPlaneProjectionMat,
                                                                                   screenToWorldPlane(self.filmedDataset.cameraSceneOriginPoint[np.newaxis, :],
                                                                                                      self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics).flatten(),
                                                                                   self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                return True
            except LinAlgError :
                QtGui.QMessageBox.critical(self, "Linear Algebra Error",
                                           "<center>A Linear Algebra Error occurred while computing the 3D view. This usually means the camera intrinsics or extrinsics are invalid.<br>"+
                                           "Try redefining the intrinsics \"Intrinsics\" stage or resetting the parallel lines in the \"Extrinsics\" stage.</center>")
                return False
                
        else :
            return False
            
        
    def updateScene3DGeometry(self) :
        if self.datasetReady :
            try :
                self.scene3DGeometryViewMat, self.scene3DGeometryProjectionMat = cvCameraToOpenGL(self.filmedDataset.cameraExtrinsics, self.filmedDataset.undistortedIntrinsics,
                                                                                                  self.filmedDataset.undistortedImage.shape[0:2])
                ## move virtual camera
                self.scene3DGeometryViewMat = moveVirtualCamera(self.scene3DGeometryViewMat, self.doMoveScene3DGeometryCameraCWCheckBox.isChecked(), 
                                                                self.doMoveScene3DGeometryCameraHigherCheckBox.isChecked())

                ## update visualization stuff
                worldFrustumPoints = np.dot(np.linalg.inv(self.filmedDataset.cameraExtrinsics), np.vstack([self.frustumPoints.T, np.ones([1, len(self.frustumPoints)])]))
                self.scene3DGeometryCameraFrustumPoints = worldToScreenSpace(self.scene3DGeometryViewMat, self.scene3DGeometryProjectionMat, worldFrustumPoints.T[:, :-1],
                                                                             self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                self.scene3DGeometryCameraGroundPlaneGeometryPoints = worldToScreenSpace(self.scene3DGeometryViewMat, self.scene3DGeometryProjectionMat,
                                                                                         screenToWorldPlane(self.filmedDataset.cameraGroundPlaneGeometryPoints,
                                                                                                            self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics),
                                                                                         self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])
                self.scene3DGeometryCameraExtrudedGeometryPoints = worldToScreenSpace(self.scene3DGeometryViewMat, self.scene3DGeometryProjectionMat, self.filmedDataset.worldExtrudedGeometryPoints,
                                                                                      self.filmedDataset.undistortedImage.shape[1], self.filmedDataset.undistortedImage.shape[0])

                threadIdx = self.getOperationThread()
                self.operationThreads[threadIdx].doneOperationSignal.connect(self.updateScene3DGeometryVis)
                self.operationThreads[threadIdx].doRun(ProjectiveTextureMeshOperation(), [self.filmedDataset.undistortedImage, self.filmedDataset,
                                                                                          self.scene3DGeometryViewMat, self.scene3DGeometryProjectionMat])
                return True
            except LinAlgError :
                QtGui.QMessageBox.critical(self, "Linear Algebra Error",
                                           "<center>A Linear Algebra Error occurred while computing the 3D view. This usually means the camera intrinsics or extrinsics are invalid.<br>"+
                                           "Try redefining the intrinsics \"Intrinsics\" stage or resetting the parallel lines in the \"Extrinsics\" stage.</center>")
                return False
        else :
            return False
                
    def updateScene3DGeometryVis(self, threadIdx) :
        if self.doShowScene3DGeometryCheckBox.isChecked() and self.operationThreads[threadIdx].longOperation is not None :
            if not self.operationThreads[threadIdx].longOperation.abortRequested :
                self.scene3DGeometryImage = np.copy(self.operationThreads[threadIdx].longOperation.texturedMeshImage)
                self.imageLabel.setImage(self.scene3DGeometryImage)
                self.imageLabel.setFrustumPoints(self.scene3DGeometryCameraFrustumPoints)
                self.imageLabel.setMesh(np.vstack([self.scene3DGeometryCameraGroundPlaneGeometryPoints, self.scene3DGeometryCameraExtrudedGeometryPoints]),
                                        np.concatenate([self.filmedDataset.groundPlaneGeometryPointsIndices,
                                                        self.filmedDataset.extrudedGeometryPointsIndices+len(self.filmedDataset.cameraGroundPlaneGeometryPoints)]),
                                        np.concatenate([np.ones(len(self.scene3DGeometryCameraGroundPlaneGeometryPoints), dtype=bool),
                                                        np.zeros(len(self.scene3DGeometryCameraExtrudedGeometryPoints), dtype=bool)]))
                
            self.operationThreads[threadIdx].doneOperationSignal.disconnect()
            self.operationThreads[threadIdx].doCleanUp()
        self.enableSceneGeometryUI(True)
        
    def initExtrinsics(self) :
        if self.filmedDataset.hasIntrinsics() :
            height, width = self.filmedDataset.undistortedImage.shape[0:2]
            self.filmedDataset.cameraLines = np.array([[width*3.0/4.0, height*2.0/3.0, width/4.0, height*2.0/3.0],
                                                       [width*3.0/4.0, height/3.0, width/4.0, height/3.0],
                                                       [width/3.0, height*3.0/4.0, width/3.0, height/4.0],
                                                       [width*2.0/3.0, height*3.0/4.0, width*2.0/3.0, height/4.0]], dtype=float)
            self.filmedDataset.updateCameraSpaceRectangle()
            self.extrinsicsChanged(self.filmedDataset.firstEdgeSize, self.filmedDataset.secondEdgeSize)
            
    def initSceneGeometry(self) :
        if self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics() :
            self.filmedDataset.cameraGroundPlaneGeometryPoints = self.filmedDataset.cameraSpaceRectangle
            self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, self.filmedDataset.extrusionHeight)
            self.deleteMeshPointIdxSpinBox.setRange(1, len(self.filmedDataset.cameraGroundPlaneGeometryPoints))
                
    def saveIntrinsicsPressed(self) :
        self.filmedDataset.saveIntrinsics()
        self.unsavedIntrinsics = False
        self.intrinsicsInfoLabel.setText("")
        self.tabWidget.setTabText(0, "Intrinsics")
        self.unsavedToDisk = True
        self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")
        
        if not self.filmedDataset.hasExtrinsics() :
            ## unlock extrinsics stage and init values and shit
            self.tabWidget.setTabEnabled(1, self.filmedDataset.hasIntrinsics())
            self.initExtrinsics()

        self.setFocus()
        
    def saveExtrinsicsPressed(self) :
        self.filmedDataset.saveExtrinsics()
        self.unsavedExtrinsics = False
        self.extrinsicsInfoLabel.setText("")
        self.tabWidget.setTabText(1, "Extrinsics")
        self.unsavedToDisk = True
        self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")
        
        if not self.filmedDataset.hasSceneGeometry() :
            ## unlock scene geometry and objects stage and init values and shit
            self.tabWidget.setTabEnabled(2, self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics())
            self.initSceneGeometry()
            self.tabWidget.setTabEnabled(3, self.filmedDataset.hasIntrinsics() and self.filmedDataset.hasExtrinsics())

        self.setFocus()
        
    def saveSceneGeometryPressed(self) :
        self.filmedDataset.saveSceneGeometry()
        self.unsavedSceneGeometry = False
        self.sceneInfoLabel.setText("")
        self.tabWidget.setTabText(2, "Scene")
        self.unsavedToDisk = True
        self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")

        self.setFocus()
        
    def saveObjectsPressed(self) :
        sortedObjectsKeys = np.sort(self.filmedDataset.filmedObjectsData.keys())
        objectsNotReadyIdxs = []
        if len(sortedObjectsKeys) > 0 :
            for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                for operationThread in self.operationThreads :
                    if operationThread.isRunning() and operationThread.longOperation is not None :
                        try :
                            if operationThread.longOperation.objectName == self.objectsTableWidget.cellWidget(i, 1).text() :
                                objectsNotReadyIdxs.append(i/2)
                        except AttributeError:
                            pass
                if self.objectsTableWidget.cellWidget(i+1, 0).value() != 100 and i/2 not in objectsNotReadyIdxs :
                    objectsNotReadyIdxs.append(i/2)
                        
        proceed = True
        if len(objectsNotReadyIdxs) > 0 :
            proceed = QtGui.QMessageBox.question(self, 'Save Objects', "<center>Objects ["+", ".join([self.filmedDataset.filmedObjectsData[sortedObjectsKeys[idx]][DICT_FILMED_OBJECT_NAME] for idx in objectsNotReadyIdxs])+
                                                 "] are still processing <b>or</b> need to be re-processed. Saving now might corrupt them.<br>Are you sure you want to save now?</center>", 
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            
        if proceed :
            objectsLengths = []
            objectsWidths = []
            objectsHeights = []
            objectsComments = []
            for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                objectsLengths.append(float(self.objectsTableWidget.cellWidget(i, 2).text()))
                objectsWidths.append(float(self.objectsTableWidget.cellWidget(i, 3).text()))
                objectsHeights.append(float(self.objectsTableWidget.cellWidget(i, 4).text()))
                objectsComments.append(self.objectsTableWidget.cellWidget(i, 5).text())
            
            if self.filmedDataset.saveObjects(objectsLengths, objectsWidths, objectsHeights, objectsComments) :
                self.unsavedObjects = False
                self.objectsInfoLabel.setText("")
                self.tabWidget.setTabText(3, "Objects")
                self.unsavedToDisk = True
                self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")
                self.showFrame(self.frameIdxSlider.value())

        self.setFocus()
        
    def saveToDiskPressed(self) :
        self.datasetInfoLabel.setText("")
        
        self.unsavedToDisk = False
        self.filmedDataset.saveFilmedDatasetToDisk()

        self.setFocus()
        
    def currentCreationStageChanged(self, currentIndex) :
        if self.tabWidget.tabText(currentIndex) == "Intrinsics" or self.tabWidget.tabText(currentIndex) == "*Intrinsics*" :
            self.setCurrentCreationStage(CREATION_STAGE_INTRINSICS)
        elif self.tabWidget.tabText(currentIndex) == "Extrinsics" or self.tabWidget.tabText(currentIndex) == "*Extrinsics*" :
            self.setCurrentCreationStage(CREATION_STAGE_EXTRINSICS)
        elif self.tabWidget.tabText(currentIndex) == "Scene" or self.tabWidget.tabText(currentIndex) == "*Scene*" :
            self.setCurrentCreationStage(CREATION_STAGE_SCENE)
        elif self.tabWidget.tabText(currentIndex) == "Objects" or self.tabWidget.tabText(currentIndex) == "*Objects*" :
            self.setCurrentCreationStage(CREATION_STAGE_OBJECTS)
            
    def intrinsicsParametersAccept(self) :
        if self.datasetReady :
            self.intrinsicsChanged(float(self.fxParameterEdit.text()), float(self.fyParameterEdit.text()),
                                   float(self.sParameterEdit.text()), float(self.x0ParameterEdit.text()),
                                   float(self.y0ParameterEdit.text()), self.filmedDataset.distortionParameter,
                                   self.filmedDataset.distortionRatio)
        
    def distortionParameterChanged(self, value) :
        if self.datasetReady :
            self.intrinsicsChanged(self.filmedDataset.originalIntrinsics[0, 0], self.filmedDataset.originalIntrinsics[1, 1],
                                   self.filmedDataset.originalIntrinsics[0, 1], self.filmedDataset.originalIntrinsics[0, 2],
                                   self.filmedDataset.originalIntrinsics[1, 2], value, self.filmedDataset.distortionRatio)
        
    def distortionRatioChanged(self, value) :
        if self.datasetReady :
            self.intrinsicsChanged(self.filmedDataset.originalIntrinsics[0, 0], self.filmedDataset.originalIntrinsics[1, 1],
                                   self.filmedDataset.originalIntrinsics[0, 1], self.filmedDataset.originalIntrinsics[0, 2],
                                   self.filmedDataset.originalIntrinsics[1, 2], self.filmedDataset.distortionParameter, value)
    
    def intrinsicsChanged(self, fx, fy, s, x0, y0, distortionParameter, distortionRatio) :
        if self.datasetReady :
            self.filmedDataset.updateIntrinsics(fx, fy, s, x0, y0, distortionParameter, distortionRatio)

            if self.doShowUndistortedCheckBox.isChecked() :
                self.setShowUndistorted(True)
            else :
                self.doShowUndistortedCheckBox.setChecked(True)

            self.unsavedIntrinsics = True
            self.intrinsicsInfoLabel.setText("<b>*There are unsaved changes*</b>")
            self.tabWidget.setTabText(0, "*Intrinsics*")
            
            if self.filmedDataset.hasExtrinsics() :
                self.extrinsicsChanged(self.filmedDataset.firstEdgeSize, self.filmedDataset.secondEdgeSize)
            
    def firstEdgeSizeChanged(self, value) :
        if self.datasetReady :
            self.extrinsicsChanged(value, self.filmedDataset.secondEdgeSize)
            
    def secondEdgeSizeChanged(self, value) :
        if self.datasetReady :
            self.extrinsicsChanged(self.filmedDataset.firstEdgeSize, value)

    def extrinsicsChanged(self, firstEdgeSize, secondEdgeSize) :
        ## here I should compute the homography and stuff
        if self.datasetReady :
            self.filmedDataset.updateExtrinsics(firstEdgeSize, secondEdgeSize)
            
            ## updating UI
            self.reprojectionErrorLabel.setText("{0:.2f}".format(self.filmedDataset.currentReprojectionError))
            
            if self.currentCreationStage == CREATION_STAGE_EXTRINSICS :
                self.setShowScene3DGroundPlane(self.doShowScene3DGroundPlaneCheckBox.isChecked())
            
            self.unsavedExtrinsics = True
            self.extrinsicsInfoLabel.setText("<b>*There are unsaved changes*</b>")
            self.tabWidget.setTabText(1, "*Extrinsics*")
            
            if self.filmedDataset.hasSceneGeometry() :
                self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, self.filmedDataset.extrusionHeight)
                self.objectsChanged()
                ## set processing for objects to 0 so that users know to re-process
                if self.objectsTableWidget.rowCount() > 1 :
                    for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                        self.objectsTableWidget.cellWidget(i+1, 0).setValue(0)
                

    def sceneGeometryChanged(self, segmentsToExtrude, extrusionHeight, force3DSceneRefresh=False) :
        """if force3DSceneRefresh == True, it refreshes the 3D scene if needed (i.e. when doShowScene3DGeometryCheckBox.isChecked() == True)"""
        if self.datasetReady :
            self.filmedDataset.updateSceneGeometry(segmentsToExtrude, extrusionHeight)
            
            if self.currentCreationStage == CREATION_STAGE_SCENE :
                self.setShowScene3DGeometry(force3DSceneRefresh and self.doShowScene3DGeometryCheckBox.isChecked())
            
            self.unsavedSceneGeometry = True
            self.sceneInfoLabel.setText("<b>*There are unsaved changes*</b>")
            self.tabWidget.setTabText(2, "*Scene*")
            
    def doShowUndistortedChanged(self, state) :
        self.setShowUndistorted(state == QtCore.Qt.Checked)
            
    def doShowScene3DGroundPlaneChanged(self, state) :
        self.setShowScene3DGroundPlane(state == QtCore.Qt.Checked)
            
    def doShowScene3DGeometryChanged(self, state) :
        self.setShowScene3DGeometry(state == QtCore.Qt.Checked)
        
    def doMoveScene3DGroundPlaneCameraCWChanged(self, state) :
        self.doMoveScene3DGroundPlaneCameraACWCheckBox.setChecked(state != QtCore.Qt.Checked)
        if not self.doShowScene3DGroundPlaneCheckBox.isChecked() :
            self.doShowScene3DGroundPlaneCheckBox.setChecked(True)
        else :
            self.setShowScene3DGroundPlane(True)
        
    def doMoveScene3DGroundPlaneCameraHigherChanged(self, state) :
        self.doMoveScene3DGroundPlaneCameraLowerCheckBox.setChecked(state != QtCore.Qt.Checked)
        if not self.doShowScene3DGroundPlaneCheckBox.isChecked() :
            self.doShowScene3DGroundPlaneCheckBox.setChecked(True)
        else :
            self.setShowScene3DGroundPlane(True)
        
    def doMoveScene3DGeometryCameraCWChanged(self, state) :
        self.doMoveScene3DGeometryCameraACWCheckBox.setChecked(state != QtCore.Qt.Checked)
        if not self.doShowScene3DGeometryCheckBox.isChecked() :
            self.doShowScene3DGeometryCheckBox.setChecked(True)
        else :
            self.setShowScene3DGeometry(True)
        
    def doMoveScene3DGeometryCameraHigherChanged(self, state) :
        self.doMoveScene3DGeometryCameraLowerCheckBox.setChecked(state != QtCore.Qt.Checked)
        if not self.doShowScene3DGeometryCheckBox.isChecked() :
            self.doShowScene3DGeometryCheckBox.setChecked(True)
        else :
            self.setShowScene3DGeometry(True)
        
    def doAddMeshPointPressed(self) :
        if self.datasetReady :
            self.filmedDataset.doAddMeshPoint()
            self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, self.filmedDataset.extrusionHeight, True)
            self.deleteMeshPointIdxSpinBox.setRange(1, len(self.filmedDataset.cameraGroundPlaneGeometryPoints))

        self.setFocus()
        
    def doDeleteMeshPointPressed(self) :
        if self.datasetReady :
            self.filmedDataset.doDeleteMeshPoint(self.deleteMeshPointIdxSpinBox.value()-1)
            self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, self.filmedDataset.extrusionHeight, True)
            if len(self.filmedDataset.cameraGroundPlaneGeometryPoints) > 0 :
                self.deleteMeshPointIdxSpinBox.setRange(1, len(self.filmedDataset.cameraGroundPlaneGeometryPoints))
            else :
                self.deleteMeshPointIdxSpinBox.setRange(0, 0)

        self.setFocus()
                                                            
    def segmentsToExtrudeAccept(self) :
        if self.datasetReady :
            try :
                segmentsToExtrude = np.array(self.segmentsToExtrudeEdit.text().split("-")).astype(int)-1
                segmentsToExtrude = np.unique(segmentsToExtrude)
                if np.all(segmentsToExtrude >= 0) and np.all(segmentsToExtrude < len(self.filmedDataset.cameraGroundPlaneGeometryPoints)) :
                    if self.doShowScene3DGeometryCheckBox.isChecked() :
                        self.doShowScene3DGeometryCheckBox.setChecked(False)
                    self.sceneGeometryChanged(segmentsToExtrude, self.filmedDataset.extrusionHeight, True)
                else :
                    outOfBoundsIndices = segmentsToExtrude[np.argwhere(np.any(np.vstack([(segmentsToExtrude < 0)[np.newaxis, :],
                                                                                         (segmentsToExtrude >= len(self.filmedDataset.cameraGroundPlaneGeometryPoints))[np.newaxis, :]]),
                                                                              axis=0)).flatten()]+1
                    QtGui.QMessageBox.critical(self, "Index Out of Bound",
                                               "<center>Segment indices [{0}] are out of bounds [{1}, {2}]</center>".format(", ".join(outOfBoundsIndices.astype(np.string_)),
                                                                                                                            1, len(self.filmedDataset.cameraGroundPlaneGeometryPoints)))
                    
            except ValueError :
                QtGui.QMessageBox.critical(self, "Wrong Index String", "<center>The sequence of characters separated by \"-\" is empty or contains characters other than integers</center>")
        
    def extrusionHeightChanged(self, value) :
        if self.datasetReady :
            if self.doShowScene3DGeometryCheckBox.isChecked() :
                self.doShowScene3DGeometryCheckBox.setChecked(False)
            self.sceneGeometryChanged(self.filmedDataset.segmentsToExtrude, value, True)
            
    def commentsChanged(self) :
        self.unsavedToDisk = True
        self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")
        
    def setSliderDrawables(self) :
        self.semanticsToDraw = []
        self.filmedObjectsSortedFrameKeys = {}
        numFrames = len(glob.glob(self.filmedDataset.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC]+os.sep+"frame-*.png"))
        for objectKey in np.sort(self.filmedDataset.filmedObjectsData.keys()) :
            if DICT_REPRESENTATIVE_COLOR in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                col = np.array(self.filmedDataset.filmedObjectsData[objectKey][DICT_REPRESENTATIVE_COLOR]).flatten()
            else :
                col = np.array([0, 0, 0])
                
            self.filmedObjectsSortedFrameKeys[objectKey] = readNukeTrack(self.filmedDataset.filmedObjectsData[objectKey][DICT_TRACK_LOCATION])[1]

            if len(self.filmedObjectsSortedFrameKeys[objectKey]) > 0 :
                self.semanticsToDraw.append({
                                                DRAW_COLOR:col,
                                                DRAW_FIRST_FRAME:np.min(self.filmedObjectsSortedFrameKeys[objectKey]),
                                                DRAW_LAST_FRAME:np.max(self.filmedObjectsSortedFrameKeys[objectKey])
                                            })
                
                
            
        self.frameIdxSlider.setRange(0, numFrames-1)        
        self.frameIdxSlider.setSemanticsToDraw(self.semanticsToDraw, numFrames)
        
    def setObjectsTable(self) :
        if self.datasetReady and len(self.filmedDataset.filmedObjectsData.keys()) > 0 :
            self.objectsTableWidget.setRowCount(len(self.filmedDataset.filmedObjectsData.keys())*2)
            self.objectsTableWidget.clearSpans()
            for i, objectKey in enumerate(np.sort(self.filmedDataset.filmedObjectsData.keys())) :
                idx = i*2
                self.objectsTableWidget.setCellWidget(idx, 0, QtGui.QCheckBox())
                
                ## name
                nameLabel = QtGui.QLabel(self.filmedDataset.filmedObjectsData[objectKey][DICT_FILMED_OBJECT_NAME])
                nameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
                bgColor = self.filmedDataset.filmedObjectsData[objectKey][DICT_REPRESENTATIVE_COLOR]
                textColor = "black"
                if np.average(bgColor) < 127 :
                    textColor = "white"                
                nameLabel.setStyleSheet("QLabel { background-color: "+"rgb({0}, {1}, {2}); ".format(bgColor[0], bgColor[1], bgColor[2])+"color: {0}; ".format(textColor)+"}")
                self.objectsTableWidget.setCellWidget(idx, 1, nameLabel)
                
                ## length
                self.objectsTableWidget.setCellWidget(idx, 2, QtGui.QLineEdit())
                self.objectsTableWidget.cellWidget(idx, 2).setValidator(QtGui.QDoubleValidator())
                if DICT_OBJECT_LENGTH not in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_LENGTH] = 1.0
                    self.objectsChanged()
                self.objectsTableWidget.cellWidget(idx, 2).setText("{0}".format(self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_LENGTH]))
                self.objectsTableWidget.cellWidget(idx, 2).setStyleSheet("border : none; border-right: 2px solid #DDDDDD; ")
                self.objectsTableWidget.cellWidget(idx, 2).textChanged.connect(self.objectsChanged)
                ## width
                self.objectsTableWidget.setCellWidget(idx, 3, QtGui.QLineEdit())
                self.objectsTableWidget.cellWidget(idx, 3).setValidator(QtGui.QDoubleValidator())
                if DICT_OBJECT_WIDTH not in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_WIDTH] = 1.0
                    self.objectsChanged()
                self.objectsTableWidget.cellWidget(idx, 3).setText("{0}".format(self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_WIDTH]))
                self.objectsTableWidget.cellWidget(idx, 3).setStyleSheet("border : none; border-right: 2px solid #DDDDDD; ")
                self.objectsTableWidget.cellWidget(idx, 3).textChanged.connect(self.objectsChanged)
                ## height
                self.objectsTableWidget.setCellWidget(idx, 4, QtGui.QLineEdit())
                self.objectsTableWidget.cellWidget(idx, 4).setValidator(QtGui.QDoubleValidator())
                if DICT_OBJECT_HEIGHT not in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_HEIGHT] = 1.0
                    self.objectsChanged()
                self.objectsTableWidget.cellWidget(idx, 4).setText("{0}".format(self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_HEIGHT]))
                self.objectsTableWidget.cellWidget(idx, 4).setStyleSheet("border : none; border-right: 2px solid #DDDDDD; ")
                self.objectsTableWidget.cellWidget(idx, 4).textChanged.connect(self.objectsChanged)
                
                ## comments
                self.objectsTableWidget.setCellWidget(idx, 5, QtGui.QLineEdit())
                if DICT_COMMENTS in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    self.objectsTableWidget.cellWidget(idx, 5).setText(self.filmedDataset.filmedObjectsData[objectKey][DICT_COMMENTS])
                self.objectsTableWidget.cellWidget(idx, 5).textChanged.connect(self.objectsChanged)
                self.objectsTableWidget.cellWidget(idx, 5).setFrame(False)
                    
                ## add a discrete loading bar under the object
                self.objectsTableWidget.setCellWidget(idx+1, 0, QtGui.QProgressBar())
                self.objectsTableWidget.setSpan(idx+1, 0, 1, 6)
                self.objectsTableWidget.cellWidget(idx+1, 0).setStyleSheet("QProgressBar { background-color : #FFCCCC; } QProgressBar::chunk { background-color : #777777; }")
                self.objectsTableWidget.cellWidget(idx+1, 0).setMaximumHeight(2)
                self.objectsTableWidget.cellWidget(idx+1, 0).setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
                self.objectsTableWidget.cellWidget(idx+1, 0).setRange(0, 100)
                self.objectsTableWidget.cellWidget(idx+1, 0).setValue(100)
                self.objectsTableWidget.cellWidget(idx+1, 0).setTextVisible(False)
                self.objectsTableWidget.setRowHeight(idx+1, 2)
            
            self.objectsTableWidget.setColumnWidth(0, 20)
            self.objectsTableWidget.resizeColumnToContents(1)
            self.objectsTableWidget.setColumnWidth(2, 35)
            self.objectsTableWidget.setColumnWidth(3, 35)
            self.objectsTableWidget.setColumnWidth(4, 35)
            
        else :
            self.objectsTableWidget.setRowCount(1)
            self.objectsTableWidget.clearSpans()
            self.objectsTableWidget.setCellWidget(0, 0, QtGui.QLabel("<center>No Objects Exist</center>"))
            self.objectsTableWidget.setSpan(0, 0, 1, 6)
            
            self.objectsTableWidget.setColumnWidth(0, 20)
            self.objectsTableWidget.setColumnWidth(1, 60)
            self.objectsTableWidget.setColumnWidth(2, 30)
            self.objectsTableWidget.setColumnWidth(3, 30)
            self.objectsTableWidget.setColumnWidth(4, 30)
            
    def newObjectPressed(self) :
        if self.datasetReady :
            objectName, trackLocation, representativeColor, length, width, height, masksLocation, exitCode = createNewFilmedObject(self, "New Filmed Object",
                                                                                                                                   self.filmedDataset.filmedDatasetData[DICT_FILMED_DATASET_BASE_LOC])
            
            proceed = True
            if objectName in self.filmedDataset.filmedObjectsData.keys() :
                proceed = QtGui.QMessageBox.question(self, 'Existing Object', "<center>An object called \""+objectName+
                                                     "\" exists for this dataset already.<br>Would you like to override?</center>", 
                                                     QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                if proceed :
                    self.doDeleteFilmedObjects([objectName])
            
            if proceed and exitCode == 1 :
                if os.path.isfile(trackLocation) :
                    sortedFrameKeys = self.filmedDataset.addNewFilmedObject(objectName, trackLocation, representativeColor, length, width, height, masksLocation)
                    ## if new object added
                    if len(sortedFrameKeys) > 0:
                        self.setObjectsVis()
                        self.doComputePatchesForObjects([objectName])
                        self.objectsChanged()
                        self.showFrame(self.frameIdxSlider.value())
                else :
                    QtGui.QMessageBox.critical(self, "Cannot Create Object", "<center>The tracks location does not exist.</center>")

        self.setFocus()
                    
    
    def doComputePatchesForObjects(self, objectsKeys) :
        for objectKey in objectsKeys :
            if objectKey in self.filmedDataset.filmedObjectsData.keys() :
                objectLength = objectWidth = objectHeight = 1.0
                if DICT_OBJECT_LENGTH in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    objectLength = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_LENGTH]
                if DICT_OBJECT_WIDTH in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    objectWidth = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_WIDTH]
                if DICT_OBJECT_HEIGHT in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    objectHeight = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_HEIGHT]
                
                masksLocation = ""
                if DICT_MASK_LOCATION in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                    masksLocation = self.filmedDataset.filmedObjectsData[objectKey][DICT_MASK_LOCATION]
                
                    
                trajectoryPoints = getUndistortedTrajectoryPoints(self.filmedDataset.filmedObjectsData[objectKey], self.filmedDataset.filmedSceneData, self.filmedDataset.undistortedIntrinsics)
                sortedFrameKeys = readNukeTrack(self.filmedDataset.filmedObjectsData[objectKey][DICT_TRACK_LOCATION])[1]
                
                threadIdx = self.getOperationThread()
                self.operationThreads[threadIdx].doneOperationSignal.connect(self.doneComputingPatchesForBillboard)
                self.operationThreads[threadIdx].updateOperationProgressSignal.connect(self.progressComputingPatchesForBillboard)
                self.operationThreads[threadIdx].doRun(ComputePatchesForBillboardOperation(objectKey),
                                                       [self.filmedDataset.filmedObjectsData[objectKey][DICT_PATCHES_LOCATION], self.filmedDataset.filmedSceneData,
                                                        self.filmedDataset.undistortedImage, self.filmedDataset.undistortedIntrinsics, trajectoryPoints, sortedFrameKeys,
                                                        objectLength, objectWidth, objectHeight, masksLocation])
        
                    
    def progressComputingPatchesForBillboard(self, progress, objectName) :
        if objectName in self.filmedDataset.filmedObjectsData.keys() :
            objectIdx = np.argwhere(np.sort(self.filmedDataset.filmedObjectsData.keys()) == objectName).flatten()[-1]
            widgetRowIdx = objectIdx*2+1
            if np.mod(widgetRowIdx, 2) == 1 and widgetRowIdx > 0 and objectIdx < self.objectsTableWidget.rowCount() :
                self.objectsTableWidget.cellWidget(widgetRowIdx, 0).setValue(progress)

    def doneComputingPatchesForBillboard(self, threadIdx) :
        if np.all(self.operationThreads[threadIdx].longOperation != None) :
            if not self.operationThreads[threadIdx].longOperation.abortRequested :
                self.filmedDataset.saveObjectRenderData(self.operationThreads[threadIdx].longOperation.objectName, self.operationThreads[threadIdx].longOperation.orientationAngles,
                                                        self.operationThreads[threadIdx].longOperation.maxBillboardHeight)
                
                objectKey = self.operationThreads[threadIdx].longOperation.objectName
                self.filmedObjectsPatches[objectKey] = np.load(self.filmedDataset.filmedObjectsData[objectKey][DICT_PATCHES_LOCATION]).item()
                
                if self.currentCreationStage == CREATION_STAGE_OBJECTS : 
                    self.showFrame(self.frameIdxSlider.value())
                
                self.unsavedToDisk = True
                self.datasetInfoLabel.setText("<b>*There are changes not written to disk*</b>")
            else :
                print "ABORTED THREAD", threadIdx
        else :
            print "FUCK THREAD", threadIdx
            
        self.operationThreads[threadIdx].doneOperationSignal.disconnect()
        self.operationThreads[threadIdx].updateOperationProgressSignal.disconnect()
        self.operationThreads[threadIdx].doCleanUp()
        
    def getOperationThread(self) :
        for threadIdx, operationThread in enumerate(self.operationThreads) :
            if not operationThread.isRunning() and operationThread.longOperation is None :
                if threadIdx != operationThread.threadIdx :
                    raise Exception("This should not fucking happen")
#                 print "USING EXISTING THREAD", threadIdx
                return threadIdx
            
        self.operationThreads.append(LongOperationThread(len(self.operationThreads)))
#         print "ADDING NEW OPERATION THREAD", self.operationThreads[-1].threadIdx, len(self.operationThreads)
        return self.operationThreads[-1].threadIdx
        
    def deleteObjectsPressed(self) :
        sortedObjectsKeys = np.sort(self.filmedDataset.filmedObjectsData.keys())
        objectsNotReadyIdxs = []
        if len(sortedObjectsKeys) > 0 :
            for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                for operationThread in self.operationThreads :
                    if operationThread.isRunning() and operationThread.longOperation is not None :
                        try :
                            if operationThread.longOperation.objectName == self.objectsTableWidget.cellWidget(i, 1).text() :
                                objectsNotReadyIdxs.append(i/2)
                        except AttributeError:
                            pass
                
        if len(objectsNotReadyIdxs) > 0 :
            QtGui.QMessageBox.critical(self, 'Delete Objects', "<center>Objects ["+", ".join([self.filmedDataset.filmedObjectsData[sortedObjectsKeys[idx]][DICT_FILMED_OBJECT_NAME] for idx in objectsNotReadyIdxs])+
                                        "] are still processing. <br>I cannot delete any objects until all are ready. Aborting...</center>")
            return
        
        objectsToDelete = []
        if len(sortedObjectsKeys) > 0 :
            for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                if self.objectsTableWidget.cellWidget(i, 0).isChecked() :
                    objectsToDelete.append(i/2)
                
        if len(objectsToDelete) > 0 :
            proceed = QtGui.QMessageBox.question(self, 'Delete Objects', "<center>Objects ["+", ".join([self.filmedDataset.filmedObjectsData[sortedObjectsKeys[idx]][DICT_FILMED_OBJECT_NAME] for idx in objectsToDelete])+
                                                 "] will be removed from this filmed dataset <b>and from disk</b>.<br>Are you sure you want to delete?</center>", 
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            if proceed :
                objectsKeys = [self.objectsTableWidget.cellWidget(i*2, 1).text() for i in objectsToDelete]
                self.doDeleteFilmedObjects(objectsKeys)

        self.setFocus()
        
    def reComputePatchesPressed(self) :
        sortedObjectsKeys = np.sort(self.filmedDataset.filmedObjectsData.keys())
        objectsNotReadyIdxs = []
        objectsToProcess = []
        if len(sortedObjectsKeys) > 0 :
            for i in np.arange(0, self.objectsTableWidget.rowCount(), 2) :
                if self.objectsTableWidget.cellWidget(i, 0).isChecked() :
                    objectsToProcess.append(i/2)
                    for operationThread in self.operationThreads :
                        if operationThread.isRunning() and operationThread.longOperation is not None :
                            try :
                                if operationThread.longOperation.objectName == self.objectsTableWidget.cellWidget(i, 1).text() :
                                    objectsNotReadyIdxs.append(i/2)
                            except AttributeError:
                                pass
                
        if len(objectsNotReadyIdxs) > 0 :
            QtGui.QMessageBox.critical(self, 'Process Objects', "<center>Objects ["+", ".join([self.filmedDataset.filmedObjectsData[sortedObjectsKeys[idx]][DICT_FILMED_OBJECT_NAME] for idx in objectsNotReadyIdxs])+
                                        "] are still processing. <br>I cannot process them are still processing. Aborting...</center>")
            return
        
        
        objectsKeys = [self.objectsTableWidget.cellWidget(i*2, 1).text() for i in objectsToProcess]
        self.doComputePatchesForObjects(objectsKeys)

        self.setFocus()
                        
    def doDeleteFilmedObjects(self, objectsKeys) :
        self.filmedDataset.doDeleteFilmedObjects(objectsKeys)
        for objectKey in objectsKeys :
            if objectKey in self.filmedObjectsPatches :
                del self.filmedObjectsPatches[objectKey]
            if objectKey in self.filmedObjectsSortedFrameKeys :
                del self.filmedObjectsSortedFrameKeys[objectKey]
        self.setObjectsVis()
        self.showFrame(self.frameIdxSlider.value())
        
    def objectsChanged(self) :
        if self.datasetReady and len(self.filmedDataset.filmedObjectsData) > 0 :
            self.unsavedObjects = True
            self.objectsInfoLabel.setText("<b>*There are unsaved changes*</b>")
            self.tabWidget.setTabText(3, "*Objects*")
            
    def changesDealtWith(self, questionToIgnoreChanges) :
        proceed = True
        
        ## check stages have been saved to dictionaries
        unsavedStagesString = ""
        if self.unsavedIntrinsics or self.unsavedExtrinsics or self.unsavedSceneGeometry or self.unsavedObjects :
            stagesNames = ["Intrinsics", "Extrinsics", "Scene", "Objects"]
            stagesUnsaved = [self.unsavedIntrinsics, self.unsavedExtrinsics, self.unsavedSceneGeometry, self.unsavedObjects]
            unsavedStagesString = "Stages ["+", ".join([stagesNames[idx] for idx, isUnsaved in enumerate(stagesUnsaved) if isUnsaved])+"] "
            unsavedStagesString += "have unsaved changes. "
        
        ## check no threads are processing objects
        objectsNotReadyNames = []
        for operationThread in self.operationThreads :
            if operationThread.isRunning() and operationThread.longOperation is not None :
                try :
                    objectsNotReadyNames.append(operationThread.longOperation.objectName)
                except AttributeError:
                    pass
        
        objectsProcessingString = ""
        if len(objectsNotReadyNames) > 0 :
            objectsProcessingString = "Objects ["+", ".join(objectsNotReadyNames)+"] are still processing. "
        
        ## check dictionaries have been saved to disk
        unsavedToDiskString = ""
        if self.unsavedToDisk :
            unsavedToDiskString = "Changes to the filmed dataset have not been saved to disk."
            
        unsavedChangesString = ""
        unsavedStuffStrings = [stuffString for stuffString in [unsavedStagesString, objectsProcessingString, unsavedToDiskString] if stuffString != ""]
        if len(unsavedStuffStrings) > 0 :
            unsavedChangesString = "<b>There are unsaved changes:</b><br><br>"
            unsavedChangesString += "<br>".join(["{0}. {1}".format(idx+1, stuffString) for idx, stuffString in enumerate(unsavedStuffStrings)])
            unsavedChangesString += "<br><br><b>They will be lost!</b>"
                
        if unsavedChangesString != "" :
            proceed = QtGui.QMessageBox.question(self, 'Unsaved Changes', "<center>"+unsavedChangesString+"<br>"+questionToIgnoreChanges+"</center>", 
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            
        return proceed
            
    def closeEvent(self, event) :
        if not self.changesDealtWith("Are you sure you want to exit?") :
            event.ignore()
            return
        
        ## the method connected to the done signal of each thread should deal with cleaning up
        for operationThread in self.operationThreads :
            operationThread.doQuit()
                
    def showFrame(self, frameKey) :
        viewMat, projectionMat = cvCameraToOpenGL(self.filmedDataset.cameraExtrinsics, self.filmedDataset.undistortedIntrinsics, self.filmedDataset.undistortedImage.shape[0:2])
        height, width = self.filmedDataset.undistortedImage.shape[0:2]
        boundingVolumes = []
        colors = []
        overlayImg = np.empty([0, 0], dtype=np.uint8)
        for objectKey in np.sort(self.filmedDataset.filmedObjectsData.keys()) :
            if objectKey in self.filmedObjectsSortedFrameKeys.keys() :
                patchIdx = np.argwhere(frameKey == self.filmedObjectsSortedFrameKeys[objectKey]).flatten()
                if len(patchIdx) == 1 :
                    patchIdx = int(patchIdx)
                    worldTrajectoryPoints = screenToWorldPlane(getUndistortedTrajectoryPoints(self.filmedDataset.filmedObjectsData[objectKey], self.filmedDataset.filmedSceneData,
                                                                                              self.filmedDataset.undistortedIntrinsics),
                                                               self.filmedDataset.undistortedIntrinsics, self.filmedDataset.cameraExtrinsics)
                    worldTrajectoryDirections = getDirectionsFromTrajectoryPoints(worldTrajectoryPoints)

                    objectLength = objectWidth = objectHeight = 1.0
                    if DICT_OBJECT_LENGTH in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                        objectLength = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_LENGTH]
                    if DICT_OBJECT_WIDTH in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                        objectWidth = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_WIDTH]
                    if DICT_OBJECT_HEIGHT in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                        objectHeight = self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_HEIGHT]

#                     objectLength = 0.58
#                     objectWidth = 0.27
#                     objectHeight = 0.18

                    worldBoundingVolumeVertices, objectTransform = placeBoundingVolumeOnTrajectory(worldTrajectoryPoints[patchIdx, :], worldTrajectoryDirections[patchIdx, :], objectLength, objectWidth, objectHeight)
                    cameraBoundingVolumeVertices = worldToScreenSpace(viewMat, projectionMat, worldBoundingVolumeVertices, width, height)
                    
                    if DICT_REPRESENTATIVE_COLOR in self.filmedDataset.filmedObjectsData[objectKey].keys() :
                        col = np.array(self.filmedDataset.filmedObjectsData[objectKey][DICT_REPRESENTATIVE_COLOR]).flatten()
                    else :
                        col = np.array([0, 0, 0])
                    colors.append(col)
                    boundingVolumes.append(cameraBoundingVolumeVertices)
                    
                    ## now morph the patch and composite onto the overlay image
                    if objectKey in self.filmedObjectsPatches.keys() and frameKey in self.filmedObjectsPatches[objectKey].keys() :
                            
                        patchColors = np.zeros(np.concatenate([self.filmedObjectsPatches[objectKey][frameKey]['patch_size'], [4]]), dtype=np.uint8)
                        patchColors[self.filmedObjectsPatches[objectKey][frameKey]['visible_indices'][:, 0],
                                    self.filmedObjectsPatches[objectKey][frameKey]['visible_indices'][:, 1], :] = self.filmedObjectsPatches[objectKey][frameKey]['sprite_colors']

                        worldBillboardVertices = getBillboardVertices(patchColors.shape[1]/float(patchColors.shape[0]), self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_BILLBOARD_SCALE])

                        billboardTransform = np.dot(objectTransform, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(self.filmedDataset.filmedObjectsData[objectKey][DICT_OBJECT_BILLBOARD_ORIENTATION][patchIdx],
                                                                                                                          np.array([0.0, 0.0, 1.0]))),
                                                                            quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0])))))

                        worldBillboardVertices = np.dot(billboardTransform, np.concatenate([worldBillboardVertices, np.ones([len(worldBillboardVertices), 1])], axis=1).T).T[:, :-1]
                        cameraBillboardVertices = worldToScreenSpace(viewMat, projectionMat, worldBillboardVertices, width, height)
                        offset = np.floor([np.min(cameraBillboardVertices[:, 0]), np.min(cameraBillboardVertices[:, 1])]).astype(int)
                        cameraBillboardVertices = cameraBillboardVertices-offset[np.newaxis, :]
                        patchSize = np.ceil([np.max(cameraBillboardVertices[:, 0]), np.max(cameraBillboardVertices[:, 1])]).astype(int)

                        cameraPatchVertices = np.array([[patchColors.shape[1], patchColors.shape[0]],
                                                        [0, patchColors.shape[0]],
                                                        [0, 0],
                                                        [patchColors.shape[1], 0]], dtype=float)
                        M = cv2.findHomography(cameraPatchVertices, cameraBillboardVertices)[0]
                        warpedImage = cv2.warpPerspective(patchColors, M, tuple(patchSize), borderValue=[0, 0, 0, 0])
                        
                        
                        if len(overlayImg) == 0 :
                            overlayImg = np.zeros([height, width, 4], dtype=np.uint8)

                        ## crop the patch if it's out of bounds
                        if np.any(offset < 0) :
                            patchSize[offset < 0] += offset[offset < 0]
                            offset[offset < 0] = 0
                            warpedImage = warpedImage[warpedImage.shape[0]-patchSize[1]:, warpedImage.shape[1]-patchSize[0]:, :]
                        if np.any(offset+patchSize > overlayImg.shape[0:2][::-1]) :
                            patchSize[offset+patchSize > overlayImg.shape[0:2][::-1]] = (overlayImg.shape[0:2][::-1]-offset)[offset+patchSize > overlayImg.shape[0:2][::-1]]
                            warpedImage = warpedImage[:patchSize[1], :patchSize[0], :]
                        
                        overlayImg[offset[1]:offset[1]+patchSize[1], offset[0]:offset[0]+patchSize[0], :] = (warpedImage*(warpedImage[:, :, -1]/255.0)[:, :, np.newaxis] +
                                                                                                             (overlayImg[offset[1]:offset[1]+patchSize[1], offset[0]:offset[0]+patchSize[0], :]*
                                                                                                              (1.0-warpedImage[:, :, -1]/255.0)[:, :, np.newaxis])).astype(np.uint8)
                        
        self.imageLabel.setOverlayImg(overlayImg)
        self.imageLabel.setBoundingVolumes(boundingVolumes, colors)

    def createGUI(self) :
        
        ## WIDGETS ##
        self.imageLabel = ImageLabel()
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.imageLabel.installEventFilter(self)
        
        self.newFilmedDatasetButton = QtGui.QPushButton("New")
        self.newFilmedDatasetButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.loadFilmedDatasetButton = QtGui.QPushButton("Load")
        self.loadFilmedDatasetButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        
        self.tabWidget = QtGui.QTabWidget()
        
        self.commentsEdit = QtGui.QTextEdit()
        self.commentsEdit.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.datasetInfoLabel = QtGui.QLabel()
        self.datasetInfoLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.saveToDiskButton = QtGui.QPushButton("Save to Disk")
        self.saveToDiskButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        
        ######## WIDGETS USED TO MANIPULATE INTRINSICS STAGE ########
        intrinsicsGroupBox = QtGui.QGroupBox()
        
        ## intrinsics
        self.acceptIntrinsicsButton = QtGui.QToolButton()
        self.acceptIntrinsicsButton.setIcon(self.acceptIntrinsicsButton.style().standardIcon(QtGui.QStyle.SP_DialogOkButton))
        
        self.fxParameterEdit = QtGui.QLineEdit()
        self.fxParameterEdit.setValidator(QtGui.QDoubleValidator())
        self.fxParameterEdit.setText("0.0")

        self.fyParameterEdit = QtGui.QLineEdit()
        self.fyParameterEdit.setValidator(QtGui.QDoubleValidator())
        self.fyParameterEdit.setText("0.0")

        self.sParameterEdit = QtGui.QLineEdit()
        self.sParameterEdit.setValidator(QtGui.QDoubleValidator())
        self.sParameterEdit.setText("0.0")

        self.x0ParameterEdit = QtGui.QLineEdit()
        self.x0ParameterEdit.setValidator(QtGui.QDoubleValidator())
        self.x0ParameterEdit.setText("0.0")

        self.y0ParameterEdit = QtGui.QLineEdit()
        self.y0ParameterEdit.setValidator(QtGui.QDoubleValidator())
        self.y0ParameterEdit.setText("0.0")
        
        
        ## distortion
        self.doShowUndistortedCheckBox = QtGui.QCheckBox("")
        self.doShowUndistortedCheckBox.setChecked(False)
        
        self.distortionParameterSpinBox = QtGui.QDoubleSpinBox()
        self.distortionParameterSpinBox.setRange(-1.0, 1.0)
        self.distortionParameterSpinBox.setSingleStep(0.01)
        self.distortionParameterSpinBox.setValue(0.0)
        
        self.distortionRatioSpinBox = QtGui.QDoubleSpinBox()
        self.distortionRatioSpinBox.setRange(-100.0, 100.0)
        self.distortionRatioSpinBox.setSingleStep(0.01)
        self.distortionRatioSpinBox.setValue(1.0)
        
        self.intrinsicsInfoLabel = QtGui.QLabel()
        self.intrinsicsSaveButton = QtGui.QPushButton("Save Intrinsics")
        
        ######## WIDGETS USED TO DEFINE EXTRINSICS BY FITTING A GROUND PLANE ########
        extrinsicsGroupBox = QtGui.QGroupBox()
        
        self.firstEdgeSizeSpinBox = QtGui.QDoubleSpinBox()
        self.firstEdgeSizeSpinBox.setRange(0.0, 100.0)
        self.firstEdgeSizeSpinBox.setSingleStep(0.01)
        self.firstEdgeSizeSpinBox.setValue(1.0)
        
        self.secondEdgeSizeSpinBox = QtGui.QDoubleSpinBox()
        self.secondEdgeSizeSpinBox.setRange(0.0, 100.0)
        self.secondEdgeSizeSpinBox.setSingleStep(0.01)
        self.secondEdgeSizeSpinBox.setValue(1.0)
        
        self.reprojectionErrorLabel = QtGui.QLabel()
        self.doShowScene3DGroundPlaneCheckBox = QtGui.QCheckBox("")
        self.doShowScene3DGroundPlaneCheckBox.setChecked(False)
        
        self.doMoveScene3DGroundPlaneCameraCWCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGroundPlaneCameraCWCheckBox.setChecked(True)
        self.doMoveScene3DGroundPlaneCameraACWCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGroundPlaneCameraACWCheckBox.setChecked(False)
        self.doMoveScene3DGroundPlaneCameraACWCheckBox.setEnabled(False)
        self.doMoveScene3DGroundPlaneCameraHigherCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGroundPlaneCameraHigherCheckBox.setChecked(True)
        self.doMoveScene3DGroundPlaneCameraLowerCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGroundPlaneCameraLowerCheckBox.setChecked(False)
        self.doMoveScene3DGroundPlaneCameraLowerCheckBox.setEnabled(False)
        
        self.extrinsicsInfoLabel = QtGui.QLabel()
        self.extrinsicsSaveButton = QtGui.QPushButton("Save Extrinsics")
        
        ######## WIDGETS USED TO DEFINE THE SCENE GEOMETRY ########
        sceneGroupBox = QtGui.QGroupBox()
        
        self.deleteMeshPointIdxSpinBox = QtGui.QSpinBox()
        self.deleteMeshPointIdxSpinBox.setRange(0, 0)
        self.deleteMeshPointIdxSpinBox.setSingleStep(1)
        self.deleteMeshPointIdxSpinBox.setValue(0)
        
        self.doDeleteMeshPointButton = QtGui.QToolButton()
        self.doDeleteMeshPointButton.setIcon(QtGui.QIcon("minus.png"))
        self.doAddMeshPointButton = QtGui.QToolButton()
        self.doAddMeshPointButton.setIcon(QtGui.QIcon("plus.png"))
        
        self.segmentsToExtrudeEdit = QtGui.QLineEdit()
        self.segmentsToExtrudeEdit.setText("")
        self.acceptSegmentsToExtrudeButton = QtGui.QToolButton()
        self.acceptSegmentsToExtrudeButton.setIcon(self.acceptIntrinsicsButton.style().standardIcon(QtGui.QStyle.SP_DialogOkButton))
        
        self.extrusionHeightSpinBox = QtGui.QDoubleSpinBox()
        self.extrusionHeightSpinBox.setRange(-100.0, +100.0)
        self.extrusionHeightSpinBox.setSingleStep(0.1)
        self.extrusionHeightSpinBox.setValue(0.0)
        
        self.doShowScene3DGeometryCheckBox = QtGui.QCheckBox("")
        self.doShowScene3DGeometryCheckBox.setChecked(False)
        
        self.doMoveScene3DGeometryCameraCWCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGeometryCameraCWCheckBox.setChecked(True)
        self.doMoveScene3DGeometryCameraACWCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGeometryCameraACWCheckBox.setChecked(False)
        self.doMoveScene3DGeometryCameraACWCheckBox.setEnabled(False)
        self.doMoveScene3DGeometryCameraHigherCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGeometryCameraHigherCheckBox.setChecked(True)
        self.doMoveScene3DGeometryCameraLowerCheckBox = QtGui.QCheckBox("")
        self.doMoveScene3DGeometryCameraLowerCheckBox.setChecked(False)
        self.doMoveScene3DGeometryCameraLowerCheckBox.setEnabled(False)
        
        self.sceneInfoLabel = QtGui.QLabel()
        self.sceneSaveButton = QtGui.QPushButton("Save Scene Geometry")
        
        ######## WIDGETS USED TO DEFINE OBJECTS ########
        objectsGroupBox = QtGui.QGroupBox()
        
        self.objectsTableWidget = QtGui.QTableWidget(0, 6)
        self.objectsTableWidget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.objectsTableWidget.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        
        self.objectsTableWidget.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem(""))
        self.objectsTableWidget.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Name"))
        self.objectsTableWidget.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("L"))
        self.objectsTableWidget.setHorizontalHeaderItem(3, QtGui.QTableWidgetItem("W"))
        self.objectsTableWidget.setHorizontalHeaderItem(4, QtGui.QTableWidgetItem("H"))
        self.objectsTableWidget.setHorizontalHeaderItem(5, QtGui.QTableWidgetItem("Comments"))
        self.objectsTableWidget.horizontalHeader().setStretchLastSection(True)
        self.objectsTableWidget.verticalHeader().setVisible(False)
        self.objectsTableWidget.setShowGrid(False)
        self.objectsTableWidget.setFocusPolicy(QtCore.Qt.NoFocus);
        
        self.newObjectButton = QtGui.QPushButton("New Object")
        self.deleteObjectsButton = QtGui.QPushButton("Delete Selected")
        self.reComputePatchesButton = QtGui.QPushButton("Re-Compute Patches for Selected")
        
        self.objectsInfoLabel = QtGui.QLabel()
        self.objectsSaveButton = QtGui.QPushButton("Save Objects")
        
        
        self.frameIdxSlider = SemanticsSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.frameIdxSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSlider.setTickInterval(100)
        self.frameIdxSlider.setSingleStep(1)
        self.frameIdxSlider.setPageStep(100)
        self.frameIdxSlider.setVisible(False)
        
        
        ## SIGNALS ##
        
        self.loadFilmedDatasetButton.clicked.connect(self.loadFilmedDatasetButtonPressed)
        self.newFilmedDatasetButton.clicked.connect(self.newFilmedDatasetButtonPressed)
        self.commentsEdit.textChanged.connect(self.commentsChanged)
        self.tabWidget.currentChanged.connect(self.currentCreationStageChanged)
        
        self.acceptIntrinsicsButton.clicked.connect(self.intrinsicsParametersAccept)
        self.distortionParameterSpinBox.valueChanged.connect(self.distortionParameterChanged)
        self.distortionRatioSpinBox.valueChanged.connect(self.distortionRatioChanged)
        self.doShowUndistortedCheckBox.stateChanged.connect(self.doShowUndistortedChanged)
        
        self.firstEdgeSizeSpinBox.valueChanged.connect(self.firstEdgeSizeChanged)
        self.secondEdgeSizeSpinBox.valueChanged.connect(self.secondEdgeSizeChanged)
        self.doShowScene3DGroundPlaneCheckBox.stateChanged.connect(self.doShowScene3DGroundPlaneChanged)
        self.doMoveScene3DGroundPlaneCameraCWCheckBox.stateChanged.connect(self.doMoveScene3DGroundPlaneCameraCWChanged)
        self.doMoveScene3DGroundPlaneCameraHigherCheckBox.stateChanged.connect(self.doMoveScene3DGroundPlaneCameraHigherChanged)
        
        self.doAddMeshPointButton.clicked.connect(self.doAddMeshPointPressed)
        self.doDeleteMeshPointButton.clicked.connect(self.doDeleteMeshPointPressed)
        self.acceptSegmentsToExtrudeButton.clicked.connect(self.segmentsToExtrudeAccept)
        self.segmentsToExtrudeEdit.returnPressed.connect(self.segmentsToExtrudeAccept)
        self.extrusionHeightSpinBox.valueChanged.connect(self.extrusionHeightChanged)
        self.doShowScene3DGeometryCheckBox.stateChanged.connect(self.doShowScene3DGeometryChanged)
        self.doMoveScene3DGeometryCameraCWCheckBox.stateChanged.connect(self.doMoveScene3DGeometryCameraCWChanged)
        self.doMoveScene3DGeometryCameraHigherCheckBox.stateChanged.connect(self.doMoveScene3DGeometryCameraHigherChanged)
        
        self.newObjectButton.clicked.connect(self.newObjectPressed)
        self.deleteObjectsButton.clicked.connect(self.deleteObjectsPressed)
        self.reComputePatchesButton.clicked.connect(self.reComputePatchesPressed)
        self.frameIdxSlider.valueChanged[int].connect(self.showFrame)
        
        self.intrinsicsSaveButton.clicked.connect(self.saveIntrinsicsPressed)
        self.extrinsicsSaveButton.clicked.connect(self.saveExtrinsicsPressed)
        self.sceneSaveButton.clicked.connect(self.saveSceneGeometryPressed)
        self.objectsSaveButton.clicked.connect(self.saveObjectsPressed)
        self.saveToDiskButton.clicked.connect(self.saveToDiskPressed)
        
        ## LAYOUTS ##
        
        mainControlsLayout = QtGui.QHBoxLayout()
        mainControlsLayout.addWidget(self.newFilmedDatasetButton)
        mainControlsLayout.addWidget(self.loadFilmedDatasetButton)
        
        ## intrinsics
        intrinsicsLayout = QtGui.QGridLayout()
        idx = 0
        intrinsicsLayout.addWidget(QtGui.QLabel("<b>Intrinsics Template</b>"), idx, 0, 1, 11, QtCore.Qt.AlignCenter); idx+=1
        intrinsicsLayout.addWidget(QtGui.QLabel("<p><i>f<sub>x</sub></i>, <i>s</i>, <i>x</i><sub>0</sub></p>"+
                                                "<p>0, <i>f<sub>y</sub></i>, <i>y</i><sub>0</sub></p>"+
                                                "<p>0, 0, 1</p>"), idx, 0, 1, 11, QtCore.Qt.AlignCenter); idx+=1
        
        intrinsicsLayout.addWidget(HorizontalLine(), idx, 0, 1, 11); idx+=1
        intrinsicsLayout.addWidget(QtGui.QLabel("<b>Original Camera Intrinsics</b>"), idx, 0, 1, 11, QtCore.Qt.AlignCenter); idx+=1
        intrinsicsLayout.addWidget(QtGui.QLabel("<i>f<sub>x</sub></i>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(self.fxParameterEdit, idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(QtGui.QLabel("<i>f<sub>y</sub></i>"), idx, 2, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(self.fyParameterEdit, idx, 3, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(QtGui.QLabel("<i>s</i>"), idx, 4, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(self.sParameterEdit, idx, 5, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(QtGui.QLabel("<i>x</i><sub>0</sub>"), idx, 6, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(self.x0ParameterEdit, idx, 7, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(QtGui.QLabel("<i>y</i><sub>0</sub>"), idx, 8, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(self.y0ParameterEdit, idx, 9, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(self.acceptIntrinsicsButton, idx, 10, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        
        intrinsicsLayout.addWidget(HorizontalLine(), idx, 0, 1, 11); idx+=1
        intrinsicsLayout.addWidget(QtGui.QLabel("<b>Distortion Parameters</b>"), idx, 0, 1, 11, QtCore.Qt.AlignCenter); idx+=1
        intrinsicsLayout.addWidget(self.doShowUndistortedCheckBox, idx, 0, 1, 1, QtCore.Qt.AlignRight)
        intrinsicsLayout.addWidget(QtGui.QLabel("Show"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        intrinsicsLayout.addWidget(self.distortionParameterSpinBox, idx, 2, 1, 4)
        intrinsicsLayout.addWidget(self.distortionRatioSpinBox, idx, 6, 1, 4); idx+=1
        
        intrinsicsLayout.addWidget(self.intrinsicsInfoLabel, idx, 0, 1, 11, QtCore.Qt.AlignCenter); idx+=1
        intrinsicsLayout.addWidget(self.intrinsicsSaveButton, idx, 0, 1, 11); idx+=1
        
        intrinsicsLayout.setRowStretch(idx, 10)
        intrinsicsGroupBox.setLayout(intrinsicsLayout)
        
        ## extrinsics
        extrinsicsLayout = QtGui.QGridLayout()
        idx = 0
        extrinsicsLayout.addWidget(QtGui.QLabel("<b>Rectangle Size [m]</b>"), idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        extrinsicsLayout.addWidget(QtGui.QLabel("1 <-> 2"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        extrinsicsLayout.addWidget(self.firstEdgeSizeSpinBox, idx, 1, 1, 2)
        extrinsicsLayout.addWidget(QtGui.QLabel("2 <-> 3"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        extrinsicsLayout.addWidget(self.secondEdgeSizeSpinBox, idx, 4, 1, 2); idx+=1
        
        extrinsicsLayout.addWidget(HorizontalLine(), idx, 0, 1, 6); idx+=1
        extrinsicsLayout.addWidget(QtGui.QLabel("<b>Visualize Extrinsics</b>"), idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        extrinsicsLayout.addWidget(QtGui.QLabel("Reprojection Error:"), idx, 0, 1, 2, QtCore.Qt.AlignRight)
        extrinsicsLayout.addWidget(self.reprojectionErrorLabel, idx, 2, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        extrinsicsLayout.addWidget(QtGui.QLabel("Show 3D"), idx, 0, 1, 2, QtCore.Qt.AlignRight)
        extrinsicsLayout.addWidget(self.doShowScene3DGroundPlaneCheckBox, idx, 2, 1, 1, QtCore.Qt.AlignLeft)
        extrinsicsLayout.addWidget(self.doMoveScene3DGroundPlaneCameraHigherCheckBox, idx, 4, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        extrinsicsLayout.addWidget(self.doMoveScene3DGroundPlaneCameraCWCheckBox, idx, 3, 1, 1, QtCore.Qt.AlignLeft)
        extrinsicsLayout.addWidget(QtGui.QLabel("3D"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        extrinsicsLayout.addWidget(self.doMoveScene3DGroundPlaneCameraACWCheckBox, idx, 5, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        extrinsicsLayout.addWidget(self.doMoveScene3DGroundPlaneCameraLowerCheckBox, idx, 4, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        
        extrinsicsLayout.addWidget(self.extrinsicsInfoLabel, idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        extrinsicsLayout.addWidget(self.extrinsicsSaveButton, idx, 0, 1, 6); idx+=1
        
        extrinsicsLayout.setRowStretch(idx, 10)
        extrinsicsGroupBox.setLayout(extrinsicsLayout)
        
        ## scene geometry
        sceneLayout = QtGui.QGridLayout()
        idx = 0
        sceneLayout.addWidget(QtGui.QLabel("<b>Change Geometry</b>"), idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        sceneLayout.addWidget(self.doAddMeshPointButton, idx, 1, 1, 1, QtCore.Qt.AlignRight)
        sceneLayout.addWidget(self.deleteMeshPointIdxSpinBox, idx, 2, 1, 2)
        sceneLayout.addWidget(self.doDeleteMeshPointButton, idx, 4, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        sceneLayout.addWidget(QtGui.QLabel("Extrude segments [i to i+1], insert \"-\"-separated i\'s"), idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        sceneLayout.addWidget(self.segmentsToExtrudeEdit, idx, 0, 1, 5)
        sceneLayout.addWidget(self.acceptSegmentsToExtrudeButton, idx, 5, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        sceneLayout.addWidget(QtGui.QLabel("Height [m]:"), idx, 0, 1, 2, QtCore.Qt.AlignRight)
        sceneLayout.addWidget(self.extrusionHeightSpinBox, idx, 2, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        
        sceneLayout.addWidget(HorizontalLine(), idx, 0, 1, 6); idx+=1
        sceneLayout.addWidget(QtGui.QLabel("<b>Visualize Scene Geometry</b>"), idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        sceneLayout.addWidget(QtGui.QLabel("Show 3D"), idx, 0, 1, 2, QtCore.Qt.AlignRight)
        sceneLayout.addWidget(self.doShowScene3DGeometryCheckBox, idx, 2, 1, 1, QtCore.Qt.AlignLeft)
        sceneLayout.addWidget(self.doMoveScene3DGeometryCameraHigherCheckBox, idx, 4, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        sceneLayout.addWidget(self.doMoveScene3DGeometryCameraCWCheckBox, idx, 3, 1, 1, QtCore.Qt.AlignLeft)
        sceneLayout.addWidget(QtGui.QLabel("3D"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        sceneLayout.addWidget(self.doMoveScene3DGeometryCameraACWCheckBox, idx, 5, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        sceneLayout.addWidget(self.doMoveScene3DGeometryCameraLowerCheckBox, idx, 4, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        
        sceneLayout.addWidget(self.sceneInfoLabel, idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        sceneLayout.addWidget(self.sceneSaveButton, idx, 0, 1, 6); idx+=1
        
        sceneLayout.setRowStretch(idx, 10)
        sceneGroupBox.setLayout(sceneLayout)
        
        ## objects
        objectsLayout = QtGui.QGridLayout()
        idx = 0
        objectsLayout.addWidget(self.objectsTableWidget, idx, 0, 1, 6); idx+=1
        objectsLayout.addWidget(self.newObjectButton, idx, 0, 1, 3)
        objectsLayout.addWidget(self.deleteObjectsButton, idx, 3, 1, 3); idx+=1
        objectsLayout.addWidget(self.reComputePatchesButton, idx, 0, 1, 6); idx+=1
        
        objectsLayout.addWidget(self.objectsInfoLabel, idx, 0, 1, 6, QtCore.Qt.AlignCenter); idx+=1
        objectsLayout.addWidget(self.objectsSaveButton, idx, 0, 1, 6); idx+=1
        
        objectsLayout.setRowStretch(idx, 10)
        objectsGroupBox.setLayout(objectsLayout)
        
        self.tabWidget.addTab(intrinsicsGroupBox, "Intrinsics")
        self.tabWidget.addTab(extrinsicsGroupBox, "Extrinsics")
        self.tabWidget.addTab(sceneGroupBox, "Scene")
        self.tabWidget.addTab(objectsGroupBox, "Objects")
        self.tabWidget.setMaximumWidth(350)
        
        
        controlsGroupBox = QtGui.QGroupBox()
#         controlsGroupBox.setMaximumWidth(350)
        controlsGroupBox.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.MinimumExpanding)
        controlsVLayout = QtGui.QVBoxLayout()
        controlsVLayout.addLayout(mainControlsLayout)
        controlsVLayout.addWidget(self.tabWidget)
        controlsVLayout.addWidget(QtGui.QLabel("<center>Comments</center>"))
        controlsVLayout.addWidget(self.commentsEdit)
        controlsVLayout.addWidget(self.datasetInfoLabel)
        controlsVLayout.addWidget(self.saveToDiskButton)
        controlsGroupBox.setLayout(controlsVLayout)
        
        mainHLayout = QtGui.QHBoxLayout()
        mainHLayout.addWidget(self.imageLabel)
        mainHLayout.addWidget(controlsGroupBox)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(mainHLayout)
        mainLayout.addWidget(self.frameIdxSlider)
        
        self.setLayout(mainLayout)


# In[27]:

window = Window()
window.show()
app.exec_()

