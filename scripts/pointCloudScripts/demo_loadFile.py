import numpy as np
import pickle
import cv2

trans = np.load('trans_2011_09_26_0001.npy')
_labeled_velo = 'full_label_2011_09_26_0001_0000000100.npy'
_Boundingbox_2d = 'BB_2d_2011_09_26_0001_0000000100.bin'
_Boundingbox_3d = 'BB_3d_2011_09_26_0001_0000000100.bin'
_rgb_img = '0000000100.png'

'''
_rgb_img : [375,1242,3]

_labeled_velo : n * [x,y,z,i,label,u,v], 
                x,y,z are in Velo coordinate
                i is intensity
                u,v are the point point projects to image coordinate
                label include ('road', 'car', 'person', 'cyclist', 'None') 

_BOundingbox : n* [ label_type,
                    [ [x1,x2,x3,x4,x5,x6,x7,x8],
                      [y1, ,,, ,,, ,,, ,,, ,y8],
                      [z1, ... ... ... ... ,z8]
                    ]
                  ]
                  for BoundingBox, x,y,z are in image coordinate
'''
#-------------------------------
# porject point from 3d (velo coordinate) to 2d (camera coordinate)
def cart2hom( pts_3d,col =1):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending col, default is 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,col))))
    return pts_3d_hom

def Project3dConerTo2dImage(point,transformation_matrix):
    pts_2d = transformation_matrix.dot( cart2hom(point).T).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d.T
#------------------------------------------

def loadBoundingBox(boundingBox):
    with open(boundingBox,'rb') as f:
        return pickle.load(f)


def plot_full_label(velo_full, image):
    mycolors = {
        'road': (255, 0, 0),
        'car': (255, 255, 0),
        'person': (0, 255, 0),
        'cyclist': (0, 255, 255),
    }

    for i in velo_full:
        x, y, z, i, label, u, v = i
        if label != 'None':
            if label not in mycolors.keys(): print('error point', i, '\n error label', velo_full[int(i[2]), :])
            cv2.circle(img=image, center=(int(u), int(v)), radius=2, thickness=-1, color=mycolors[label])

    return image

def GT_3d_image(image, boundingbox):
    type_c = {'car': (0, 255, 255), 'person': (0, 255, 0), 'cyclist': (255, 255, 0)}

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])
    _image = np.zeros([375, 1242, 3])
    for BB in boundingbox:
        label, corner = BB
        print(label)
        print(corner)

        if label!='None':
            for k in line_order:
                cv2.line(image,
                         (int(corner[0][k[0]]), int(corner[1][k[0]])),
                         (int(corner[0][k[1]]), int(corner[1][k[1]])),
                         type_c[label], 2)

    #plt.title("3D Tracklet display on image")
    #plt.axis('off')
    #plt.imshow(image)
    #plt.show()
    return image


velo_full_labeled = np.load(_labeled_velo,allow_pickle=True)
boundingbox_2d = loadBoundingBox(_Boundingbox_2d)
boundingbox_3d = loadBoundingBox(_Boundingbox_3d)

rgb_img = cv2.imread(_rgb_img)
velo_on_image = plot_full_label(velo_full_labeled, rgb_img)

test3dbox = 0
if test3dbox:
    BB_3d_to_2d = []
    for i in range(len(boundingbox_3d)):
        tmp = []
        label, bb3d = boundingbox_3d[i]
        tmp.append(label)
        tmp.append(Project3dConerTo2dImage(bb3d.T, transformation_matrix = trans))
        BB_3d_to_2d.append(tmp)
    image_velo_3dbox = GT_3d_image(velo_on_image, BB_3d_to_2d)
    vis_fov_img = 'image_velo_from3d.png'
else:
    image_velo_3dbox = GT_3d_image(velo_on_image, boundingbox_2d)
    vis_fov_img = 'image_velo_from2d.png'


cv2.imwrite(vis_fov_img,image_velo_3dbox)
