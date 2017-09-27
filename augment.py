import cv2
import numpy as np

def augment_scale(img, scale=[1.0, 1.0]):
    rows,cols,_ = img.shape
    
    if scale[1] <= 1.0:
        return img
    
    s = np.random.rand(1) * (scale[1]-1.0) + 1.0

    if s < scale[0]:
    	s = scale[0]

    M = np.float32([[s, 0, -rows/2*(s-1)], [0, s, -cols/2*(s-1)]])
    res = cv2.warpAffine(img,M,(cols,rows))
    if len(res.shape) < 3:
        res = res[:,:,np.newaxis]
    
    return res

def augment_affine(img, translation, rotation=0):
    rows,cols,_ = img.shape
    
    dx,dy = np.random.randint(-translation,+translation+1,2)
    
    if rotation != 0:
        alpha = (np.random.rand(1)*2-1) * rotation
    else:
        alpha = 0
    
    dx = translation
    dy = translation
    alpha = rotation
    
    #print(dx,dy,alpha)
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),alpha,1)
    M[0,2] += dx
    M[1,2] += dy
    
    res = cv2.warpAffine(img, M, (cols,rows))
    if len(res.shape) < 3:
        res = res[:,:,np.newaxis]
    
    return res



def augment_images(imgs, N_desired, scale=[1.05, 2.0], translation=5, rotation=5):
    N = len(imgs)

    if N>N_desired:
        return imgs

    rands = np.random.randint(0,N,size=N_desired-N)
    
    N_diff = len(rands)

    X_ = [0]*N_diff

    # warp new data
    for i in range(N_diff):
        img = imgs[rands[i]]
        img = augment_affine(img, translation, rotation)
        img = augment_scale(img, scale)
        X_[i] = img
    
    result = list(imgs)
    result.extend(X_)

    return result
