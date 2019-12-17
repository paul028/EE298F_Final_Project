from PIL import  Image
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from scipy import ndimage
from numpy import random,argsort,sqrt
from sklearn.metrics import jaccard_score, recall_score, precision_score
#from scipy.ndimage import erosion
from skimage.morphology import erosion, disk
from scipy.ndimage.morphology import distance_transform_edt
### im1, im2 should be 2D grayscale ndarray, range is (0, 255) unnormalized.
### You can use "im1 = np.array(Image.open("ILSVRC2012_test_00000025.png").convert("L"))"
### im1 = GT salient map
### im2 = pred. salient map
def getMAE(im1, im2):
    im1 = im1 / 255.0 ### Normalize
    im2 = im2 / 255.0
    if im1.shape != im2.shape:
        print("im1 and im2 don't have the same shape!")
        return
    h, w = im1.shape
    im_diff = im1 - im2
    im_diff = np.absolute(im_diff)
    im_sum = np.sum(im_diff)
    mae = im_sum / (h * w)
    return mae

### im1 = GT salient map
### im2 = pred. salient map
def getPRCurve(im1, im2):
    im1 = im1 / 255.0 ### Normalize
    im2 = im2 / 255.0
    if im1.shape != im2.shape:
        print("im1 and im2 don't have the same shape!")
        return
    h, w = im1.shape
    n1 = im1.flatten()
    n2 = im2.flatten()
    ### Binarized map
    n1[n1 > 0] = 1
    n2[n2 > 0] = 1
    n1 = n1.astype(int)
    n2 = n2.astype(int)
    average_precision = average_precision_score(n1, n2)
    precision, recall, _ = precision_recall_curve(n1, n2)
    return precision, recall
    # step_kwargs = ({'step': 'post'}
    #            if 'step' in signature(plt.fill_between).parameters
    #            else {})
    # plt.step(recall, precision, color='b', alpha=0.1, where='post')
    # plt.fill_between(recall, precision, alpha=0.1, color='b', **step_kwargs)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    # plt.show()

### Returns two numbers only!
### Returns the precision and recall of two images
### im1 = GT salient map
### im2 = pred. salient map
# def getImagePrecRecall(im1, im2):
#     im1 = im1 / 255.0 ### Normalize
#     im2 = im2 / 255.0
#     if im1.shape != im2.shape:
#         print("im1 and im2 don't have the same shape!")
#         return
#     h, w = im1.shape
#     n1 = im1.flatten()
#     n2 = im2.flatten()
#     n1[n1 >= 0.5] = 1
#     n2[n2 >= 0.5] = 1
#     n1 = n1.astype(int)
#     n2 = n2.astype(int)
#     prec = precision_score(n1, n2, average='macro')
#     rec = recall_score(n1, n2, average='macro')
#     return prec, rec

### Compute precision, recall from getPRCurve
### precision and recall should be single numbers, not arrays!
### fmeasure - should only be a single number
def getMaxFMeasure(precision, recall):
    score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
    beta2 = math.sqrt(0.3)  # for max F_beta metric
    denom = (beta2 * precision + recall)
    denom[denom <= 0] = 100000
    fmeasure = ((1+beta2)* precision * recall) / denom
    return fmeasure

def knn_search(x, D, K):
     """ find K nearest neighbours of data among D """
     ndata = D.shape[1]
     K = K if K < ndata else ndatagetPRCurve
     # euclidean distances from the other points
     sqd = sqrt(((D - x[:,:ndata])**2).sum(axis=0))
     idx = argsort(sqd) #                sorting
     # return the indexes of K nearest neighbours and the euclidean distance of the nearest points.
     # sqd[idx[:K][0]] = 1 means that the nearest point picked from D to x has euclidean distance = 1
     return idx[:K], sqd[idx[:K][0]]

def boundary_extraction(mask):

    (h,w) = mask.shape
    mask_pad = np.zeros((h+10,w+10))
    mask_pad[5:h+5,5:w+5] = mask

    mask_pad_erd = erosion(mask_pad,disk(1))
    mask_pad_edge = np.logical_xor(mask_pad, mask_pad_erd)

    return mask_pad_edge[5:h+5,5:w+5]

def compute_align_error(gt,pd):

    gt_bw = np.zeros(gt.shape)
    pd_bw = np.zeros(pd.shape)

    #binaries gt
    gte = 127 #2*np.mean(gte)
    gt_bw[gt>gte]=1
    gt_bw[gt<=gte]=0

    #binaries pd
    pde = 127
    pd_bw[pd>pde]=1
    pd_bw[pd<=pde]=0

    gt_edge = boundary_extraction(gt_bw)
    pd_edge = boundary_extraction(pd_bw)

    gt_dist = distance_transform_edt(np.logical_not(gt_edge))
    pd_dist = distance_transform_edt(np.logical_not(pd_edge))

    buffer = 3
    #precision
    pd_edge_buffer = np.zeros(pd_edge.shape)
    pd_edge_buffer = pd_edge.copy()
    try:
        pd_edge_buffer[gt_dist>buffer] = 0
    except:
        return 0.0, 0.0, 0.0

    #recall
    gt_edge_buffer = np.zeros(gt_edge.shape)
    gt_edge_buffer = gt_edge.copy()
    try:
        gt_edge_buffer[pd_dist>buffer] = 0
    except:
        return 0.0, 0.0, 0.0



    precision_edge =np.sum(pd_edge_buffer).astype(np.float)/(np.sum(pd_edge).astype(np.float)+1e-8)

    recall_edge =np.sum(gt_edge_buffer).astype(np.float)/(np.sum(gt_edge).astype(np.float)+1e-8)

    f1_edge =(1+0.3)*precision_edge*recall_edge/(0.3*precision_edge+recall_edge+1e-8)


    return precision_edge, recall_edge, f1_edge #meanAl

def own_RelaxedFMeasure(im1,im2): ##own version of relaxed measure based from basnet forum
    rprecission,rrecall,rrfmeasure=compute_align_error(im1,im2)

    return rrfmeasure

def getRelaxedFMeasure(im1, im2):
    im1 = im1 / 255.0 ### Normalize
    im2 = im2 / 255.0
    if im1.shape != im2.shape:
        print("im1 and im2 don't have the same shape!")
        return
    h, w = im1.shape
    ### Binarized map
    im1[im1 >= 0.5] = 1
    im2[im2 >= 0.5] = 1
    im1[im1 < 1] = 0
    im2[im2 < 1] = 0
    ### Get one-pixel boundary
    gt_onepix_mask = np.logical_xor(im1, ndimage.binary_erosion(im1).astype(im1.dtype))
    pred_onepix_mask = np.logical_xor(im2, ndimage.binary_erosion(im2).astype(im2.dtype))
    # pilBrr = im2 * 255.0
    # pilBImg = Image.fromarray(pilBrr)
    # pilBImg.show()
    ### Will return a tuple of (array([0, 1, 1]), array([1, 0, 1])) where array([0, 1, 1]) are the row indices and col indices, repectively.
    ### For example, the value gt_ones_px_coords[0][0]=0 and gt_ones_px_coords[1][0]=0 gives out the coords (in index form) of the 2D image in x,y (row, col) form.
    ### In this coord, there is a corresponding white pixel = 1 in the GT saliency map. The tuple's first and second arrays will always have the same length (obviously).
    gt_ones_px_coords = np.where(gt_onepix_mask == 1)
    pred_onepix_mask = np.where(pred_onepix_mask == 1)
    if len(gt_ones_px_coords[0]) == 0:
        print("gt_ones_px_coords has no white pixel boundary")
        exit()
    if len(pred_onepix_mask[0]) == 0:
        print("pred_onepix_mask has no white pixel boundary")
        exit()
    stacked_gt_whitepx_coords = np.vstack((gt_ones_px_coords[0], gt_ones_px_coords[1])) ### Stack everything into a (2, n) ndarray
    stacked_pred_whitepx_coords = np.vstack((pred_onepix_mask[0], pred_onepix_mask[1])) ### Stack everything into a (2, n) ndarray

    rho_px = 3 ### In BASNet paper. For a true positive to happen, dist between gt and pred pixel should be less than rho_px or less
    ### Compute relaxed precision = fraction of predicted boundary pixels within a range of ρ=3 pixels from ground truth boundary pixels
    relaxed_precTP = 0
    for idx_pixcoord in range(0, stacked_pred_whitepx_coords.shape[1]): ### Iterate all 2D pixels
        _, nearest_px_dist = knn_search(stacked_pred_whitepx_coords[:,idx_pixcoord].reshape((2,1)), stacked_gt_whitepx_coords, 1)
        if nearest_px_dist <= rho_px: relaxed_precTP += 1 ### compare distance of the nearest pixel
    relaxed_prec = relaxed_precTP / stacked_gt_whitepx_coords.shape[1]

    ### Compute relaxed recall = fraction of ground truth boundary pixels that are within ρ=3 pixels of predicted boundary pixels
    relaxed_recTP = 0
    for idx_pixcoord in range(0, stacked_gt_whitepx_coords.shape[1]): ### Iterate all 2D pixels
        _, nearest_px_dist = knn_search(stacked_gt_whitepx_coords[:,idx_pixcoord].reshape((2,1)), stacked_pred_whitepx_coords, 1)
        if nearest_px_dist <= rho_px: relaxed_recTP += 1 ### compare distance of the nearest pixel
    relaxed_rec = relaxed_recTP / stacked_pred_whitepx_coords.shape[1]

    ### Calculate final f-measure
    beta2 = 0.3
    if beta2 * relaxed_prec + relaxed_rec == 0: return 0
    fmeasure = ((1+beta2)* relaxed_prec * relaxed_rec) / (beta2 * relaxed_prec + relaxed_rec)
    return fmeasure

# imA = np.array(Image.open("./DUTS/DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00000003.png").convert("L"))
# imB = np.array(Image.open("./DUTS/DUTS-TE/DUTS-TE-STRUCT/ILSVRC2012_test_00000003.png").convert("L"))
# # precision, recall = getPRCurve(imA, imB)
# #prec, rec = getPRCurve(imA, imB)
# #fmeasure = getMaxFMeasure(prec, rec)
# #mae = getMAE(imA,imB)
# #print("prec: ", prec)
# #print("rec: ", rec)
# #print("fmeasure: ", fmeasure)
# #print("mae: ", mae)
# x=own_RelaxedFMeasure(imA,imB)
# print(x)
