import numpy as np
from scipy.spatial import KDTree
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def normalize_depth(sdpt, pre_clip_max, lower_thresh=2, upper_thresh=98, min_max_dilate=0.2):
    val_dpt = sdpt[sdpt > 0.]
    
    if pre_clip_max > 0 and len(val_dpt) > 0:
        val_dpt = val_dpt.clip(0., pre_clip_max)

    if len(val_dpt) > 0:
        dpt_min = np.percentile(val_dpt, lower_thresh)
    else:
        dpt_min = 0.
    
    if len(val_dpt) > 0:
        dpt_max = np.percentile(val_dpt, upper_thresh)
    else:
        dpt_max = 0.

    if min_max_dilate > 0.0:
        assert min_max_dilate < 1.0
        dpt_max = dpt_max * (1 + min_max_dilate)
        dpt_min = dpt_min * (1 - min_max_dilate)

    if dpt_max - dpt_min < 1e-6: dpt_max = dpt_min + 2e-6

    sdpt = np.clip(sdpt, dpt_min, dpt_max)
    sdpt_norm = (sdpt - dpt_min) / (dpt_max - dpt_min)
    return sdpt_norm, dpt_min, dpt_max


def interp_depth(sdpt, k=3, w_dist=10.0, lb=0.):
    h, w  = sdpt.shape

    if (sdpt <= lb).all(): return np.ones((h, w)) * lb, np.zeros((h, w))
    
    # interpolation
    val_x, val_y = np.where(sdpt > lb)
    inval_x, inval_y = np.where(sdpt <= lb)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)

    if (sdpt != 0).sum() < k:
        k = (sdpt != 0).sum()

    tree = KDTree(val_pos)
    dists, inds = tree.query(inval_pos, k=k)
    dpt = np.copy(sdpt).reshape(-1)

    if k == 1:
        dpt[inval_x * w + inval_y] = sdpt.reshape(-1,)[val_pos[inds][..., 0] * w + val_pos[inds][..., 1]]
    else:
        dists = np.where(dists == 0, 1e-10, dists)
        weights = 1 / dists
        weights /= np.sum(weights, axis=1, keepdims=True)
        dpt = np.copy(sdpt).reshape(-1)
        nearest_vals = sdpt[val_x[inds], val_y[inds]]
        weighted_avg = np.sum(nearest_vals * weights, axis=1)
        dpt[inval_x * w + inval_y] = weighted_avg

    # compute distance map
    val_msk = sdpt > lb
    dist_map = cv2.distanceTransform((1-val_msk).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    dist_map = dist_map / np.sqrt(h**2 + w**2)
    dist_map = dist_map * w_dist

    return dpt.reshape(h, w), dist_map


def renorm_depth(dpt, d_min, d_max):
    return (d_max - d_min) * dpt + d_min


def align_depth(pred_dpt, ref_dpt):
    # align with RANSAC

    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    ransac = RANSACRegressor(max_trials=1000)
    model = make_pipeline(poly_features, ransac)

    mask = ref_dpt > 1e-8

    if mask.sum() < 10:
        print('no enough samples')
        return None, None
    
    gt_mask = ref_dpt[mask]
    pred_mask = pred_dpt[mask]
    if len(gt_mask.shape) == 1:
        gt_mask = gt_mask.reshape(-1, 1)
    if len(pred_mask.shape) == 1:
        pred_mask = pred_mask.reshape(-1, 1)
    
    model.fit(pred_mask, gt_mask)
    a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
    
    if a > 0:
        pred_metric = a * pred_dpt + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred_dpt * (gt_mean / pred_mean)
    
    return pred_metric