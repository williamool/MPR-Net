"""
Explicit Motion Modeling (EMM) -- Sec. III-B-1 of the paper.

Given a three-frame clip {I_{t-1}, I_t, I_{t+1}} the explicit motion prior M_t is built by

  1. local contrast suppression            I~ = max(I - alpha * blur(I), 0)
  2. background suppression                grid-point KLT tracking + RANSAC homography H_{j->t},
                                           warp I~_j -> I^_{j->t}, valid-region mask V_{j->t}
                                           D^c_{j->t} = [I~_t - I^_{j->t}]_+ (.) V_{j->t}        (Eq. 1-2)
  3. dual differential fusion              D^f = lambda * D^c + (1 - lambda) * D^p,
                                           D^p_{j->t} = [I~_t - I~_j]_+                          (Eq. 3)
  4. bidirectional aggregation             D_t = 0.5 * (D^f_{t-1->t} + D^f_{t+1->t})              (Eq. 4)
  5. spatial response refinement           M_t = gamma * MexicanHat(D_t) + (1 - gamma) * D_t       (Eq. 5)

Dataset presets (`PRESETS`) reproduce the maps used in the paper:
  lambda = 0.5 for DAUB-R / IRDST-H, lambda = 1.0 (compensated differencing only) for IRSTD-UAV, gamma = 0.3.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass
class EMMConfig:
    # local contrast suppression
    contrast_alpha: float = 0.3
    contrast_blur_ksize: int = 3
    contrast_sigma: float = 1.0
    # KLT grid tracking (on a 4x up-sampled frame)
    upsample: int = 4
    grid_w: int = 64
    grid_h: int = 64
    klt_win: int = 15
    klt_max_level: int = 3
    klt_epsilon: float = 0.001
    klt_max_iter: int = 50
    motion_distance_threshold: float | None = 30.0  # drop tracks whose displacement is larger than this (px, up-sampled)
    min_tracking_points: int = 15
    ransac_threshold: float = 1.0
    # dual differential fusion (Eq. 3) and spatial refinement (Eq. 5)
    lam: float = 0.5     # weight of the compensated difference; 1.0 = compensated only, 0.0 = raw only
    gamma: float = 0.3   # weight of the Mexican-hat filtered response
    # optional resize of the input frames (W, H); None keeps the native resolution
    target_size: tuple[int, int] | None = None


PRESETS = {
    "DAUB": EMMConfig(contrast_alpha=0.3, grid_w=64, grid_h=64, klt_epsilon=0.001, klt_max_iter=50,
                      motion_distance_threshold=30.0, lam=0.5, gamma=0.3),
    "IRDST-H": EMMConfig(contrast_alpha=0.6, grid_w=90, grid_h=60, klt_epsilon=0.003, klt_max_iter=30,
                         motion_distance_threshold=10.0, lam=0.5, gamma=0.3),
    "IRSTD-UAV": EMMConfig(contrast_alpha=0.3, grid_w=80, grid_h=64, klt_epsilon=0.003, klt_max_iter=30,
                           motion_distance_threshold=10.0, lam=1.0, gamma=0.3, target_size=(640, 512)),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.copy()


def local_contrast_enhance(img_gray: np.ndarray, alpha: float = 0.3, blur_ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """C = max(I - alpha * blur(I), 0), returned as uint8."""
    I = img_gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(I, (blur_ksize, blur_ksize), sigma)
    C = np.maximum(I - alpha * blur, 0.0)
    return np.clip(C * 255.0, 0, 255).astype(np.uint8)


def mexican_hat_kernel(size: int = 9, sigma1: float = 1.0, sigma2: float = 2.0, wsurr: float = 0.5) -> np.ndarray:
    """Zero-mean difference-of-Gaussians (Mexican-hat) kernel."""
    assert size % 2 == 1, "kernel size must be odd"
    k = size // 2
    xs = np.arange(-k, k + 1, dtype=np.float32)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    g1 = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma1 ** 2 + 1e-6))
    g1 /= g1.sum() + 1e-6
    g2 = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma2 ** 2 + 1e-6))
    g2 /= g2.sum() + 1e-6
    dog = g1 - wsurr * g2
    dog -= dog.mean()
    return dog


def spatial_filter(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Mexican-hat filtering F(.) that suppresses diffuse residuals and enhances compact responses."""
    if kernel is None:
        kernel = mexican_hat_kernel(size=9, sigma1=1.0, sigma2=2.0, wsurr=0.5)
    x = img.astype(np.float32)
    y = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
    y = np.maximum(y, 0.0)
    if y.max() > 0:
        y = y / (y.max() + 1e-6) * (img.max() + 1e-6)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# background suppression (Eq. 1-2)
# --------------------------------------------------------------------------- #
def motion_compensate(frame_nb_gray: np.ndarray, frame_t_gray: np.ndarray, cfg: EMMConfig):
    """
    Estimate the dominant background motion between a neighbouring frame and the current frame with
    grid-point KLT tracking + RANSAC homography, and warp the neighbouring frame onto frame t.

    Returns (compensated, invalid_mask, H) where invalid_mask == 255 marks pixels outside the warped frame.
    """
    lk_params = dict(
        winSize=(cfg.klt_win, cfg.klt_win),
        maxLevel=cfg.klt_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.klt_max_iter, cfg.klt_epsilon),
    )

    height, width = frame_t_gray.shape[:2]
    s = cfg.upsample
    nb_up = cv2.resize(frame_nb_gray, (width * s, height * s), interpolation=cv2.INTER_CUBIC)
    t_up = cv2.resize(frame_t_gray, (width * s, height * s), interpolation=cv2.INTER_CUBIC)

    # uniformly sampled grid points p_i^j on the neighbouring frame
    grid_num_w = int(t_up.shape[1] / cfg.grid_w - 1)
    grid_num_h = int(t_up.shape[0] / cfg.grid_h - 1)
    pts = [(np.float32(i * cfg.grid_w + cfg.grid_w / 2.0), np.float32(j * cfg.grid_h + cfg.grid_h / 2.0))
           for i in range(grid_num_w) for j in range(grid_num_h)]
    pts_prev = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    # KLT tracks the grid points onto the current frame -> q_i^t
    pts_cur, st, _ = cv2.calcOpticalFlowPyrLK(nb_up, t_up, pts_prev, None, **lk_params)
    good_new = pts_cur[st == 1]
    good_old = pts_prev[st == 1]

    if len(good_old) < cfg.min_tracking_points:
        H = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]], dtype=np.float32)
    else:
        H, _ = cv2.findHomography(good_new, good_old, cv2.RANSAC, cfg.ransac_threshold)
        if H is None:
            H = np.eye(3, dtype=np.float32)

    # bring the homography back from the up-sampled to the original coordinate system
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
    H = np.linalg.inv(S) @ H @ S

    compensated = cv2.warpPerspective(frame_nb_gray, H, (width, height),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

    # valid-region mask V_{j->t}: pixels that fall outside the warped frame are invalid
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.eye(3, dtype=np.float32)
    vertex_t = np.asarray(cv2.perspectiveTransform(vertex, H_inv), dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame_nb_gray.shape[:2], dtype="uint8")
    cv2.polylines(im, vertex_t, 1, 255)
    cv2.fillPoly(im, vertex_t, 255)
    invalid_mask = 255 - im

    return compensated, invalid_mask, H


def compensated_diff(frame_t_gray: np.ndarray, compensated_gray: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """D^c = [I~_t - I^_{j->t}]_+ (.) V   (Eq. 2)"""
    diff = frame_t_gray.astype(np.float32) - compensated_gray.astype(np.float32)
    return np.maximum(diff, 0.0) * (invalid_mask == 0).astype(np.float32)


# --------------------------------------------------------------------------- #
# one-sided motion difference (Eq. 1-3) and the final motion prior (Eq. 4-5)
# --------------------------------------------------------------------------- #
def pair_motion_diff(frame_neighbor: np.ndarray, frame_t: np.ndarray, cfg: EMMConfig, lam: float | None = None):
    """
    D^f_{j->t} = lam * D^c_{j->t} + (1 - lam) * D^p_{j->t}.
    Returns (D^f, I~_t).
    """
    lam = cfg.lam if lam is None else lam
    nb_gray = local_contrast_enhance(to_gray(frame_neighbor).astype(np.uint8),
                                     cfg.contrast_alpha, cfg.contrast_blur_ksize, cfg.contrast_sigma)
    t_gray = local_contrast_enhance(to_gray(frame_t).astype(np.uint8),
                                    cfg.contrast_alpha, cfg.contrast_blur_ksize, cfg.contrast_sigma)

    # raw positive differencing D^p (kept when lam < 1)
    pure_diff = np.maximum(t_gray.astype(np.float32) - nb_gray.astype(np.float32), 0.0)
    if lam <= 0.0:
        return pure_diff, t_gray

    nb_comp, invalid_mask, _ = motion_compensate(nb_gray, t_gray, cfg)
    comp_diff = compensated_diff(t_gray, nb_comp, invalid_mask)
    if lam >= 1.0:
        return comp_diff, t_gray

    return lam * comp_diff + (1.0 - lam) * pure_diff, t_gray


def motion_prior_three_frames(frame_tm1: np.ndarray, frame_t: np.ndarray, frame_tp1: np.ndarray,
                              cfg: EMMConfig, lam: float | None = None):
    """
    Explicit motion prior M_t for the clip (t-1, t, t+1).
    Returns (M_t, D^f_{t-1->t}, D^f_{t+1->t}, I~_t) as float32 arrays in [0, 255].
    """
    if cfg.target_size is not None:
        w, h = cfg.target_size
        frame_tm1, frame_t, frame_tp1 = (f if f.shape[1] == w and f.shape[0] == h
                                         else cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                                         for f in (frame_tm1, frame_t, frame_tp1))

    d_prev, gray_t = pair_motion_diff(frame_tm1, frame_t, cfg, lam)
    d_next, _ = pair_motion_diff(frame_tp1, frame_t, cfg, lam)

    d_t = 0.5 * (d_prev + d_next)                                    # Eq. 4
    m_t = cfg.gamma * spatial_filter(d_t) + (1.0 - cfg.gamma) * d_t  # Eq. 5
    return m_t.astype(np.float32), d_prev.astype(np.float32), d_next.astype(np.float32), gray_t


def motion_prior_to_uint8(m_t: np.ndarray) -> np.ndarray:
    return np.clip(m_t, 0, 255).astype(np.uint8)


__all__ = ["EMMConfig", "PRESETS", "replace", "motion_compensate", "pair_motion_diff",
           "motion_prior_three_frames", "motion_prior_to_uint8"]
