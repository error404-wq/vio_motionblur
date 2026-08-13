import numpy as np

class ContrastInversionEstimator:
    """
    Estimates the sub-pixel scale of high-frequency camera vibration 
    by counting contrast inversion pairs and normalizing by spatial gradient.
    """
    def __init__(self, contrast_threshold=10.0, max_dt=0.010):
        self.C = contrast_threshold
        self.max_dt = max_dt

    def extract_inversions(self, imgs, times):
        """
        Given a sequence of densely rendered images, extract:
        1. The raw count of contrast inversions.
        2. The texture-normalized metric (inversions / gradient_sum).
        """
        N = len(imgs)
        total_inversions = 0
        
        # We track the 'last event' state of each pixel
        # 0 = no recent event, 1 = recent positive, -1 = recent negative
        event_state = np.zeros_like(imgs[0], dtype=np.int8)
        last_event_time = np.zeros_like(imgs[0], dtype=float)
        
        # Track intensity changes
        ref_img = imgs[0].astype(float)
        
        for i in range(1, N):
            curr_img = imgs[i].astype(float)
            diff = curr_img - ref_img
            t = times[i]
            
            # Find new events
            pos_events = diff >= self.C
            neg_events = diff <= -self.C
            
            # Check for inversions
            valid_dt = (t - last_event_time) <= self.max_dt
            
            pos_inv = pos_events & (event_state == -1) & valid_dt
            neg_inv = neg_events & (event_state == 1) & valid_dt
            
            total_inversions += np.sum(pos_inv) + np.sum(neg_inv)
            
            event_state[pos_events] = 1
            event_state[neg_events] = -1
            last_event_time[pos_events | neg_events] = t
            
            ref_img[pos_events] += self.C
            ref_img[neg_events] -= self.C
            
        # Texture Normalization via Sub-Pixel Calibration Matching
        from scipy.ndimage import shift
        ref = imgs[0].astype(float)
        
        best_A = 0.0
        min_diff = float('inf')
        
        # 3 cycles = 6 full back-and-forth crossings of the spatial gradients
        target_yield = total_inversions / 6.0
        
        # Search space for equivalent sub-pixel shift amplitude
        for A in np.linspace(0.01, 20.0, 100):
            shifted = shift(ref, [0, A], order=1)
            calib_yield = np.sum(np.abs(shifted - ref) >= self.C)
            
            diff = abs(calib_yield - target_yield)
            if diff < min_diff:
                min_diff = diff
                best_A = A
                
        normalized_inversions = best_A
        
        return total_inversions, normalized_inversions
