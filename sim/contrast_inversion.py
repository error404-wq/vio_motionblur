import numpy as np
from scipy.signal import correlate

class ContrastInversionEstimator:
    """
    Estimates the sub-pixel scale of high-frequency camera vibration 
    by counting contrast inversion pairs in the event stream.
    """
    def __init__(self, contrast_threshold=5.0, max_dt=0.010):
        self.C = contrast_threshold
        self.max_dt = max_dt

    def extract_inversions(self, imgs, times):
        """
        Given a sequence of densely rendered images (simulating the continuous
        intensity field), extract the total number of inversion pairs.
        
        An inversion pair is a pixel that fires a positive event and then a 
        negative event (or vice versa) within max_dt.
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
            
            # Positive inversion: previous was negative, now positive
            pos_inv = pos_events & (event_state == -1) & valid_dt
            # Negative inversion: previous was positive, now negative
            neg_inv = neg_events & (event_state == 1) & valid_dt
            
            total_inversions += np.sum(pos_inv) + np.sum(neg_inv)
            
            # Update state for pixels that fired
            event_state[pos_events] = 1
            event_state[neg_events] = -1
            last_event_time[pos_events | neg_events] = t
            
            # Update reference image for pixels that fired
            ref_img[pos_events] += self.C
            ref_img[neg_events] -= self.C
            
        return total_inversions

    def estimate_scale(self, inversions, imu_data, dt):
        """
        The total number of inversions is proportional to the total sub-pixel
        path length of the vibration. We use the IMU to estimate the shape,
        and the inversions to lock the scale.
        """
        # Double integrate IMU accel to get relative path length shape
        accel_y = imu_data['accel'][:, 1] - np.mean(imu_data['accel'][:, 1]) # remove gravity/bias
        vel_y = np.cumsum(accel_y) * dt
        pos_y = np.cumsum(vel_y) * dt
        
        # Total path length of vibration from IMU
        path_length_shape = np.sum(np.abs(np.diff(pos_y)))
        
        # Scale is proportional to inversions / path_length_shape
        # (In a real system, this constant depends on scene texture, but 
        # the correlation between inversion density and path length is the core signal)
        return inversions / max(1e-6, path_length_shape)
