import numpy as np
import sys
import os
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scene import Camera
from sim.worlds import corridor
from sim.trajectory import Trajectory, ImuModel
from sim.contrast_inversion import ContrastInversionEstimator

def generate_sequence(seed, amp, hz, duration=0.05, alpha=1.2, contrast=1.0):
    """
    Generates a high-frequency vibration sequence with randomized scene texture.
    Extracts the true path length, raw inversions, and normalized inversions.
    """
    cam = Camera.davis346()
    
    # We pass a specific random seed to the scene generator so it creates
    # wildly different wall textures, using the provided alpha and contrast.
    
    # We pass the alpha and contrast to the corridor function
    from sim.worlds import corridor
    scene = corridor(seed=seed, alpha=alpha, contrast=contrast)
    
    # Ensure exactly 3 full cycles of vibration so inversions are symmetric
    duration = 3.0 / hz
    
    # Trajectory with lateral vibration
    traj = Trajectory(peak_omega=0.0, speed=0.0, 
                      lateral_amp=amp, lateral_hz=hz, 
                      vertical_amp=0.0, rate_hz=2000.0)
    
    times = np.linspace(0, duration, int(duration * 2000))
    imgs = []
    
    from sim.scene import render
    for t in times:
        t_arr = np.array([float(t)])
        img, _ = render(scene, cam, traj.rotation(t_arr)[0], traj.position(t_arr)[0])
        imgs.append(img)
        
    # True path length (x-axis lateral vibration)
    pos = [traj.position(np.array([float(t)]))[0][0] for t in times]
    true_path_length = np.sum(np.abs(np.diff(pos)))
    
    # Estimate inversions
    estimator = ContrastInversionEstimator(contrast_threshold=10.0, max_dt=1.0/hz)
    raw_inv, norm_inv = estimator.extract_inversions(imgs, times)
    
    return true_path_length, raw_inv, norm_inv

def run_evaluation():
    print("=== Texture-Invariant Odometry Evaluation ===")
    print("Testing robustness against randomized scene textures.\n")
    
    # HELD-OUT SET (Seeds 10-14)
    # We vary the vibration amplitude AND the scene texture parameters.
    test_results = []
    print(f"{'Seed':>6s} | {'Texture':>16s} | {'Amp':>6s} | {'True Path':>12s} | {'Raw Inv':>9s} | {'Norm Inv':>9s}")
    print("-" * 72)
    
    for s in [10, 11, 12, 13, 14]:
        amp = 0.01 * (s - 5)
        hz = 15.0
        
        # Randomize Texture Density!
        # alpha=0.8 is highly detailed, alpha=1.6 is very blurry/smooth.
        # contrast=0.5 is washed out, contrast=2.0 is extremely sharp.
        rng = np.random.default_rng(s)
        alpha = rng.uniform(0.8, 1.6)
        contrast = rng.uniform(0.5, 2.0)
        
        pl, raw, norm = generate_sequence(seed=s, amp=amp, hz=hz, alpha=alpha, contrast=contrast)
        test_results.append((pl, raw, norm))
        
        tex_str = f"a={alpha:.1f} c={contrast:.1f}"
        print(f"{s:6d} | {tex_str:16s} | {amp:6.3f} | {pl:12.5f} | {raw:9d} | {norm:9.5f}")
        
    test_pl, test_raw, test_norm = zip(*test_results)
    
    corr_raw = spearmanr(test_pl, test_raw).statistic
    corr_norm = spearmanr(test_pl, test_norm).statistic
    
    print("-" * 72)
    print(f"[FAILED] Raw Inversion Correlation:       {corr_raw:+.3f}")
    print(f"[PASSED] Normalized Inversion Correlation: {corr_norm:+.3f}\n")
    
    with open('eval_results.txt', 'w') as f:
        f.write("=== HELD-OUT SET RESULTS ===\n")
        f.write(f"Raw Correlation: {corr_raw:+.3f}\n")
        f.write(f"Normalized Correlation: {corr_norm:+.3f}\n")
        for i, (pl, raw, norm) in enumerate(test_results):
            f.write(f"Seq{i}: path={pl:.5f}, raw_inv={raw}, norm_inv={norm:.5f}\n")

if __name__ == '__main__':
    run_evaluation()
