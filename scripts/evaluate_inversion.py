import numpy as np
import sys
import os
from scipy.stats import spearmanr

# Add parent directory to path so we can import sim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scene import Camera
from sim.worlds import corridor
from sim.trajectory import Trajectory, ImuModel
from sim.contrast_inversion import ContrastInversionEstimator

def generate_sequence(seed, amp, hz, duration=0.05):
    """
    Generates a high-frequency vibration sequence and extracts
    the true path length and the number of contrast inversions.
    """
    cam = Camera.davis346()
    scene = corridor(seed=seed)
    
    # Trajectory with specific vibration amplitude
    traj = Trajectory(peak_omega=0.0, speed=0.0, 
                      lateral_amp=amp, lateral_hz=hz, 
                      vertical_amp=0.0, rate_hz=2000.0)
    
    imu = ImuModel(rate_hz=2000.0, seed=seed)
    imu_data = imu.sample(traj, duration)
    
    # Render images at high framerate to simulate continuous intensity
    times = np.linspace(0, duration, int(duration * 2000))
    imgs = []
    
    from sim.scene import render
    for t in times:
        t_arr = np.array([float(t)])
        img, _ = render(scene, cam, traj.rotation(t_arr)[0], traj.position(t_arr)[0])
        imgs.append(img)
        
    # True path length
    pos = [traj.position(np.array([float(t)]))[0][0] for t in times] # x-axis lateral vibration
    true_path_length = np.sum(np.abs(np.diff(pos)))
    
    # Estimate inversions
    estimator = ContrastInversionEstimator(contrast_threshold=10.0, max_dt=1.0/hz)
    inversions = estimator.extract_inversions(imgs, times)
    
    return true_path_length, inversions

def run_evaluation():
    print("=== Zero-Leakage Evaluation: Contrast Inversion Odometry ===")
    print("Generating simulated data on the fly. No external datasets used.\n")
    
    # DEV SET (Seeds 1-3)
    # Used to tune the threshold and dt.
    dev_results = []
    for s in [1, 2, 3]:
        amp = 0.05 * s
        pl, inv = generate_sequence(seed=s, amp=amp, hz=20.0)
        dev_results.append((pl, inv))
        
    dev_pl, dev_inv = zip(*dev_results)
    dev_corr = spearmanr(dev_pl, dev_inv).statistic
    print(f"[DEV SET] Spearman Rank Correlation (Path Length vs Inversions): {dev_corr:+.3f}")
    
    # HELD-OUT SET (Seeds 10-14)
    # Touched EXACTLY ONCE for the final paper numbers.
    test_results = []
    print("\nRunning Held-Out Set (N=5):")
    print(f"{'Seed':>6s} | {'Vib Amp':>10s} | {'True Path (m)':>15s} | {'Inversions':>12s}")
    print("-" * 52)
    
    for s in [10, 11, 12, 13, 14]:
        amp = 0.01 * (s - 5) # Varying amplitude
        hz = 10.0 + s        # Varying frequency
        pl, inv = generate_sequence(seed=s, amp=amp, hz=hz)
        test_results.append((pl, inv))
        print(f"{s:6d} | {amp:10.3f} | {pl:15.5f} | {inv:12d}")
        
    test_pl, test_inv = zip(*test_results)
    test_corr = spearmanr(test_pl, test_inv).statistic
    print("-" * 52)
    print(f"!! [HELD-OUT SET] Spearman Rank Correlation: {test_corr:+.3f} !!\n")
    
    # Save results for paper inclusion
    with open('eval_results.txt', 'w') as f:
        f.write("=== HELD-OUT SET RESULTS ===\n")
        f.write(f"Correlation: {test_corr:+.3f}\n")
        for i, (pl, inv) in enumerate(test_results):
            f.write(f"Seq{i}: path={pl:.5f}, inv={inv}\n")
            
    print("Evaluation complete. Results saved to eval_results.txt.")

if __name__ == '__main__':
    run_evaluation()
