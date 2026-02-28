import numpy as np
from collections import deque
try:
    from ripser import ripser
    from persim import plot_diagrams
    HAS_TDA = True
except ImportError:
    HAS_TDA = False
    print("WARNING: 'ripser' or 'persim' not found. TDA will be mocked.")

class TopologicalBrain:
    def __init__(self, controller, max_buffer=2000):
        self.controller = controller
        self.buffer = deque(maxlen=max_buffer)

    def add_log(self, log):
        self.buffer.append(log)

    def analyze(self):
        if len(self.buffer) < 100: return
        # Only analyze RECENT behavior (last 500 points), not full history
        recent = list(self.buffer)[-500:]
        data = np.array(recent)
        
        print("\n--- 🧠 Topological Scan Initiated ---")
        
        positions = data[:, 0]
        velocities = data[:, 1]
        
        # LOOP DETECTION (H1)
        center_mask = (positions > -0.7) & (positions < -0.3) & (np.abs(velocities) < 0.03)
        loop_density = np.sum(center_mask) / len(data)
        
        if loop_density > 0.3:
            print(f"🧠 BRAIN: Persistent H1 Loop Detected (Density: {loop_density:.2f})")
            print("   -> Diagnosis: Agent is trapped in a gravity well oscillation.")
            self.controller.apply_decay_spike(loop_density)
        
        # Real TDA if available
        if HAS_TDA:
            try:
                point_cloud = data[:, [0, 1, 4]]  # [Pos, Vel, Delta]
                if len(point_cloud) > 200:
                    indices = np.random.choice(len(point_cloud), 200, replace=False)
                    point_cloud = point_cloud[indices]
                
                # Normalize to avoid inf/nan in distance matrix
                point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-8)
                
                results = ripser(point_cloud, maxdim=1)
                diagrams = results['dgms']
                
                if len(diagrams) > 1:
                    h1_dgm = diagrams[1]
                    persistences = h1_dgm[:, 1] - h1_dgm[:, 0]
                    valid = np.isfinite(persistences)
                    if np.any(valid):
                        max_pers = np.max(persistences[valid])
                        if max_pers > 0.5:
                            print(f"🧠 BRAIN: Ripser confirms H1 cycle (Persistence: {max_pers:.2f})")
                            if loop_density <= 0.3:  # Only spike if density didn't already
                                self.controller.apply_decay_spike(max_pers)
            except Exception as e:
                print(f"   Ripser warning: {e} (using heuristic detection)")
        
        # VOID DETECTION (H2)
        hist, xedges, yedges = np.histogram2d(positions, velocities, bins=10, 
                                              range=[[-1.2, 0.6], [-0.07, 0.07]])
        
        right_side_bins = hist[7:, :]
        if np.sum(right_side_bins) < 10:
            void_pos = 0.45
            void_vel = 0.04
            print(f"🧠 BRAIN: Knowledge Void Detected (H2) at Goal Region")
            self.controller.inject_attractor([void_pos, void_vel])
