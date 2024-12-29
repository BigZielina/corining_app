import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    figs = []
    
    for _ in range(5):
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        ax.plot(x, y)
        ax.set_title('Random Plot')
        figs.append(fig)
    
    return tuple(figs)
