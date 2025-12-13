import numpy as np
import matplotlib.pyplot as plt

# plot fused score
def plot_score_hist(y_true, score, title="Fused score histogram"):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    plt.figure(figsize=(10,5))
    plt.hist(score[y_true==0], bins=60, alpha=0.6, label="Normal")
    plt.hist(score[y_true==1], bins=60, alpha=0.6, label="Anomaly")
    plt.title(title)
    plt.xlabel("fused_score")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show() 

# Shade anomaly regions in plots
def shade_anoms(ax, y):
    
    in_seg = False
    s = 0
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            in_seg = True
            s = i
        if in_seg and (v == 0 or i == len(y)-1):
            e = i if v == 0 else i+1
            ax.axvspan(s, e, alpha=0.2, color='lightblue')
            in_seg = False

# plot with anomaly shading
def plot_monitoring_charts(d, ew, re, y, detector, title_suffix=""):
   
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Shewhart (MD)
    axes[0].plot(d, linewidth=1, color='blue')
    axes[0].axhline(detector.ucl_shewhart, linestyle="--", color='r', 
                   label=f'UCL={detector.ucl_shewhart:.2f}')
    shade_anoms(axes[0], y)
    axes[0].set_title(f"Shewhart Chart - Mahalanobis Distance {title_suffix}")
    axes[0].set_ylabel('MD Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # EWMA (MD)
    axes[1].plot(ew, linewidth=1, color='green')
    axes[1].axhline(detector.ucl_ewma, linestyle="--", color='r', 
                   label=f'UCL={detector.ucl_ewma:.2f}')
    shade_anoms(axes[1], y)
    axes[1].set_title(f"EWMA Chart (λ={detector.ewma_lambda}) {title_suffix}")
    axes[1].set_ylabel('EWMA Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Reconstruction Error
    if detector.use_re:
        axes[2].plot(re, linewidth=1, color='purple')
        axes[2].axhline(detector.ucl_re, linestyle="--", color='r', 
                       label=f'UCL={detector.ucl_re:.6f}')
        shade_anoms(axes[2], y)
        axes[2].set_title(f"Reconstruction Error Chart {title_suffix}")
        axes[2].set_ylabel('RE Score')
        axes[2].set_xlabel('Sample Index')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
