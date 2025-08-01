import matplotlib.pyplot as plt

def plot_vehicle_events(file_name, vehicle_events, save_path=None):
    """
    Plot vehicle detection events as horizontal bars over time.

    Args:
        file_name (str): Name of the file being visualized
        vehicle_events (list of tuples): [(enter_time, leave_time, set(vehicle_types))]
        save_path (str, optional): If provided, saves the plot to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, (start, end, types) in enumerate(vehicle_events):
        ax.plot([start, end], [idx, idx], linewidth=6, label=", ".join(types) if idx == 0 else "", color='tab:blue')
        # ax.text((start + end) / 2, idx + 0.2, f"{','.join(types)}", ha='center', fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vehicle Index")
    ax.set_title(f"Vehicle Detections — {file_name}")
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

