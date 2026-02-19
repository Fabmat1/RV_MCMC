#!/usr/bin/env python3
"""
Plot periodogram for a given Gaia ID.

The script reads periodogram data from out/{gaia_id}/pgram.csv
and creates a plot of power vs period.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_periodogram(gaia_id):
    """
    Plot the periodogram for a given Gaia ID.
    
    Parameters
    ----------
    gaia_id : str
        The Gaia source ID
    """
    # Construct the path to the periodogram file
    pgram_file = Path(f"out/{gaia_id}/pgram.csv")
    
    # Check if the file exists
    if not pgram_file.exists():
        print(f"Error: File not found: {pgram_file}")
        sys.exit(1)
    
    # Load the data (period, power)
    try:
        data = np.loadtxt(pgram_file, delimiter=',')
        period = data[:, 0]
        power = data[:, 1]
    except Exception as e:
        print(f"Error reading file {pgram_file}: {e}")
        sys.exit(1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(period, power, 'b-', linewidth=0.8)
    
    # Labels and title
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title(f'Periodogram for Gaia ID: {gaia_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark the peak
    max_idx = np.argmax(power)
    max_period = period[max_idx]
    max_power = power[max_idx]
    ax.plot(max_period, max_power, 'r*', markersize=15, 
            label=f'Peak: P={max_period:.4f}, Power={max_power:.4f}')
    ax.legend()
    
    # Tight layout and show
    plt.tight_layout()
    
    # Save the figure
    output_file = Path(f"out/{gaia_id}/periodogram.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Periodogram saved to: {output_file}")
    
    # Display the plot
    plt.show()


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python plot_periodogram.py <gaia_id>")
        print("Example: python plot_periodogram.py 1234567890123456")
        sys.exit(1)
    
    gaia_id = sys.argv[1]
    plot_periodogram(gaia_id)


if __name__ == "__main__":
    main()