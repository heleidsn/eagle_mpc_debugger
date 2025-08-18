#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib utilities for safe plotting with proper signal handling
"""

import matplotlib.pyplot as plt
import signal
import sys
import threading
import time

# Global variable to track if we should exit
exit_flag = False

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global exit_flag
    print("\nReceived interrupt signal. Exiting...")
    exit_flag = True
    plt.close('all')
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful exit"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def safe_show(block=False, pause_time=0.1):
    """
    Safely show matplotlib plots with proper event loop handling
    
    Args:
        block (bool): Whether to block the main thread
        pause_time (float): Time to pause after showing plot
    """
    plt.show(block=block)
    if not block:
        plt.pause(pause_time)

def keep_plots_alive():
    """
    Keep matplotlib plots alive until user closes them or interrupts
    
    This function should be called after showing plots to prevent
    the program from hanging when ROS nodes are terminated.
    """
    global exit_flag
    exit_flag = False
    
    print("Plots displayed. Press Ctrl+C to exit or close the plot windows.")
    
    try:
        while not exit_flag:
            plt.pause(0.1)  # Small pause to allow for interrupt handling
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        plt.close('all')
        print("Program terminated.")

def plot_with_safe_exit(plot_function, *args, **kwargs):
    """
    Execute a plotting function with safe exit handling
    
    Args:
        plot_function: Function that creates and shows plots
        *args: Arguments to pass to plot_function
        **kwargs: Keyword arguments to pass to plot_function
    """
    # Setup signal handlers
    setup_signal_handlers()
    
    # Execute the plotting function
    plot_function(*args, **kwargs)
    
    # Keep plots alive
    keep_plots_alive()

def create_safe_plot_context():
    """
    Create a context manager for safe plotting
    
    Usage:
        with create_safe_plot_context():
            plt.plot(x, y)
            plt.show(block=False)
    """
    class SafePlotContext:
        def __enter__(self):
            setup_signal_handlers()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:  # No exception occurred
                keep_plots_alive()
            else:
                plt.close('all')
    
    return SafePlotContext()

def configure_matplotlib_for_ros():
    """
    Configure matplotlib settings for better ROS integration
    """
    # Set backend to non-interactive if no display is available
    import os
    if 'DISPLAY' not in os.environ:
        plt.switch_backend('Agg')
        print("No display detected, using non-interactive backend")
    
    # Configure matplotlib for better performance
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Enable interactive mode for non-blocking plots
    plt.ion()

def save_and_show_plot(filename=None, dpi=300, bbox_inches='tight'):
    """
    Save plot to file and show it safely
    
    Args:
        filename (str): Filename to save plot (optional)
        dpi (int): DPI for saved image
        bbox_inches (str): Bounding box setting for saved image
    """
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to: {filename}")
    
    safe_show(block=False)

def cleanup_plots():
    """Clean up all matplotlib plots and close windows"""
    plt.close('all')
    plt.clf()
    plt.cla()

# Example usage functions
def example_safe_plot():
    """Example of how to use the safe plotting utilities"""
    def create_plots():
        # Create some example plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax1.plot(x, y, 'b-', label='sin(x)')
        ax1.set_title('Sine Wave')
        ax1.legend()
        ax1.grid(True)
        
        # Second plot
        ax2.scatter(x[::5], y[::5], c='red', s=50, alpha=0.7)
        ax2.set_title('Scatter Plot')
        ax2.grid(True)
        
        plt.tight_layout()
        safe_show(block=False)
    
    # Use the safe plotting context
    plot_with_safe_exit(create_plots)

if __name__ == "__main__":
    import numpy as np
    example_safe_plot()