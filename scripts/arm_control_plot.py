#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm Current vs Angle Relationship Plot
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import signal
import sys

# Set font for better display
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

def plot_arm_current_vs_angle():
    """Plot arm current vs angle relationship"""
    
    # Data points: angle(degrees), current(mA)
    angles = [90, 60, 45, 30, 0]
    currents = [-220, -200, -160, -110, -6]
    
    # Calculate cos(angle)
    angles_rad = np.array(angles) * np.pi / 180
    sin_angles = np.sin(angles_rad)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: current vs angle
    ax1.scatter(angles, currents, color='red', s=100, zorder=5, label='Measured Data')
    
    # Fit curve (polynomial fit)
    if len(angles) > 2:
        # Use 3rd order polynomial fit
        coeffs = np.polyfit(angles, currents, 3)
        poly = np.poly1d(coeffs)
        
        # Generate smooth fit curve
        angle_smooth = np.linspace(0, 90, 100)
        current_smooth = poly(angle_smooth)
        
        # Plot fit curve
        ax1.plot(angle_smooth, current_smooth, 'b-', linewidth=2, 
                label=f'Fit Curve (3rd Order Polynomial)', alpha=0.8)
    
    ax1.set_xlabel('Arm Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Current (mA)', fontsize=12)
    ax1.set_title('Arm Current vs Angle Relationship', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(-5, 95)
    ax1.set_ylim(-230, 10)
    
    # Right plot: current vs cos(angle)
    ax2.scatter(sin_angles, currents, color='green', s=100, zorder=5, label='Measured Data')
    
    # Linear fit cos(angle) vs current
    coeffs_cos = np.polyfit(sin_angles, currents, 1)
    poly_cos = np.poly1d(coeffs_cos)
    
    # Generate smooth fit curve
    cos_smooth = np.linspace(0, 1, 100)
    current_cos_smooth = poly_cos(cos_smooth)
    
    # Plot fit curve
    ax2.plot(cos_smooth, current_cos_smooth, 'orange', linewidth=2, 
             label=f'Linear Fit: y = {coeffs_cos[0]:.2f}x + {coeffs_cos[1]:.2f}', alpha=0.8)
    
    ax2.set_xlabel('cos(Angle)', fontsize=12)
    ax2.set_ylabel('Current (mA)', fontsize=12)
    ax2.set_title('Arm Current vs cos(Angle) Relationship', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-230, 10)
    
    # Add data point annotations
    for i, (angle, current, cos_angle) in enumerate(zip(angles, currents, sin_angles)):
        # Left plot annotation
        ax1.annotate(f'({angle}°, {current}mA)', 
                     xy=(angle, current), 
                     xytext=(10, 10), 
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Right plot annotation
        ax2.annotate(f'({angle}°, {current}mA)', 
                     xy=(cos_angle, current), 
                     xytext=(10, 10), 
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add statistical information
    if len(angles) > 1:
        correlation_angle = np.corrcoef(angles, currents)[0, 1]
        correlation_cos = np.corrcoef(sin_angles, currents)[0, 1]
        
        ax1.text(0.02, 0.98, f'Angle Correlation: {correlation_angle:.3f}', 
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        ax2.text(0.02, 0.98, f'cos(Angle) Correlation: {correlation_cos:.3f}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Show plot with non-blocking mode
    plt.show(block=False)
    plt.pause(0.1)  # Give a small pause to ensure plot is displayed
    
    # Print data statistics
    print("Data Statistics:")
    print(f"Angle Range: {min(angles)}° - {max(angles)}°")
    print(f"Current Range: {min(currents)}mA - {max(currents)}mA")
    print(f"cos(Angle) Range: {min(sin_angles):.3f} - {max(sin_angles):.3f}")
    print(f"Number of Data Points: {len(angles)}")
    
    if len(angles) > 1:
        print(f"Angle Correlation: {correlation_angle:.3f}")
        print(f"cos(Angle) Correlation: {correlation_cos:.3f}")
        
        # Compare correlations
        if abs(correlation_cos) > abs(correlation_angle):
            print("✓ cos(Angle) has stronger correlation with current")
        else:
            print("✗ Angle has stronger correlation with current")
        
        # Calculate trend
        if correlation_cos > 0.5:
            trend = "Positive correlation"
        elif correlation_cos < -0.5:
            trend = "Negative correlation"
        else:
            trend = "Weak or no correlation"
        print(f"cos(Angle) Trend: {trend}")
        
        # Print cos fit equation
        print(f"cos(Angle) Linear Fit Equation: Current = {coeffs_cos[0]:.2f} × cos(Angle) + {coeffs_cos[1]:.2f}")

def plot_with_different_fits():
    """Plot comparison of different fitting methods"""
    
    angles = [90, 60, 45, 30, 0]
    currents = [-220, -200, -160, -110, -6]
    
    # Calculate cos(angle)
    angles_rad = np.array(angles) * np.pi / 180
    sin_angles = np.cos(angles_rad)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top-left: Angle linear fit
    ax1.scatter(angles, currents, color='red', s=100, zorder=5, label='Measured Data')
    
    # Linear fit
    coeffs_linear = np.polyfit(angles, currents, 1)
    poly_linear = np.poly1d(coeffs_linear)
    angle_smooth = np.linspace(0, 90, 100)
    current_linear = poly_linear(angle_smooth)
    ax1.plot(angle_smooth, current_linear, 'b-', linewidth=2, 
             label=f'Linear Fit: y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}')
    
    ax1.set_xlabel('Arm Angle (degrees)')
    ax1.set_ylabel('Current (mA)')
    ax1.set_title('Angle Linear Fit')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-5, 95)
    ax1.set_ylim(-230, 10)
    
    # Top-right: Angle polynomial fit
    ax2.scatter(angles, currents, color='red', s=100, zorder=5, label='Measured Data')
    
    # 3rd order polynomial fit
    coeffs_poly = np.polyfit(angles, currents, 3)
    poly_poly = np.poly1d(coeffs_poly)
    current_poly = poly_poly(angle_smooth)
    ax2.plot(angle_smooth, current_poly, 'g-', linewidth=2, 
             label='3rd Order Polynomial Fit')
    
    ax2.set_xlabel('Arm Angle (degrees)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Angle Polynomial Fit')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-5, 95)
    ax2.set_ylim(-230, 10)
    
    # Bottom-left: cos(angle) linear fit
    ax3.scatter(sin_angles, currents, color='green', s=100, zorder=5, label='Measured Data')
    
    # cos(angle) linear fit
    coeffs_cos = np.polyfit(sin_angles, currents, 1)
    poly_cos = np.poly1d(coeffs_cos)
    cos_smooth = np.linspace(0, 1, 100)
    current_cos = poly_cos(cos_smooth)
    ax3.plot(cos_smooth, current_cos, 'orange', linewidth=2, 
             label=f'cos(Angle) Linear Fit: y = {coeffs_cos[0]:.2f}x + {coeffs_cos[1]:.2f}')
    
    ax3.set_xlabel('cos(Angle)')
    ax3.set_ylabel('Current (mA)')
    ax3.set_title('cos(Angle) Linear Fit')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-230, 10)
    
    # Bottom-right: cos(angle) polynomial fit
    ax4.scatter(sin_angles, currents, color='green', s=100, zorder=5, label='Measured Data')
    
    # cos(angle) polynomial fit
    coeffs_cos_poly = np.polyfit(sin_angles, currents, 3)
    poly_cos_poly = np.poly1d(coeffs_cos_poly)
    current_cos_poly = poly_cos_poly(cos_smooth)
    ax4.plot(cos_smooth, current_cos_poly, 'purple', linewidth=2, 
             label='cos(Angle) 3rd Order Polynomial Fit')
    
    ax4.set_xlabel('cos(Angle)')
    ax4.set_ylabel('Current (mA)')
    ax4.set_title('cos(Angle) Polynomial Fit')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-230, 10)
    
    plt.tight_layout()
    
    # Show plot with non-blocking mode
    plt.show(block=False)
    plt.pause(0.1)  # Give a small pause to ensure plot is displayed
    
    # Calculate fit quality
    try:
        from sklearn.metrics import r2_score
        
        # Calculate R² values
        r2_angle_linear = r2_score(currents, poly_linear(angles))
        r2_angle_poly = r2_score(currents, poly_poly(angles))
        r2_cos_linear = r2_score(currents, poly_cos(sin_angles))
        r2_cos_poly = r2_score(currents, poly_cos_poly(sin_angles))
        
        print("\nFit Quality Comparison (R² values):")
        print(f"Angle Linear Fit: {r2_angle_linear:.4f}")
        print(f"Angle Polynomial Fit: {r2_angle_poly:.4f}")
        print(f"cos(Angle) Linear Fit: {r2_cos_linear:.4f}")
        print(f"cos(Angle) Polynomial Fit: {r2_cos_poly:.4f}")
        
        # Find best fit
        fits = [
            ("Angle Linear Fit", r2_angle_linear),
            ("Angle Polynomial Fit", r2_angle_poly),
            ("cos(Angle) Linear Fit", r2_cos_linear),
            ("cos(Angle) Polynomial Fit", r2_cos_poly)
        ]
        
        best_fit = max(fits, key=lambda x: x[1])
        print(f"\nBest Fit Method: {best_fit[0]} (R² = {best_fit[1]:.4f})")
        
    except ImportError:
        print("\nsklearn not available, skipping R² calculation")

def main():
    """Main function with proper signal handling"""
    # Setup signal handlers
    setup_signal_handlers()
    
    print("Plotting arm current vs angle relationship...")
    plot_arm_current_vs_angle()
    
    print("\nPlotting comparison of different fitting methods...")
    plot_with_different_fits()
    
    print("\nPlots displayed. Press Ctrl+C to exit or close the plot windows.")
    
    # Keep the program running until user closes plots or interrupts
    try:
        while not exit_flag:
            plt.pause(0.1)  # Small pause to allow for interrupt handling
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        plt.close('all')
        print("Program terminated.")

if __name__ == "__main__":
    main()
