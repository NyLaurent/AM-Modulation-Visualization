import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# =============================================================================
# PARAMETERS DEFINITION
# =============================================================================
E_m = 100           # Baseband signal amplitude (V)
E_c = 150           # Carrier signal amplitude (V)
omega_m = 200       # Baseband angular frequency (rad/s)
omega_c = 3000      # Carrier angular frequency (rad/s)

# Calculate frequencies in Hz
f_m = omega_m / (2 * np.pi)   # Baseband frequency = 31.83 Hz
f_c = omega_c / (2 * np.pi)   # Carrier frequency = 477.46 Hz
modulation_index = E_m / E_c  # Modulation index = 0.667

# Time parameters
duration = 0.1      # Total simulation time (seconds)
samples = 1000      # Number of samples
t = np.linspace(0, duration, samples)  # Time array

# =============================================================================
# SIGNAL GENERATION
# =============================================================================
# Baseband signal (message signal)
baseband_signal = E_m * np.sin(omega_m * t)

# Carrier signal
carrier_signal = E_c * np.sin(omega_c * t)

# AM Modulated signal: e_AM(t) = [E_c + e_m(t)] * sin(ω_c t)
modulated_signal = (E_c + baseband_signal) * np.sin(omega_c * t)

# Envelope for visualization
upper_envelope = E_c + baseband_signal
lower_envelope = -(E_c + baseband_signal)  # Negative for lower envelope

# =============================================================================
# PLOT 1: TIME DOMAIN SIGNALS
# =============================================================================
plt.figure(figsize=(12, 10))

# Subplot 1: Baseband Signal
plt.subplot(3, 1, 1)
plt.plot(t * 1000, baseband_signal, 'b', linewidth=2, label='Baseband Signal')
plt.ylabel('Amplitude (V)')
plt.title(f'Baseband Signal: $e_m = 100\\sin(200t)$ | Frequency = {f_m:.2f} Hz')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 100)  # Show first 100ms

# Subplot 2: Carrier Signal
plt.subplot(3, 1, 2)
plt.plot(t * 1000, carrier_signal, 'g', linewidth=1, label='Carrier Signal')
plt.ylabel('Amplitude (V)')
plt.title(f'Carrier Signal: $e_c = 150\\sin(3000t)$ | Frequency = {f_c:.2f} Hz')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 100)

# Subplot 3: AM Modulated Signal
plt.subplot(3, 1, 3)
plt.plot(t * 1000, modulated_signal, 'r', linewidth=1, alpha=0.7, label='AM Modulated Signal')
plt.plot(t * 1000, upper_envelope, 'k--', linewidth=1.5, label='Envelope')
plt.plot(t * 1000, lower_envelope, 'k--', linewidth=1.5)
# Fill between envelopes for better visualization
plt.fill_between(t * 1000, upper_envelope, lower_envelope, alpha=0.1, color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (V)')
plt.title(f'AM Modulated Signal | Modulation Index = {modulation_index:.3f}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 100)

plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 2: ZOOMED VIEW FOR DETAIL
# =============================================================================
# Create zoomed view for first 20ms to see modulation details
zoom_duration = 0.02  # 20ms for detailed view
zoom_mask = t <= zoom_duration

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t[zoom_mask] * 1000, baseband_signal[zoom_mask], 'b', linewidth=2)
plt.ylabel('Amplitude (V)')
plt.title('Baseband Signal (Zoomed View)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

plt.subplot(3, 1, 2)
plt.plot(t[zoom_mask] * 1000, carrier_signal[zoom_mask], 'g', linewidth=1)
plt.ylabel('Amplitude (V)')
plt.title('Carrier Signal (Zoomed View) - High Frequency Detail')
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

plt.subplot(3, 1, 3)
plt.plot(t[zoom_mask] * 1000, modulated_signal[zoom_mask], 'r', linewidth=1, alpha=0.7)
plt.plot(t[zoom_mask] * 1000, upper_envelope[zoom_mask], 'k--', linewidth=1.5)
plt.plot(t[zoom_mask] * 1000, lower_envelope[zoom_mask], 'k--', linewidth=1.5)
plt.fill_between(t[zoom_mask] * 1000, upper_envelope[zoom_mask], lower_envelope[zoom_mask], 
                 alpha=0.1, color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (V)')
plt.title('AM Modulated Signal (Zoomed View) - Showing Envelope Clearly')
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

plt.tight_layout()
plt.show()

# =============================================================================
# PLOT 3: FREQUENCY SPECTRUM ANALYSIS
# =============================================================================
# Calculate FFT for frequency domain analysis
N = len(t)                          # Number of samples
T = t[1] - t[0]                     # Sampling interval
frequencies = fftfreq(N, T)[:N//2]  # Frequency array (only positive frequencies)

# Compute FFT of modulated signal
modulated_fft = fft(modulated_signal)
modulated_magnitude = (2.0 / N) * np.abs(modulated_fft[:N//2])  # Normalized magnitude

plt.figure(figsize=(12, 6))

# Plot frequency spectrum
plt.plot(frequencies, modulated_magnitude, 'purple', linewidth=2, label='AM Spectrum')

# Mark important frequencies
plt.axvline(x=f_c, color='red', linestyle='--', linewidth=2, 
            label=f'Carrier: {f_c:.1f} Hz')
plt.axvline(x=f_c + f_m, color='green', linestyle='--', linewidth=2, 
            label=f'Upper Sideband: {f_c + f_m:.1f} Hz')
plt.axvline(x=f_c - f_m, color='green', linestyle='--', linewidth=2, 
            label=f'Lower Sideband: {f_c - f_m:.1f} Hz')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of AM Modulated Signal')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)  # Focus on relevant frequency range
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# PRINT SUMMARY RESULTS
# =============================================================================
print("=" * 60)
print("AM MODULATION ANALYSIS RESULTS")
print("=" * 60)
print(f"Baseband Signal Frequency:  f_m  = {f_m:.2f} Hz")
print(f"Carrier Wave Frequency:     f_c  = {f_c:.2f} Hz")
print(f"Modulation Index:           m    = {modulation_index:.3f}")
print("\nSignal Equations:")
print(f"Baseband:    e_m(t) = 100·sin(200t)")
print(f"Carrier:     e_c(t) = 150·sin(3000t)")
print(f"Modulated:   e_AM(t) = [150 + 100·sin(200t)]·sin(3000t)")
print("\nFrequency Components:")
print(f"Carrier:     {f_c:.1f} Hz")
print(f"Upper Sideband: {f_c + f_m:.1f} Hz")
print(f"Lower Sideband: {f_c - f_m:.1f} Hz")
print("=" * 60)