import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.fft import fft, fftfreq

df = pd.read_csv('shm.csv')
time_data = df['time'].values
velocity_data = df['velocity'].values
acceleration_data = df['acceleration'].values
displacement_data = df['displacement'].values

# Plot displacement against time
plt.plot(df['time'], df['displacement'])
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs Time')
plt.grid(True)
plt.show()


# Count zero crossings
def count_zero_crossings(displacement):
    zero_crossings = 0
    for i in range(1, len(displacement)):
        if displacement[i] * displacement[i - 1] < 0:
            zero_crossings += 1
    return zero_crossings


num_crossings = count_zero_crossings(df['displacement'])
print("Number of zero crossings:", num_crossings)

# Calculate potential energy
k = 1
PE = 0.5 * k * (df['displacement']) ** 2
# Calculate kinetic energy
mass = 1
KE = 0.5 * mass * (df['velocity']) ** 2
# Calculate total energy
TE = PE + KE
# Plot total energy vs time
plt.plot(df['time'], TE)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy vs Time')
plt.grid(True)
plt.show()


def euler_method(m, c, k, x0, v0, time_steps):
    # Simulates a damped mass-spring system using Euler's method.
    # time_steps: Array of time steps.

    dt = np.diff(time_steps)
    num_steps = len(time_steps)

    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    a = np.zeros(num_steps)
    x[0] = x0
    v[0] = v0

    for i in range(1, num_steps):
        a[i] = (-c * v[i - 1] - k * x[i - 1]) / m
        v[i] = v[i - 1] + a[i] * dt[i - 1]
        x[i] = x[i - 1] + v[i] * dt[i - 1]

    return time_steps, x, v, a  # Returns: Tuple of simulated time, displacement, velocity, and acceleration.


# Parameters
m = 1  # mass
c = 0.1  # damping coefficient
k = 1  # spring constant
x0 = 1  # initial displacement
v0 = 0  # initial velocity

# Simulate using Euler's method
simulated_time, simulated_displacement, simulated_velocity, simulated_acceleration = euler_method(m, c, k, x0, v0, time_data)

# Plot all three graphs in a single figure
plt.figure(figsize=(12, 8))

# Displacement
plt.subplot(3, 1, 1)
plt.plot(time_data, displacement_data, label='Real Displacement')
plt.plot(simulated_time, simulated_displacement, label='Simulated Displacement')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()

# Velocity
plt.subplot(3, 1, 2)
plt.plot(time_data, velocity_data, label='Real Velocity')
plt.plot(simulated_time, simulated_velocity, label='Simulated Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

# Acceleration
plt.subplot(3, 1, 3)
plt.plot(time_data, acceleration_data, label='Real Acceleration')
plt.plot(simulated_time, simulated_acceleration, label='Simulated Acceleration')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate mean squared error (MSE) using sklearn
mse_displacement = mean_squared_error(displacement_data, simulated_displacement)
mse_velocity = mean_squared_error(velocity_data, simulated_velocity)
mse_acceleration = mean_squared_error(acceleration_data, simulated_acceleration)

# Print MSE results
print("Mean Squared Error:")
print(f"  Displacement: {mse_displacement:.4f}")
print(f"  Velocity: {mse_velocity:.4f}")
print(f"  Acceleration: {mse_acceleration:.4f}")


def apply_fft_and_visualize(displacement_data):
    # Applies Fast Fourier Transform to the displacement data in a DataFrame.
    # Returns Tuple of frequency and FFT values.

    # Extract displacement data and sampling rate
    sampling_rate = 1 / (time_data[1] - time_data[0])  # Assuming uniform sampling

    # Perform FFT
    fft_result = fft(displacement_data)
    freqs = fftfreq(len(displacement_data), 1 / sampling_rate)

    # Keep positive frequencies only
    positive_freqs_mask = freqs >= 0
    freqs = freqs[positive_freqs_mask]
    fft_result = fft_result[positive_freqs_mask]

    # Magnitude of FFT results
    fft_magnitude = np.abs(fft_result)

    # Plot frequency spectrum
    plt.plot(freqs, fft_magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum of Displacement Data")
    plt.grid(True)
    plt.show()


apply_fft_and_visualize(displacement_data)
