import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a double Gaussian function
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    gaussian1 = a1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    gaussian2 = a2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return gaussian1 + gaussian2



### MEASURMENT 1
distances_1 = []
for i in range(16):
    data = pd.read_csv("2_analysis_22\distances_{}.csv".format(i))
    
    x = data['Distance_(m)']  # Distance
    y = data['Gray_Value']  # Gray Value

    # Initial guess for the parameters (a1, mu1, sigma1, a2, mu2, sigma2)
    initial_guess = [
        max(y), np.percentile(x,40), np.std(x)/2,  # First Gaussian
        max(y), np.percentile(x,60), np.std(x)/2  # Second Gaussian
    ]

    # Fit the double Gaussian
    params, _ = curve_fit(double_gaussian, x, y, p0=initial_guess, maxfev=100000)

    # Extract the fitted parameters
    a1, mu1, sigma1, a2, mu2, sigma2 = params

    # Calculate the distance between the peaks
    peak_distance = abs(mu2 - mu1)
    distances_1.append(peak_distance)
    # Plot the data and the fit
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = double_gaussian(x_fit, *params)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", color="blue")
    plt.plot(x_fit, y_fit, label="Double Gaussian Fit".format(i), color="black")
    plt.axvline(mu1, color="darkviolet", linestyle="--", label=f"Peak 1: {mu1:.2f}")
    plt.axvline(mu2, color="darkviolet", linestyle="--", label=f"Peak 2: {mu2:.2f}")
    plt.title("Double Gaussian Fit {}".format(i))
    plt.xlabel(r"Distance ($\mu$m)")
    plt.ylabel("Gray Value")
    plt.xlim(min(x), max(x))
    plt.legend()
    plt.grid()
    plt.savefig("images\measurement_1_{}.eps".format(i), format="eps")
    plt.close()




### MEASURMENT 2
distances_2 = []
for i in range(14):
    data = pd.read_csv("2_analysis_19\distances_{}.csv".format(i))
    
    x = data['Distance_(m)']  # Distance
    y = data['Gray_Value']  # Gray Value

    # Initial guess for the parameters (a1, mu1, sigma1, a2, mu2, sigma2)
    initial_guess = [
        max(y), np.percentile(x,40), np.std(x)/2,  # First Gaussian
        max(y), np.percentile(x,60), np.std(x)/2  # Second Gaussian
    ]

    # Fit the double Gaussian
    params, _ = curve_fit(double_gaussian, x, y, p0=initial_guess, maxfev=100000)

    # Extract the fitted parameters
    a1, mu1, sigma1, a2, mu2, sigma2 = params

    # Calculate the distance between the peaks
    peak_distance = abs(mu2 - mu1)
    distances_2.append(peak_distance)
    # Plot the data and the fit
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = double_gaussian(x_fit, *params)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", color="blue")
    plt.plot(x_fit, y_fit, label="Double Gaussian Fit".format(i), color="black")
    plt.axvline(mu1, color="darkviolet", linestyle="--", label=f"Peak 1: {mu1:.2f}")
    plt.axvline(mu2, color="darkviolet", linestyle="--", label=f"Peak 2: {mu2:.2f}")
    plt.title("Double Gaussian Fit {}".format(i))
    plt.xlabel(r"Distance ($\mu$m)")
    plt.ylabel("Gray Value")
    plt.xlim(min(x), max(x))
    plt.legend()
    plt.grid()
    plt.savefig("images\measurement_2_{}.eps".format(i), format="eps")
    plt.close()

plt.scatter(range(len(distances_1)), distances_1, color='royalblue', label='measurement 1')
plt.scatter(range(len(distances_1), len(distances_1) + len(distances_2)), distances_2, color='navy', label='measurement 2')
plt.xlim(0, len(distances_1) + len(distances_2) - 1)
plt.ylim(0, 0.20)
plt.ylabel(r"Distance ($\mu$m)")
plt.xlabel("measurement")
plt.legend()
plt.grid()
plt.savefig("images\d_1_and_2_with_false.eps", format="eps")
plt.show()

# looking at the figure we can decide to remove measurement 1,4 and 7 of the second measurement. 
indices_to_remove = {0, 4, 6}
distances_2 = [item for i, item in enumerate(distances_2) if i not in indices_to_remove]

distances = distances_1 + distances_2
print(np.mean(distances), np.std(distances))


plt.scatter(range(len(distances_1)), distances_1, color='royalblue', label='measurement 1')
plt.scatter(range(len(distances_1), len(distances_1) + len(distances_2)), distances_2, color='navy', label='measurement 2')
plt.xlim(0, len(distances_1) + len(distances_2) - 1)
plt.ylim(0, 0.15)
plt.legend()
plt.ylabel(r"Distance ($\mu$m)")
plt.xlabel("measurement")
plt.grid()
plt.savefig("images\d_1_and_2.eps", format="eps")
plt.close()

### DRIFT ANALYSIS MEASUREMENT 1
data = pd.read_csv("2_analysis_22\Values_3.csv")
x2 = data['X2']
y2 = data['Y2']

x3 = data['X3']
y3 = data['Y3']

plt.plot(x2,y2, color='navy',label='y drift')
plt.plot(x3,y3, color='royalblue',label='x drift')
plt.xlabel('frame')
plt.ylabel('drift (px)')
plt.xlim(min(x2), max(x2))
plt.ylim(min(y2), max(y3))
plt.legend()
plt.grid()
plt.savefig("images\drift_measurement_1.eps", format="eps")
plt.close()

### DRIFT ANALYSIS MEASUREMENT 2
data = pd.read_csv("2_analysis_19\Values_2.csv")
x2 = data['X2']
y2 = data['Y2']

x3 = data['X3']
y3 = data['Y3']

plt.plot(x2,y2, color='navy',label='y drift')
plt.plot(x3,y3, color='royalblue',label='x drift')
plt.xlabel('frame')
plt.ylabel('drift (px)')
plt.xlim(min(x2), max(x2))
plt.ylim(min(y2), max(y3))
plt.legend()
plt.grid()
plt.savefig("images\drift_measurement_2.eps", format="eps")
plt.show()







