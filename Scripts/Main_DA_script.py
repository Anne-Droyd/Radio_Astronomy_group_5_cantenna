
#Author: yara maybe? 


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


data3_real = np.fromfile('dataoutside_veclen8192_samprate10m_withintegrationblock_full10mcoax_centrefreq1418_coldload', dtype=np.float32) #Cold load / source 

data3_reshape = data3_real.reshape((-1,8192)) #number of channels = 8192
# print(data3_reshape)
print(data3_reshape.shape)

#this is voltage^2 and its proportional to power
data3_mean = np.mean(data3_reshape, axis=0)
print(data3_mean.shape)


signal = data3_mean
n = signal.size
print(n)
timestep = 1/1e7
freq = np.fft.fftfreq(n, d=timestep)
freq_shift = np.fft.fftshift((freq+1.418e9)/1e9) #shifts back from flowchart
print(freq_shift)

#hot load

data32_real = np.fromfile('dataoutside_veclen8192_samprate10m_withintegrationblock_withhotload', dtype=np.float32) #FFT Autocorrelated Data

data32_reshape = data32_real.reshape((-1,8192)) #number of channels = 8192
# print(data3_reshape)
print(data32_reshape.shape)

#this is voltage^2 and its proportional to power
data32_mean = np.mean(data32_reshape, axis=0)
print(data32_mean.shape)

plt.scatter(np.arange(8192), data32_mean, s=5)
plt.xlabel('Channel')
plt.ylabel('Power')
#plt.ylim(0, 0.1)
plt.show()


signal2 = data32_mean
n = signal2.size
print(n)
timestep = 1/1e7
freq = np.fft.fftfreq(n, d=timestep)
freq_shift = np.fft.fftshift((freq+1.418e9)/1e9) #shifts back from flowchart
print(freq_shift)

plt.plot(freq_shift, data32_mean) #hot load
plt.plot(freq_shift, data3_mean) #cold load/ source spectrum
plt.ylim(0, 0.2e6)
#Plt.xlim(-1e6,1e6)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power")
plt.title('Autocorrelated Data')
plt.show()


f = freq_shift


mask = (
    (f > 1.4135) &
    (f < 1.4225) &
    ~((f > 1.4178) & (f < 1.4195)) &
    ~((f > 1.4168) & (f < 1.4172)) &
    ~((f > 1.4210) & (f < 1.4220))
) #mask for the cold load

mask2 = (
    (f > 1.4135) &
    (f < 1.4225) &
    ~((f > 1.4178) & (f < 1.4188))) #mask for the hot load


x_fit_cold = f[mask]
x_fit_hot = f[mask2]
y_fit_cold = data3_mean[mask]
y_fit_hot = data32_mean[mask2]

# Spline fit
spline_hot = UnivariateSpline(x_fit_hot, y_fit_hot, s=1e8)
spline_cold = UnivariateSpline(x_fit_cold, y_fit_cold, s=1e8)

continuum_hot = spline_hot(f)
continuum_cold = spline_cold(f)


mask_final =  (
    (f > 1.414) &(f < 1.4215)) #to capture the bandwidth continuum better


plt.plot(f[mask_final], continuum_hot[mask_final], label="Hot Load Spline Spectrum")
plt.plot(f[mask_final], continuum_cold[mask_final], label="Cold Load Spline Spectrum", linewidth=2)

plt.xlabel("Frequency [MHz]")
plt.ylim(0,0.2e6)
plt.ylabel("Power")
plt.legend()
plt.show()


P_hot = continuum_hot[mask_final]
P_cold = continuum_cold[mask_final]

Y = P_hot / P_cold 

T_hot =  293 #K (20 degrees celcius outside)
T_cold = 11 #K estimated (CMB + atmospheric)

T_sys = (T_hot - T_cold * Y) / (Y-1)
Gain = (P_hot - P_cold)/(T_hot - T_cold)


plt.plot(freq_shift[mask_final], Y)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Y-factor")
plt.show()

plt.plot(freq_shift[mask_final], T_sys)
plt.xlabel("Frequency [MHz]")
plt.ylabel("$T_{sys}$ [K]")
plt.show()


plt.plot(freq_shift[mask_final], Gain)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Gain")
plt.show()



T_sky = data3_mean[mask_final]/ Gain - T_sys

f = freq_shift[mask_final]

f_mask_gaussian = (f > 1.4182) & (f < 1.4188) 
plt.plot(f[f_mask_gaussian],T_sky[f_mask_gaussian])
plt.ylim(6,20)
#plt.xlim(1.4182,1.4188)
plt.xlabel("Frequency [MHz]")
plt.ylabel("$T_{sky}$ [K]")
plt.show()


# calibrated spectrum
T_sky = data3_mean[mask_final] / Gain - T_sys
f = freq_shift[mask_final]

# HI mask
f_mask_gaussian = (f > 1.4181) & (f < 1.419)

x = f[f_mask_gaussian]
y = T_sky[f_mask_gaussian]

# Gaussian + baseline
def gaussian(x, A, mu, sigma, T0):
    return T0 + A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Initial guesses
A_guess = 10
mu_guess = x[np.argmax(y)]
sigma_guess = 0.00001
T0_guess = np.min(y)

p0 = [A_guess, mu_guess, sigma_guess, T0_guess]


popt, pcov = curve_fit(gaussian, x, y, p0=p0)

A, mu, sigma, T0 = popt


y_fit = gaussian(x, *popt)

width = 2 * 2*np.sqrt(2*np.log(2))*sigma #2 x FWHM



plt.figure(figsize=(8, 5))
plt.plot(x, y, label="HI line")
plt.plot(x, y_fit, label="Gaussian fit", linewidth=2)
plt.axvline(mu, label = f' Central Frequency  = {mu:0.4f} [MHz]' , linestyle = '--', color = 'red')
plt.axvline(mu+ 2*np.sqrt(2*np.log(2))*sigma, label = f' Width = {width:.4f} [MHz]', linestyle = '--', color = 'green')
plt.axvline(mu- 2*np.sqrt(2*np.log(2))*sigma, linestyle = '--', color = 'green')



plt.xlabel("Frequency [MHz]")
plt.ylabel("$T_{sky}$ [K]")
#plt.ylim(6, 20)
plt.legend()
plt.show()

print(f"Amplitude = {A:.3f} K")
print(f"Central frequency = {mu:.6f} MHz")
print(f"Sigma = {sigma:.6f} MHz")
print(f"Baseline = {T0:.3f} K")
print(f"Width = {width:.4f} MHz")


import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

LAB.plot_spectrum(
    ax,
    lon=92.55 * u.deg,
    lat=58.36 * u.deg,
    beam_size=79.72 * u.deg,
    beam_fn=airy_beam_fn,
    x_axis='frequency'
)



f = freq_shift[mask_final]       # GHz
T = T_sky.copy()

# Regions assumed to be outside the observed HI feature
baseline_mask = (f < 1.4182) | (f > 1.4188)

T_baseline = np.median(T[baseline_mask])

T_corrected = T - T_baseline


# Central frequencies in GHz
f_obs_centre = 1.418
f_lab_centre = 1.4204058

delta_f = f_lab_centre - f_obs_centre

f_shifted = f + delta_f

# Convert GHz to MHz
f_MHz = f_shifted * 1000

# Frequency range containing the observed HI emission
HI_mask = (
    (f_MHz >= 1420.5) &
    (f_MHz <= 1421.5)
)


# Because the spectrum has already been baseline corrected,
# use zero as the continuum level.
T_flat = T_corrected.copy()

T_flat[~HI_mask] = 0.0


ax.plot(
    f_MHz,
    T_flat,
    label='Observation'
)

ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel(r'$T_{\rm B}$ [K]')

ax.set_xlim(1419.5, 1422.0)
ax.set_ylim(0, 20)

ax.legend()

plt.tight_layout()
plt.show()