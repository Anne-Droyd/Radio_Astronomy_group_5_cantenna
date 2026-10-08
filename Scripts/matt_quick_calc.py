
import numpy as np
freq_obs = 1.4185e9
freq_lit = 1.420405e9
line_width = 0.1e6
delta_line = line_width/2
c = 2.99792e8

target_velocity_relative_to_earth=c*((freq_lit-freq_obs)/(freq_obs))
medium_velocity = c*(delta_line/freq_obs)

print(f'target velocity: {target_velocity_relative_to_earth/1000:.2f} km/s')
print(f'medium velocity: {medium_velocity/1000:.2f} km/s')

sig_medium_velocity = medium_velocity/(np.sqrt(2*np.log(2)))
print(f'Sigma medium velocity: {sig_medium_velocity/1000:.2f} km/s')

mh=1.6735e-27
kb = 1.38064852e-23
medium_temp = mh*sig_medium_velocity**2/kb
print(f'Medium temperature: {medium_temp:.2f}K')


