import numpy as np



def compute_ECDF(vector_of_band_gaps, lower_threshold, upper_threshold):
#input vector_of_band_gaps could be label band gaps if computing label ECDF or model's predicted band gaps if computing model prediction ECDF; the method returns both a vector of energies and vector of corresponding fractions
    
    num_energy_points = 1000000
    vector_of_energies = np.linspace(lower_threshold, upper_threshold, num_energy_points)

    vector_of_fractions = np.zeros(num_energy_points)
    for i in range(num_energy_points):
        energy = vector_of_energies[i]
        fraction = np.sum(vector_of_band_gaps <= energy) / len(vector_of_band_gaps)
        vector_of_fractions[i] = fraction
        
    return vector_of_energies, vector_of_fractions