import pickle
import numpy as np
from matminer.datasets import load_dataset
import matplotlib.pyplot as plt
import scipy
import chemical_elements
import empirical_distribution



def compute_label_statistics(list_of_band_gaps, lower_threshold, upper_threshold):
#takes as input a list of band gap labels as well as lower and upper threshold energies for the ECDF. Computes statistics about the set of labels.

    vector_of_band_gaps = np.array(list_of_band_gaps)
    band_gap_max = np.max(vector_of_band_gaps)
    frac_w_zero_band_gap = np.sum(vector_of_band_gaps == 0) / len(vector_of_band_gaps)
    band_gap_mean = np.mean(vector_of_band_gaps)
    band_gap_std = np.std(vector_of_band_gaps, ddof=1)
    vector_of_energies_ECDF, vector_of_fractions_ECDF = empirical_distribution.compute_ECDF(vector_of_band_gaps, lower_threshold, upper_threshold)
    
    return frac_w_zero_band_gap, band_gap_max, band_gap_mean, band_gap_std, vector_of_energies_ECDF, vector_of_fractions_ECDF
    


def main():

    # load saved data for matbench_expt_gap (this is a processed version of that dataset, where a material with the rare elt Xe has been removed)
    with open('data/chemical_formulas.pkl', 'rb') as f:
        list_of_chemical_formulas_matbench_expt_gap = pickle.load(f)    
    with open('data/band_gaps.pkl', 'rb') as f:
        list_of_band_gaps_matbench_expt_gap = pickle.load(f)

    # load data for matbench_mp_gap using matminer and get the list of band gaps
    df_matbench_mp_gap = load_dataset("matbench_mp_gap")
    list_of_band_gaps_matbench_mp_gap = df_matbench_mp_gap['gap pbe'].to_list()

    #energy thresholds for calculating empirical distributions
    lower_threshold = -4.0
    upper_threshold = 12.0
    
    #num materials, zero frac, max, mean, stdev, and ECDF for processed matbench_expt_gap
    print("\nbasic statistics for processed matbench_expt_gap:")
    print("num materials:", len(list_of_band_gaps_matbench_expt_gap))
    frac_w_zero_band_gap_expt_gap, band_gap_max_expt_gap, band_gap_mean_expt_gap, band_gap_std_expt_gap,\
                                            vector_of_energies_ECDF, vector_of_fractions_ECDF_expt_gap =\
                                                compute_label_statistics(list_of_band_gaps_matbench_expt_gap, lower_threshold, upper_threshold)
    print("fraction of materials with zero band gap:", frac_w_zero_band_gap_expt_gap)
    print("max band gap:", band_gap_max_expt_gap)
    print("mean band gap:", band_gap_mean_expt_gap)
    print("std band gap:", band_gap_std_expt_gap)

    #num materials, zero frac, max, mean, stdev, and ECDF for matbench_mp_gap
    print("\nbasic statistics for matbench_mp_gap:")
    print("num materials:", len(list_of_band_gaps_matbench_mp_gap))
    frac_w_zero_band_gap_mp_gap, band_gap_max_mp_gap, band_gap_mean_mp_gap, band_gap_std_mp_gap, _, vector_of_fractions_ECDF_mp_gap =\
                                                    compute_label_statistics(list_of_band_gaps_matbench_mp_gap, lower_threshold, upper_threshold)
    print("fraction of materials with zero band gap:", frac_w_zero_band_gap_mp_gap)
    print("max band gap:", band_gap_max_mp_gap)
    print("mean band gap:", band_gap_mean_mp_gap)
    print("std band gap:", band_gap_std_mp_gap)

    #calculate CDF of normal distribution with mean and std equal to sample mean and sample stdev of processed matbench_expt_gap
    vector_of_cum_probs_CDF_normal_dist = scipy.stats.norm.cdf(vector_of_energies_ECDF, loc=band_gap_mean_expt_gap, scale=band_gap_std_expt_gap)
    
    #plot the two ECDFs as well as the reference normal distribution CDF
    plt.rcParams.update({'font.size': 14})
    plt.plot(vector_of_energies_ECDF, vector_of_fractions_ECDF_expt_gap, alpha=0.9, c='b', linestyle='--',
                                                                             linewidth=1.1, label='eCDF, labels from Zhuo et al.')
    plt.plot(vector_of_energies_ECDF, vector_of_fractions_ECDF_mp_gap, alpha=0.65, c='darkorange',
                                                                             linewidth=1.1, label='eCDF, labels from MP')
    plt.plot(vector_of_energies_ECDF, vector_of_cum_probs_CDF_normal_dist, alpha=0.55, c='g',
                                                                             linewidth=1.1, label='CDF, normal distribution')
    plt.xlim(lower_threshold, upper_threshold)
    plt.ylim(-0.03, 1.03)
    plt.legend(loc='lower right')
    plt.xlabel('band gap energy (eV), $y$')
    plt.ylabel(r'CDF$(y)$')
    plt.grid()
    plt.show()

    #breakdown by number of distinct elements for processed matbench_expt_gap (1, 2, 3, 4, 5+).
    num_materials_w_one_distinct_elt = 0
    num_materials_w_two_distinct_elt = 0
    num_materials_w_three_distinct_elt = 0
    num_materials_w_four_distinct_elt = 0
    num_materials_w_five_plus_distinct_elt = 0
    for chemical_formula in list_of_chemical_formulas_matbench_expt_gap:
        set_of_atomic_numbers_in_chemical_formula = chemical_elements.get_set_of_atomic_numbers_in_chemical_formula(chemical_formula)
        num_elts_in_chemical_formula = len(set_of_atomic_numbers_in_chemical_formula)
        if num_elts_in_chemical_formula == 1:
            num_materials_w_one_distinct_elt += 1
        elif num_elts_in_chemical_formula == 2:
            num_materials_w_two_distinct_elt += 1
        elif num_elts_in_chemical_formula == 3:
            num_materials_w_three_distinct_elt += 1
        elif num_elts_in_chemical_formula == 4:
            num_materials_w_four_distinct_elt += 1
        else:
            num_materials_w_five_plus_distinct_elt += 1
    print("\nbreakdown of processed matbench_expt_gap dataset based on number of distinct elements in material:")
    print("num materials with one distinct element:", num_materials_w_one_distinct_elt)
    print("num materials with two distinct elements:", num_materials_w_two_distinct_elt)
    print("num materials with three distinct elements:", num_materials_w_three_distinct_elt)
    print("num materials with four distinct elements:", num_materials_w_four_distinct_elt)
    print("num materials with five or more distinct elements:", num_materials_w_five_plus_distinct_elt)


    
if __name__ == '__main__':
    main()