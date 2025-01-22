import pickle
from matminer.datasets import load_dataset
import numpy as np
import chemical_elements



def main():

    #load from matbench
    df_matbench_expt_gap = load_dataset("matbench_expt_gap")
    seed = 0
    shuffled_df_matbench_expt_gap = df_matbench_expt_gap.sample(frac = 1, random_state=seed)
    list_of_chemical_formulas_unprocessed = shuffled_df_matbench_expt_gap['composition'].to_list()
    list_of_band_gaps_unprocessed = shuffled_df_matbench_expt_gap['gap expt'].to_list()
    
    #remove material containing rare element Xe
    num_samples_unprocessed = len(list_of_chemical_formulas_unprocessed)
    list_of_chemical_formulas = []
    list_of_band_gaps = []
    atomic_number_for_Xe = chemical_elements.get_atomic_number("Xe")
    for i in range(num_samples_unprocessed):
        chemical_formula = list_of_chemical_formulas_unprocessed[i]
        band_gap = list_of_band_gaps_unprocessed[i]
        set_of_atomic_numbers_in_this_chemical_formula =\
                    chemical_elements.get_set_of_atomic_numbers_in_chemical_formula(chemical_formula)
        if atomic_number_for_Xe in set_of_atomic_numbers_in_this_chemical_formula:
            print("removed chemical formula", chemical_formula, ", which has band gap", band_gap)
        else:
            list_of_chemical_formulas.append(chemical_formula)
            list_of_band_gaps.append(band_gap)
    print("\nnum samples before processing:", num_samples_unprocessed)
    print("num samples after processing:", len(list_of_chemical_formulas))
    atomic_number_occurence_vector_after_processing =\
        chemical_elements.number_of_occurences_of_each_atomic_number_in_data(list_of_chemical_formulas)
    print("\natomic number occurence vector after processing:", atomic_number_occurence_vector_after_processing)
    print("sorted atomic number occurence vector after processing:", np.sort(atomic_number_occurence_vector_after_processing))

    #save the processed list of chemical formulas
    filename_with_path_for_chemical_formulas = "data/chemical_formulas.pkl"
    output_for_chemical_formulas = open(filename_with_path_for_chemical_formulas, "wb")
    pickle.dump(list_of_chemical_formulas, output_for_chemical_formulas, pickle.HIGHEST_PROTOCOL)
    output_for_chemical_formulas.close()

    #save the processed list of band gaps
    filename_with_path_for_band_gaps = "data/band_gaps.pkl"
    output_for_band_gaps = open(filename_with_path_for_band_gaps, "wb")
    pickle.dump(list_of_band_gaps, output_for_band_gaps, pickle.HIGHEST_PROTOCOL)
    output_for_band_gaps.close()



if __name__ == '__main__':
    main()