import numpy as np
from pymatgen.core.composition import Composition
import chemical_elements



def convert_chemical_formula_to_vector_rep(str_chemical_formula, list_of_atomic_numbers_for_featurization):
# takes as input a string chemical formula.  Returns a np vector that is its representation as a vector of element fractions
    
    chemical_comp = Composition(str_chemical_formula)
    normalized_chemical_comp_el_amt_dict = chemical_comp.fractional_composition.get_el_amt_dict()
    
    rep_dim = len(list_of_atomic_numbers_for_featurization)
    material_vector = np.zeros(rep_dim)
    for str_elt in normalized_chemical_comp_el_amt_dict:
        fraction_for_this_elt = normalized_chemical_comp_el_amt_dict[str_elt]
        atomic_number_corr_to_this_str_elt = chemical_elements.get_atomic_number(str_elt)
        index_corr_to_this_str_elt = list_of_atomic_numbers_for_featurization.index(atomic_number_corr_to_this_str_elt)
        material_vector[index_corr_to_this_str_elt] = fraction_for_this_elt

    return material_vector



def build_matrix_of_inputs_from_list_of_chemical_formulas(list_of_chemical_formulas, list_of_atomic_numbers_for_featurization):
# takes as input a list of chemical formula strings.  Returns a matrix where the i-th row is the element fraction vector representation of the i-th chemical formula in the list of chemical formula strings.
    
    rep_dim = len(list_of_atomic_numbers_for_featurization)
    num_of_materials = len(list_of_chemical_formulas)
    
    matrix_of_inputs = np.zeros((num_of_materials,rep_dim))
    
    for i in range(num_of_materials):
        material_vector = convert_chemical_formula_to_vector_rep(list_of_chemical_formulas[i], list_of_atomic_numbers_for_featurization)
        matrix_of_inputs[i] = material_vector
        
    return matrix_of_inputs