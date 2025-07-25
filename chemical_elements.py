import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element



def get_atomic_number(str_elt):
# takes as input a string element symbol.  Returns the atomic number.  

    elt = Element(str_elt)
    return elt.Z



def get_str_elt(atomic_number):
#takes as input an integer atomic number.  Returns the string element symbol
    
    elt = Element.from_Z(atomic_number)
    return elt.symbol



def get_set_of_atomic_numbers_in_chemical_formula(str_chemical_formula):
# takes as input a string chemical formula.  Returns the corresponding set of atomic numbers that appear in this chemical formula

    chemical_comp = Composition(str_chemical_formula)
    el_amt_dict = chemical_comp.get_el_amt_dict()
    set_of_atomic_numbers = set()
    for str_elt in el_amt_dict:
        atomic_number = get_atomic_number(str_elt)
        set_of_atomic_numbers.add(atomic_number)
        
    return set_of_atomic_numbers
    
    
    
def get_list_of_atomic_numbers_in_list_of_chemical_formulas(list_of_chemical_formulas):
# takes as input a list of chemical formulas.  Returns a list of atomic numbers (sorted according to ascending atomic number) that appear in this list of chemical formulas

    set_of_atomic_numbers_in_list_of_chemical_formulas = set()

    for str_chemical_formula in list_of_chemical_formulas:
        set_of_atomic_numbers_in_this_chemical_formula = get_set_of_atomic_numbers_in_chemical_formula(str_chemical_formula)
        set_of_atomic_numbers_in_list_of_chemical_formulas.update(set_of_atomic_numbers_in_this_chemical_formula)

    list_of_atomic_numbers_in_list_of_chemical_formulas = list(set_of_atomic_numbers_in_list_of_chemical_formulas)
    list_of_atomic_numbers_in_list_of_chemical_formulas.sort()

    return list_of_atomic_numbers_in_list_of_chemical_formulas



def number_of_occurences_of_each_atomic_number_in_data(list_of_chemical_formulas):
# takes as input a list of chemical formulas.  Returns a vector of length 118, where entry at index i indicates the number of chemical formulas that contain the element with atomic number (i+1).

    num_of_atomic_numbers_in_periodic_table = 118 #this is for all possible elements, not just the ones that are in our dataset
    atomic_number_occurence_vector =  np.zeros(num_of_atomic_numbers_in_periodic_table, dtype=np.int64)
    
    for str_chemical_formula in list_of_chemical_formulas:
        set_of_atomic_numbers_in_this_chemical_formula = get_set_of_atomic_numbers_in_chemical_formula(str_chemical_formula)
        for atomic_number in set_of_atomic_numbers_in_this_chemical_formula:
            atomic_number_occurence_vector[atomic_number-1] += 1
            
    return atomic_number_occurence_vector