import pickle
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pymatgen.util.plotting import periodic_table_heatmap
import empirical_distribution
import material_representation
import chemical_elements



def create_cv_folds(list_of_chemical_formulas_all, list_of_band_gaps_all, list_of_atomic_numbers_for_featurization, num_folds):
#create a list of matrices and a list of vectors. In the list of matrices (vectors), the i-th entry contains a matrix of features (vector of labels) of the i-th fold for cv

    list_of_matrices_of_inputs_for_cv = []
    list_of_vectors_of_labels_for_cv = []
    num_samples_per_fold = round(len(list_of_chemical_formulas_all) / num_folds)
    
    for i in range(num_folds):
        if i != (num_folds-1):
            list_of_chemical_formulas_this_fold = list_of_chemical_formulas_all[i*num_samples_per_fold:(i+1)*num_samples_per_fold]
            list_of_band_gaps_this_fold = list_of_band_gaps_all[i*num_samples_per_fold:(i+1)*num_samples_per_fold]
        else:
            list_of_chemical_formulas_this_fold = list_of_chemical_formulas_all[i*num_samples_per_fold:]
            list_of_band_gaps_this_fold = list_of_band_gaps_all[i*num_samples_per_fold:]
        matrix_of_inputs_this_fold = material_representation.build_matrix_of_inputs_from_list_of_chemical_formulas(
                                                    list_of_chemical_formulas_this_fold, list_of_atomic_numbers_for_featurization)
        vector_of_labels_this_fold = np.array(list_of_band_gaps_this_fold)
        list_of_matrices_of_inputs_for_cv.append(matrix_of_inputs_this_fold)
        list_of_vectors_of_labels_for_cv.append(vector_of_labels_this_fold)

    return list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv



def create_dict_holding_all_cv_results(num_folds, num_energy_points_ECDF):
#creates a dict that will hold all relevant results from cv (train MAE, train RMSE, test MAE, test RMSE, and test model ECDF)

    all_cv_results_dict = {}
    all_cv_results_dict['train_RMSE'] = np.zeros(num_folds)
    all_cv_results_dict['train_MAE'] = np.zeros(num_folds)
    all_cv_results_dict['test_RMSE'] = np.zeros(num_folds)
    all_cv_results_dict['test_MAE'] = np.zeros(num_folds)
    all_cv_results_dict['test_model_ECDF'] = np.zeros((num_folds, num_energy_points_ECDF))
    
    return all_cv_results_dict



def return_aggregated_ECDF_and_print_aggregated_point_metrics(all_cv_results_dict):
#takes a dict w the cv results from each fold.  computes mean and stdev.  prints the train/test RMSE and train/test MAE aggregated metrics and returns test model ECDF aggregated metrics

    num_folds = len(all_cv_results_dict['train_RMSE'])
    num_energy_points_ECDF = all_cv_results_dict['test_model_ECDF'].shape[1]

    # compute mean and stdev of each point metric, where mean and stdev are taken over the cv folds
    for str_metric_name in ['train_RMSE', 'train_MAE', 'test_RMSE', 'test_MAE']:
        cv_results_for_this_metric = all_cv_results_dict[str_metric_name]
        mean_cv_result = np.mean(cv_results_for_this_metric)
        std_cv_result = np.std(cv_results_for_this_metric, ddof=1)
        print("mean_"+str_metric_name+":", mean_cv_result)
        print("std_"+str_metric_name+":", std_cv_result)

    #compute a vector of mean[F_{emp}(y)] at each value of y as well as a vector of stdev[F_{emp}(y)] at each value of y,
    # where mean and stdev are taken over the cv folds
    cv_results_for_test_model_ECDF = all_cv_results_dict['test_model_ECDF']
    mean_test_model_ECDF = np.mean(cv_results_for_test_model_ECDF, axis=0) 
    std_test_model_ECDF = np.std(cv_results_for_test_model_ECDF, axis=0, ddof=1)

    return mean_test_model_ECDF, std_test_model_ECDF
        
    

def create_train_test_partition_for_given_fold(list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv, index_of_fold):
#partition for (index_of_fold)-th cv split. Return matrix of features and vector of labels for train, as well as a separate matrix of features and vector of labels for test
    
    matrix_of_inputs_for_test = list_of_matrices_of_inputs_for_cv[index_of_fold]
    vector_of_labels_for_test = list_of_vectors_of_labels_for_cv[index_of_fold]
    list_of_matrices_of_inputs_for_train =\
                        list_of_matrices_of_inputs_for_cv[:index_of_fold] + list_of_matrices_of_inputs_for_cv[(index_of_fold+1):]
    list_of_vectors_of_labels_for_train =\
                        list_of_vectors_of_labels_for_cv[:index_of_fold] + list_of_vectors_of_labels_for_cv[(index_of_fold+1):]
    matrix_of_inputs_for_train = np.concatenate(list_of_matrices_of_inputs_for_train)
    vector_of_labels_for_train = np.concatenate(list_of_vectors_of_labels_for_train)

    return matrix_of_inputs_for_train, vector_of_labels_for_train, matrix_of_inputs_for_test, vector_of_labels_for_test



def intialize_and_fit_sklearn_OLS_model(matrix_of_inputs, vector_of_labels):
#initializes a sklearn linear model (w/o intercept) and fits it on matrix_of_inputs and vector_of_labels using OLS, returns the fitted model as well as in-sample RMSE and MAE metrics

    #initialize and fit model
    model = LinearRegression(fit_intercept=False)
    model.fit(matrix_of_inputs, vector_of_labels)

    #in-sample MAE and RMSE
    vector_of_predictions = model.predict(matrix_of_inputs)
    in_sample_MAE = np.sum(np.abs(vector_of_labels - vector_of_predictions)) / len(vector_of_labels)
    in_sample_MSE = np.sum((vector_of_labels - vector_of_predictions)**2) / len(vector_of_labels)
    in_sample_RMSE = np.sqrt(in_sample_MSE)

    return model, in_sample_RMSE, in_sample_MAE



def intialize_and_optimize_torch_model(matrix_of_inputs, vector_of_labels, num_iterations, model_type, show_plot):
#initializes a pytorch model (either linear or relu), optimizes it on matrix_of_inputs and vector_of_labels by minimizing MSE using Adam, returns the optimized model as well as in-sample RMSE and MAE metrics

    #initialize the model
    num_elts_total = matrix_of_inputs.shape[1]
    if model_type == 'relu':
        model = torch.nn.Sequential(torch.nn.Linear(num_elts_total, 1, bias=False), torch.nn.ReLU())
    elif model_type == 'linear':
        model = torch.nn.Linear(num_elts_total, 1, bias=False)
    else:
        raise

    #convert np matrix and np vector to torch tensors
    matrix_of_inputs_torch = torch.from_numpy(matrix_of_inputs)
    vector_of_labels_torch = torch.from_numpy(vector_of_labels)

    #Adam optimizer that optimizes the model params
    optimizer = torch.optim.Adam(model.parameters())

    #error metrics
    MSE_loss = torch.nn.MSELoss()
    MAE_loss = torch.nn.L1Loss()

    #perform optimization
    log = []
    for i in range(num_iterations):
        vector_of_predictions_torch = model(matrix_of_inputs_torch).reshape((-1,))
        empirical_loss = MSE_loss(vector_of_predictions_torch, vector_of_labels_torch)
        model.zero_grad()
        empirical_loss.backward()
        optimizer.step()
        log.append(empirical_loss.item())

    #final error metrics
    with torch.no_grad():
        vector_of_predictions_torch = model(matrix_of_inputs_torch).reshape((-1,))
        final_empirical_MSE = MSE_loss(vector_of_predictions_torch, vector_of_labels_torch)
        final_empirical_MAE = MAE_loss(vector_of_predictions_torch, vector_of_labels_torch)
        in_sample_RMSE = np.sqrt(final_empirical_MSE.item())
        in_sample_MAE = final_empirical_MAE.item()
        log.append(final_empirical_MSE.item())

    #plotting mse loss at each iteration
    if show_plot:
        plt.ylabel('MSE loss')
        plt.xlabel('iteration')
        plt.plot(log)
        plt.show()
    
    return model, in_sample_RMSE, in_sample_MAE
    


def evaluate_model(matrix_of_inputs, vector_of_labels, model, model_package, lower_energy_threshold, upper_energy_threshold):
#evaluates either a pytorch or sklearn model on a matrix of features and vector of corresponding labels.  Evaluation includes error metrics as well as the ECDF of the model's predictions
    
    if model_package == 'torch': #convert matrix of inputs to torch form, make model predictions, then convert that to a numpy vector
        matrix_of_inputs_torch = torch.from_numpy(matrix_of_inputs)
        with torch.no_grad():
            vector_of_predictions_torch = model(matrix_of_inputs_torch).reshape((-1,))
        vector_of_predictions = vector_of_predictions_torch.detach().numpy()
    elif model_package == 'sklearn': #make model predictions, which will already be a numpy vector
        vector_of_predictions = model.predict(matrix_of_inputs)
    else:
        raise

    #compute MAE and RMSE
    MAE = np.sum(np.abs(vector_of_labels - vector_of_predictions)) / len(vector_of_labels)
    MSE = np.sum((vector_of_labels - vector_of_predictions)**2) / len(vector_of_labels)
    RMSE = np.sqrt(MSE)

    #compute ECDF of model predictions on materials in this sample
    vector_of_energies_ECDF, vector_of_fractions_ECDF = empirical_distribution.compute_ECDF(
                                                            vector_of_predictions, lower_energy_threshold, upper_energy_threshold)

    return RMSE, MAE, vector_of_energies_ECDF, vector_of_fractions_ECDF



def perform_kfold_cross_validation(list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv, lower_energy_threshold, upper_energy_threshold, num_energy_points_ECDF, modeling_option, num_iterations=None):
#performs k-fold cross validation. returns dictionary holding the cv results for each fold

    if modeling_option == "relu_model_with_torch" and num_iterations is None:
        raise

    num_folds = len(list_of_matrices_of_inputs_for_cv)
    all_cv_results_dict = create_dict_holding_all_cv_results(num_folds, num_energy_points_ECDF)
    
    for index_of_fold in range(num_folds):
        matrix_of_inputs_for_train, vector_of_labels_for_train, matrix_of_inputs_for_test, vector_of_labels_for_test =\
                create_train_test_partition_for_given_fold(list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv, index_of_fold)
        if modeling_option == "linear_model_with_sklearn":
            linear_model, train_RMSE, train_MAE = intialize_and_fit_sklearn_OLS_model(matrix_of_inputs_for_train, vector_of_labels_for_train)
            test_RMSE, test_MAE, _, test_model_ECDF = evaluate_model(matrix_of_inputs_for_test, vector_of_labels_for_test,
                                                                     linear_model, 'sklearn', lower_energy_threshold, upper_energy_threshold)
        elif modeling_option == "relu_model_with_torch":
            relu_model, train_RMSE, train_MAE = intialize_and_optimize_torch_model(
                                                matrix_of_inputs_for_train, vector_of_labels_for_train, num_iterations, 'relu', False)
            test_RMSE, test_MAE, _, test_model_ECDF = evaluate_model(matrix_of_inputs_for_test, vector_of_labels_for_test,
                                                                     relu_model, 'torch', lower_energy_threshold, upper_energy_threshold)
        else:
            raise
        all_cv_results_dict['train_RMSE'][index_of_fold] = train_RMSE
        all_cv_results_dict['train_MAE'][index_of_fold] = train_MAE
        all_cv_results_dict['test_RMSE'][index_of_fold] = test_RMSE
        all_cv_results_dict['test_MAE'][index_of_fold] = test_MAE
        all_cv_results_dict['test_model_ECDF'][index_of_fold] = test_model_ECDF

    return all_cv_results_dict



def visualize_ecdf_of_labels_and_model_predictions(vector_of_energies_ECDF, label_ECDF_for_entire_dataset, mean_test_model_ECDF, std_test_model_ECDF, model_name):
# plots the ECDF for all the labels as well as the cv mean and cv stdev of the ECDF for the model's predictions

    top_of_shaded_region = mean_test_model_ECDF + std_test_model_ECDF
    bottom_of_shaded_region = mean_test_model_ECDF - std_test_model_ECDF

    plt.rcParams.update({'font.size': 14})
    plt.plot(vector_of_energies_ECDF, label_ECDF_for_entire_dataset, alpha=0.9, c='b', linestyle='--', linewidth=1.1, label='eCDF, labels')
    if model_name == 'relu':
        plt.plot(vector_of_energies_ECDF, mean_test_model_ECDF, alpha=0.65, c='r', linewidth=1.1,
                             label=r'eCDF, $\hat{\varepsilon}_{\mathrm{relu}}(\cdot)$ predictions')
    elif model_name == 'linear':
        plt.plot(vector_of_energies_ECDF, mean_test_model_ECDF, alpha=0.65, c='r', linewidth=1.1,
                             label=r'eCDF, $\hat{\varepsilon}_{\mathrm{linear}}(\cdot)$ predictions')
    else:
        raise
    plt.fill_between(vector_of_energies_ECDF, top_of_shaded_region, bottom_of_shaded_region, alpha=0.2, color='g', linewidth=0)
    plt.xlim(np.min(vector_of_energies_ECDF), np.max(vector_of_energies_ECDF))
    plt.ylim(-0.03, 1.03)
    plt.legend(loc='lower right')
    plt.xlabel('band gap energy (eV), $y$')
    plt.ylabel(r'CDF$(y)$')
    plt.grid()
    plt.show()



def visualize_learned_weights_on_periodic_table(model, modeling_option, list_of_atomic_numbers_for_featurization):
# visualizes learned weights of a given model on the periodic table

    if modeling_option == "linear_model_with_sklearn":
        vector_of_learned_weights = model.coef_
    elif modeling_option == "relu_model_with_torch":
        vector_of_learned_weights = model[0].weight[0].detach().numpy()
    elif modeling_option == 'linear_model_with_torch':
        vector_of_learned_weights = model.weight[0].detach().numpy()
    else:
        raise

    num_elts_total = len(list_of_atomic_numbers_for_featurization)
    parameters_to_visualize = {}
    for i in range(num_elts_total):
        atomic_number = list_of_atomic_numbers_for_featurization[i]
        str_elt = chemical_elements.get_str_elt(atomic_number)
        learned_weight_for_this_elt = vector_of_learned_weights[i]
        parameters_to_visualize[str_elt] = learned_weight_for_this_elt

    w_abs_max = np.max(np.abs(vector_of_learned_weights))

    periodic_table_heatmap(elemental_data=parameters_to_visualize, cbar_label="learned parameter (eV)",
                               show_plot=True, cmap="seismic_r", cmap_range=(-w_abs_max,w_abs_max), blank_color='gainsboro', value_format='%.2f')


    
def main():

    #make pytorch float64 to be consistent w/ numpy
    torch.set_default_dtype(torch.float64)
    
    #random seed for pytorch
    random_seed = 0
    torch.manual_seed(random_seed)
    
    # load saved data
    with open('data/chemical_formulas.pkl', 'rb') as f:
        list_of_chemical_formulas_all = pickle.load(f)    
    with open('data/band_gaps.pkl', 'rb') as f:
        list_of_band_gaps_all = pickle.load(f)

    #list of all atomic numbers present in dataset, in ascending order
    list_of_atomic_numbers_for_featurization = chemical_elements.get_list_of_atomic_numbers_in_list_of_chemical_formulas(
                                                                                                list_of_chemical_formulas_all)
    
    #energy thresholds and points for ECDFs, as well as ECDF for the labels
    lower_energy_threshold = -4.0
    upper_energy_threshold = 12.0
    vector_of_energies_ECDF, label_ECDF_for_entire_dataset = empirical_distribution.compute_ECDF(
                                                            np.array(list_of_band_gaps_all), lower_energy_threshold, upper_energy_threshold)
    num_energy_points_ECDF = len(vector_of_energies_ECDF)

    #num iterations for adam optimization
    num_opt_iterations_adam = 100000
    
    #featurize and create cv folds (will be used for both linear model and relu model)
    num_folds = 10
    list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv = create_cv_folds(
                                    list_of_chemical_formulas_all, list_of_band_gaps_all, list_of_atomic_numbers_for_featurization, num_folds)

    #cross validation for linear model using sklearn
    all_cv_results_dict_for_linear_model = perform_kfold_cross_validation(list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv,
                                            lower_energy_threshold, upper_energy_threshold, num_energy_points_ECDF, "linear_model_with_sklearn")
    print("\nCross Validation Results for Linear Model")
    mean_test_model_ECDF_for_linear_model, std_test_model_ECDF_for_linear_model =\
                                                return_aggregated_ECDF_and_print_aggregated_point_metrics(all_cv_results_dict_for_linear_model)
    visualize_ecdf_of_labels_and_model_predictions(vector_of_energies_ECDF, label_ECDF_for_entire_dataset,
                                                   mean_test_model_ECDF_for_linear_model, std_test_model_ECDF_for_linear_model, 'linear')

    #cross validation for relu model using torch
    all_cv_results_dict_for_relu_model = perform_kfold_cross_validation(list_of_matrices_of_inputs_for_cv, list_of_vectors_of_labels_for_cv,
                        lower_energy_threshold, upper_energy_threshold, num_energy_points_ECDF, "relu_model_with_torch", num_opt_iterations_adam)
    print("\nCross Validation Results for ReLU Model")
    mean_test_model_ECDF_for_relu_model, std_test_model_ECDF_for_relu_model =\
                                                return_aggregated_ECDF_and_print_aggregated_point_metrics(all_cv_results_dict_for_relu_model)
    visualize_ecdf_of_labels_and_model_predictions(vector_of_energies_ECDF, label_ECDF_for_entire_dataset,
                                                   mean_test_model_ECDF_for_relu_model, std_test_model_ECDF_for_relu_model, 'relu')

    #create matrix and vector of all data
    matrix_of_inputs_all = material_representation.build_matrix_of_inputs_from_list_of_chemical_formulas(
                                                        list_of_chemical_formulas_all, list_of_atomic_numbers_for_featurization)
    vector_of_labels_all = np.array(list_of_band_gaps_all)

    #fit linear model on all data, and visualize on periodic table
    final_linear_model, retrain_RMSE_linear_model, retrain_MAE_linear_model = intialize_and_fit_sklearn_OLS_model(
                                                                                    matrix_of_inputs_all, vector_of_labels_all)
    print("\nResults for Final Linear Model:")
    print("Retrain RMSE:", retrain_RMSE_linear_model)
    print("Retrain MAE:", retrain_MAE_linear_model)
    visualize_learned_weights_on_periodic_table(final_linear_model, "linear_model_with_sklearn", list_of_atomic_numbers_for_featurization)

    #fit relu model on all data, and visualize on periodic table
    final_relu_model, retrain_RMSE_relu_model, retrain_MAE_relu_model = intialize_and_optimize_torch_model(
                                            matrix_of_inputs_all, vector_of_labels_all, num_opt_iterations_adam, 'relu', False)
    print("\nResults for Final ReLU Model:")
    print("Retrain RMSE:", retrain_RMSE_relu_model)
    print("Retrain MAE:", retrain_MAE_relu_model)
    visualize_learned_weights_on_periodic_table(final_relu_model, "relu_model_with_torch", list_of_atomic_numbers_for_featurization)



if __name__ == '__main__':
    main()