from configs.test_transformer_configs import *
from configs.test_brits_configs import *
from configs.test_other_configs import *
from configs.test_naomi_configs import *
from configs.test_neurdy_configs import * # added by RG

from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
import numpy as np
import torch
import random
from tqdm import tqdm

from ast import literal_eval # added by RG
from utils.loss_mask import mse_mask_loss # added by RG
import pandas as pd # added by RG

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars


if __name__=='__main__':
    
    # commented out by RG
    # bootstrap = (1000, 1) # num of bootstraps, size of bootstrap sample compared to test size
    configs = [bdc883_emb256_layer2_mimic_ppg_test2]


    for config in configs:
        print(config["modelname"]+config["annotate"]+config["annotate_test"])

        random_seed(10, True)
        
        # added by RG
        try:
            load = getattr(__import__(f'utils.universal_loader', fromlist=['']), "load")
            X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"],
                missingness_path = config["missingness_path"], **config["file_paths"], train=False, val=False, test=True)
        except:
            # original behavior from pulseimpute
            load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
            X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"])

        path = os.path.join("out/", config["data_name"]+config["annotate_test"], config["modelname"]+config["annotate"])

        model_type = config["modeltype"]
        model_module = __import__(f'models.{model_type}_model', fromlist=[''])
        model_module_class = getattr(model_module, model_type)
        model = model_module_class(modelname=config["modelname"], data_name=config["data_name"], 
                                train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                annotate=config["annotate"],  annotate_test=config["annotate_test"],  
                                **config["modelparams"],
                                **config["train"])
        imputation = model.testimp()

        # added by RG
        if config["save_imputation_file"]:
            np.save(os.path.join("vis_output", config["data_name"]+config["annotate_test"]+'_imputation'), imputation)
        
        # added by RG
        if config["save_mse"]:
            mse_all_data = []
            for i in range(len(imputation)):
                mse_scan_specific = []
                ppgmiss_idx = i % len(Y_dict_test['list_of_missing'])
                running_index = 0
                
                for j in range(len(Y_dict_test['list_of_missing'][ppgmiss_idx])):
                    list_of_miss = literal_eval(Y_dict_test['list_of_missing'][ppgmiss_idx][j])
            
            
                    if list_of_miss[0] == 0:
                        ground_truth = Y_dict_test['target_seq'][i][running_index:running_index+list_of_miss[1]]
                        imputed = torch.from_numpy(imputation[i][running_index:running_index+list_of_miss[1]])
                        mse_loss, missing_length = mse_mask_loss(ground_truth, imputed)
                        mse_scan_specific.append((mse_loss/missing_length).item())
                    running_index += list_of_miss[1]
                mse_all_data.append(mse_scan_specific)
            df = pd.DataFrame(mse_all_data)
            df.to_csv(os.path.join("vis_output", config["data_name"] + "_MSE.csv"), index=False, header=False)
   
        # block of code commented out by RG
        # if bootstrap is not None: # only support for mimic right now
        #     mse_losses, missing_totals = eval_mse(imputation, Y_dict_test["target_seq"], path, return_stats=True)
        #     printlog(f"MSE: {(torch.sum(mse_losses)/torch.sum(missing_totals)).item()}", path, type="w")
        #     mse_losses_bootstraplist = []

        #     if "mimic" in config["data_name"]:
        #         stats = eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], 
        #                                          input=X_test, path=path, return_stats=True)
        #         sens = np.nanmean(stats[:,0])
        #         prec = np.nanmean(stats[:,1])
        #         f1 = 2*sens*prec/(sens+prec)

        #         printlog(f"F1: {f1}", path)
        #         printlog(f"Precision: {prec}", path)
        #         printlog(f"Sensitivity: {sens}", path)
                
        #         f1_bootstraplist = []
        #         sens_bootstraplist = []
        #         prec_bootstraplist = []
        #     else:
        #         from sklearn.metrics import roc_auc_score
        #         eval_cardiac_classification(imputation, path)
        #         stats_true = {}
        #         stats_pred = {}
        #         for category in ["rhythm", "form", "diagnostic"]:
        #             y_test_true = np.load(os.path.join("out", config["data_name"]+config["annotate_test"],
        #                                                         config["modelname"]+config["annotate"], category,
        #                                                         "data", "y_test.npy"), allow_pickle=True)
        #             stats_true[category] = y_test_true
        #             y_test_pred = np.load(os.path.join("out", config["data_name"]+config["annotate_test"],
        #                                                 config["modelname"]+config["annotate"], category,
        #                                                 "models", "fastai_xresnet1d101", "y_test_pred.npy"), 
        #                                                 allow_pickle=True)
        #             stats_pred[category] = y_test_pred
        #         auc_bootstraplist = {
        #             "rhythm": [],
        #             "form": [],
        #             "diagnostic": []  
        #         }

        #     for bootstrap_iter in tqdm(range(bootstrap[0])):
        #         np.random.seed(bootstrap_iter)
        #         bootstrap_idxes = np.random.choice(mse_losses.shape[0],mse_losses.shape[0], replace=True)
                
        #         mse_losses_temp, missing_totals_temp = mse_losses[bootstrap_idxes], missing_totals[bootstrap_idxes]
        #         mse_losses_bootstraplist.append((torch.sum(mse_losses_temp)/torch.sum(missing_totals_temp)).item())

        #         if "mimic" in config["data_name"]:
        #             stats_temp = stats[bootstrap_idxes]
        #             sens = np.nanmean(stats_temp[:,0])
        #             prec = np.nanmean(stats_temp[:,1])
        #             f1 = 2*sens*prec/(sens+prec)


        #             f1_bootstraplist.append(f1)
        #             prec_bootstraplist.append(sens)
        #             sens_bootstraplist.append(prec)
        #         else:
        #             for category in ["rhythm", "form", "diagnostic"]:
        #                 np.random.seed(bootstrap_iter)
        #                 bootstrap_idxes = np.random.choice(stats_true[category].shape[0],  
        #                                                     stats_true[category].shape[0], replace=True)
        #                 y_test_true_temp = stats_true[category][bootstrap_idxes]
        #                 y_test_pred_temp = stats_pred[category][bootstrap_idxes]
        #                 # import pdb; pdb.set_trace()
        #                 try:
        #                     auc_bootstraplist[category].append(roc_auc_score(y_test_true_temp, 
        #                                                                     y_test_pred_temp, 
        #                                                                     average='macro'))
        #                 except ValueError:
        #                     pass
                                                



        #     printlog(f"95% CI MSE from Bootstrap {bootstrap}: {2*np.std(mse_losses_bootstraplist)}",path)
        #     if "mimic" in config["data_name"]:
        #         printlog(f"95% CI F1 from Bootstrap {bootstrap}: {2*np.std(f1_bootstraplist)}",path)
        #         printlog(f"95% CI Prec from Bootstrap {bootstrap}: {2*np.std(prec_bootstraplist)}",path)
        #         printlog(f"95% CI Sens from Bootstrap {bootstrap}: {2*np.std(sens_bootstraplist)}",path)
        #     else:
        #         printlog(f"95% CI Rhy AUC from Bootstrap {bootstrap}: {2*np.std(auc_bootstraplist['rhythm'])}",path)
        #         printlog(f"95% CI Form AUC from Bootstrap {bootstrap}: {2*np.std(auc_bootstraplist['form'])}",path)
        #         printlog(f"95% CI Diag AUC from Bootstrap {bootstrap}: {2*np.std(auc_bootstraplist['diagnostic'])}",path)

        # else:
        #     eval_mse(imputation, Y_dict_test["target_seq"], path)
        #     if "mimic" in config["data_name"]:
        #         eval_heartbeat_detection(imputation=imputation, target_seq=Y_dict_test["target_seq"], input=X_test, path=path)
        #     else:
        #         eval_cardiac_classification(imputation, path)