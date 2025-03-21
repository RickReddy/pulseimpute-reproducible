import os

epoch = "best"

file_root_path = os.path.join("data", "neurdy_waveforms", "eegfmri_nih_ppg")
bdc883_emb256_layer2_mimic_ppg_test2 = {'modelname':'bdc883_emb256_layer2', "annotate":"_mimic_ppg", 'modeltype':'transformer', 
             "annotate_test":"_test",
            "data_name":"eegfmri_nih_ppg","data_load": {"addmissing":True, "mean":True, "bounds":1},
            "modelparams":{"convertolong":{"attention_window":[800,800], "attention_dilation":[4,4] }},
            "train":{"iter_save":10, "reload_epoch_long":54200,"bs": 2, "gpus":[0,1], "train_realppg":True,
                "model_weights_path": os.path.join("out","mimic_ppg","bdc883_emb256_layer2_mimic_ppg","epoch_54200","epoch_54200.pkl",),
                "retrain_for_X_epochs": 2,
                "freeze_layer_1": False,
                "freeze_layer_23": False,
                "freeze_layer_4": False},
            "save_imputation_file": True,
            "save_mse": True,
            
            "missingness_path": os.path.join("data", "neurdy_missingness_patterns", "missing_ppg_test_10sec.csv"),
            
            "file_paths": {"file_train": os.path.join(file_root_path, "eegfmri_nih_ppg_train.npy"),
            "file_val": os.path.join(file_root_path, "eegfmri_nih_ppg_val.npy"),
            "file_test": os.path.join(file_root_path, "eegfmri_nih_ppg_test.npy")}
            
            }