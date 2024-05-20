import os
from os.path import abspath, dirname, join, split

import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import Initializer, ModelManager
from ehr2vec.common.loader import load_and_select_splits, get_pids_file
from ehr2vec.common.setup import (DirectoryPreparer, copy_data_config,
                                  copy_pretrain_config, get_args)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.data.split import get_n_splits_cv
from ehr2vec.evaluation.utils import (
    check_data_for_overlap, compute_and_save_scores_mean_std, save_data,
    split_into_test_data_and_train_val_indices)
from ehr2vec.trainer.trainer import EHRTrainer
from ehr2vec.model.model_with_temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader, Dataset
from ehr2vec.dataloader.collate_fn import dynamic_padding

CONFIG_NAME = 'TS_calibration_test.yaml'
N_SPLITS = 5 #5  # You can change this to desired value
BLOBSTORE='XUE'
DEAFAULT_VAL_SPLIT = 0.2

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def calibrate_fold(cfg, cali_data:Data, test_data:Data, fold:int)->None:
    """Calibrate model on one fold using temperature scaling."""
    # adjust cfg: set pretrain_model_path to fold_{i} folder
    cfg.paths.pretrain_model_path = join(cfg.paths.pretrain_model_path, f'fold_{fold}')
    print(cfg.paths.pretrain_model_path)
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving patient numbers")
    logger.info("Saving pids")
    torch.save(cali_data.pids, join(fold_folder, 'cali_pids.pt'))
    torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))
    # cali_data saved as train_data, test_data saved as val_data
    dataset_preparer.saver.save_patient_nums(cali_data, test_data, folder=fold_folder)

    logger.info('Initializing datasets')
    cali_dataset = BinaryOutcomeDataset(cali_data.features, cali_data.outcomes)
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes)
    modelmanager = ModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint() 
    modelmanager.load_model_config() 
    model = modelmanager.initialize_finetune_model(checkpoint, cali_dataset)
    
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(model, cali_dataset)
    epoch = modelmanager.get_epoch()

    cali_dataloader = DataLoader(
            cali_dataset, 
            batch_size=cfg.trainer_args.batch_size, 
            shuffle=False, 
            collate_fn=dynamic_padding
        )
    
    test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.trainer_args.batch_size, 
            shuffle=False, 
            collate_fn=dynamic_padding
        )

    model_with_temperature_scaling = ModelWithTemperature(model, metrics=cfg.metrics, run_folder=fold_folder)
    model_with_temperature_scaling.set_temperature(test_loader=test_dataloader, cali_loader=cali_dataloader)
    # remove fold_{i} from pretrain_model_path
    # /tmp/tmpa20vtp6p/models/finetune_ANTI_PCSK9_censored_7_days_cv_5folds_all_data/fold_0
    cfg.paths.pretrain_model_path = cfg.paths.pretrain_model_path[:-7]

def cv_loop_predefined_splits(data: Data, predefined_splits_dir: str)->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Calibrating fold {fold}/{len(fold_dirs)}")
        logger.info("Load and select pids")
        cali_pids = torch.load(get_pids_file(fold_dir, 'calibrate'))
        test_pids = torch.load(get_pids_file(fold_dir, 'test'))
        cali_data = data.select_data_subset_by_pids(cali_pids, mode='calibration')
        test_data = data.select_data_subset_by_pids(test_pids, mode='test')
        check_data_for_overlap(cali_data, test_data, None)
        calibrate_fold(cfg, cali_data, test_data, fold)
    return N_SPLITS

if __name__ == '__main__':
    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_data()    
    
    if 'predefined_splits' in cfg.paths:
        logger.info('Using predefined splits')
        N_SPLITS = cv_loop_predefined_splits(data, cfg.paths.predefined_splits)

    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name, overwrite=False))
        mount_context.stop()
    logger.info('Done')
