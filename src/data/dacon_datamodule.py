from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from omegaconf import DictConfig

class VideoDataModule(LightningDataModule):

    def __init__(
        self,
        data_cfg : DictConfig,
        loader_cfg : DictConfig
    ):
        super().__init__()
        
        # load model
        train_df = pd.read_csv(str(cfg.base_path/'train.csv'))
        test_df  = pd.read_csv(str(cfg.base_path/'test.csv'))

        train_df['video_path'] = train_df['video_path'].str.replace('./train/','')
        test_df['video_path'] = test_df['video_path'].str.replace('./test/','')

        # splite dataset
        train_df['fold'] = -1
        skf = StratifiedKFold(6)
        for i, (_, valid_index) in enumerate(skf.split(train_df,train_df['label'])):
            train_df.loc[valid_index,'fold'] = i 

        train_df['easy_annot'] = train_df['label'].apply(easy_annot)

        # set dataset
        datamodule = VideoClassificationData.from_data_frame(
            input_field         = 'video_path',
            target_fields       = 'easy_annot',
            train_data_frame    = train_df.query(f'fold!={cfg.fold}').reset_index(drop=True),
            train_videos_root   = str(cfg.train_path),
            val_data_frame      = train_df.query(f'fold=={cfg.fold}').reset_index(drop=True),
            val_videos_root     = str(cfg.train_path),
            predict_data_frame  = test_df,
            predict_videos_root = str(cfg.test_path),
            transform_kwargs    = dict(image_size=cfg.img_size),
            batch_size          = cfg.batch_size,
            clip_sampler        = "uniform",
            clip_duration       = 2,
            num_workers         = cfg.num_workers
            )


        self.save_hyperparameters(logger=False)
        

        
    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        self.data_train = hydra.utils.instantiate(self.hparams.data_cfg, mode='train')
        self.data_val = hydra.utils.instantiate(self.hparams.data_cfg, mode='valid')
        self.data_test = hydra.utils.instantiate(self.hparams.data_cfg, mode='valid')

    def train_dataloader(self):
        self.train_loader =  hydra.utils.instantiate(self.hparams.loader, 
                                    dataset=self.data_test,
                                    shuffle=True)

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = hydra.utils.instantiate(self.hparams.loader, 
                                            dataset=self.data_val,
                                            shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = hydra.utils.instantiate(self.hparams.loader, 
                                    dataset=self.data_test,
                                    shuffle=False)

        return self.test_loader

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MMGDataModule()
