from timm import create_model, list_models
from timm.data import create_transform
from torch import nn
import torch 

class VideoModel(nn.Module):
    def __init__(self, aux_label, model_type:str, dropout=0.,pretrained=False):
        super().__init__()
        model = VideoClassifier(
                        backbone      = cfg.network, 
                        num_classes   = train_df['easy_annot'].max()+1,
                        labels        = train_df['easy_annot'], 
                        pretrained    = cfg.pretrained,
                        learning_rate = cfg.lr,
                        optimizer     = cfg.optimizer,
                        metrics       = cfg.metrics,
                        loss_fn       = cfg.loss_fn,
                        )


        self.flat         = nn.Flatten()  
        self.crash_head   =  nn.Linear(400,2)
        self.ego_head     =  nn.Linear(400,2)
        self.weather_head =  nn.Linear(400,3)
        self.time_head    =  nn.Linear(400,2)

    def forward(self, x):
        x = self.flat(extracted_feature)
        crash_f = self.crash_head(x)
        ego_f = self.ego_head(x)
        weather_f = self.weather_head(x)
        time_f = self.time_head(x)
        return crash_f, ego_f, weather_f, time_f