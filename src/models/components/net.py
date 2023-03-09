from timm import create_model, list_models
from timm.data import create_transform
from torch import nn
import torch 

class BreastCancerModel(nn.Module):
    def __init__(self, aux_label, model_type:str, dropout=0.,pretrained=False):
        super().__init__()
        self.model = create_model(model_type, pretrained=pretrained, num_classes=0, drop_rate=dropout, features_only=False)
        
        self.backbone_dim = self.model(torch.randn(1, 3, 512, 512))
        
        self.last_feat = self.backbone_dim[-1].shape[-1]

        # self.middle_layer = [] 
        # for num,feat in enumerate(self.backbone_dim[:-1]): 
        #     self.middle_layer.append(torch.nn.Linear(feat.shape[-1],1))
        # self.middle_layer = torch.nn.ModuleList(self.middle_layer)
        
        # self.middle_layer = torch.nn.ModuleList([
        #                         torch.nn.Linear(feat.shape[-1],1) 
        #                         for feat in self.backbone_dim[:-1]]
        #                         )
       
        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(self.last_feat, 1),
        )
        
        self.nn_aux = torch.nn.ModuleList([
            torch.nn.Linear(self.last_feat, n) for n in aux_label
        ])

    def forward(self, x):
        # returns logits
        x = self.model(x)
        
        cancer = self.nn_cancer(x).squeeze()

        # mid_feat = [midnn(feat).squeeze() for midnn,feat in zip(self.middle_layer,x[:-1])]
        
        # print([i.shape for i in mid_feat],'123-10923-09')
        aux = []
        for nn in self.nn_aux:
            aux.append(nn(x).squeeze())
        return cancer, aux, cancer

    def predict(self, x):
        cancer, aux, featuers = self.forward(x)
        
        sigaux = []
        for a in aux:
            sigaux.append(torch.softmax(a, dim=-1))

        return torch.sigmoid(cancer), sigaux, featuers

# class BreastCancerModel(nn.Module):
#     def __init__(self, aux_label, model_type:str, dropout=0.,pretrained=False):
#         super().__init__()
#         self.model = create_model(model_type, pretrained=pretrained, num_classes=0, drop_rate=dropout)
        
#         self.backbone_dim = self.model(torch.randn(1, 3, 512, 512)).shape[-1]

#         self.nn_cancer = torch.nn.Sequential(
#             torch.nn.Linear(self.backbone_dim, 1),
#         )
#         self.nn_aux = torch.nn.ModuleList([
#             torch.nn.Linear(self.backbone_dim, n) for n in aux_label
#         ])

#     def forward(self, x):
#         # returns logits
#         x = self.model(x)

#         cancer = self.nn_cancer(x).squeeze()
#         aux = []
#         for nn in self.nn_aux:
#             aux.append(nn(x).squeeze())
#         return cancer, aux

#     def predict(self, x):
#         cancer, aux = self.forward(x)
#         sigaux = []
#         for a in aux:
#             sigaux.append(torch.softmax(a, dim=-1))
#         return torch.sigmoid(cancer), sigaux

# def convnext_tiny(pretrained=False, **kwargs):
#     model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
#     model = _create_convnext('convnext_tiny.in12k_ft_in1k_384', pretrained=pretrained, **model_args)
#     setattr(model, 'depths', [3, 3, 9, 3])
#     return model


# #modify to output all layers
# class Encoder(nn.Module):
#     def __init__(self, ):
#         super(Encoder, self).__init__()
#         e = convnext_tiny(pretrained=True)
#         self.stem = e.stem
#         self.stage1 = e.stages[         0  : e.depths[0]]
#         self.stage2 = e.stages[e.depths[0] : e.depths[1]]
#         self.stage3 = e.stages[e.depths[1] : e.depths[2]]
#         self.stage4 = e.stages[e.depths[2] : e.depths[3]]
#         self.norm_pre = e.norm_pre
#         del e

#     def forward(self, x):
#         x0 = self.stem(x)
#         x1 = self.stage1(x0)
#         x2 = self.stage2(x1)
#         x3 = self.stage3(x2)
#         x4 = self.stage4(x3)
#         return [x1,x2,x3,x4]
