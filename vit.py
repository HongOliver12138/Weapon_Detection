



# This file implement VIT model

import torch.nn as nn
from transformers import ViTForImageClassification
import torch

class VIT(nn.Module):

    def __init__(self,vit_out_dim = 256, num_classes = 2,enable_adapter = True, freeze_vit = False ):
        super(VIT,self).__init__()
        self.num_classes = num_classes
        self.vit = self.createVitForImageClassification(vit_out_dim)

        # replace fine classifier to output 2 dimensions
        self.vit.classifier = nn.Linear(in_features=self.vit.classifier.in_features, out_features=2)

        """
        if enable_adapter:
            self.add_adapter()
        if freeze_vit:
            self.freeze(self.vit)
        """


    def set_fineturning_layers(self,model, fineturning_layers = ["classifier.4.conv1"]):
        for name, param in model.named_parameters():
            if any(layer in name for layer in fineturning_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False




    """
    Adapters are small,lightweight modules that can be add ed to a pre-train model to improve fine-tuning.
    """
    def add_adapter(self,mode = "mediate"):
        if mode == "mediate":
            pass
            #self.vit = adapt_vit(self.vit)
        elif mode =="after":
            pass


    """
    This freeze funct will lock other parameter except the classifier layer
   
    def freeze(self,model):
        for name, param in model.named_



    """


    def createVitForImageClassification(self, num_classes = 512):

        # Ignore_mismatched_size usually use when the pre-train model has different output than the model you want to output
        # Any mismatched layers will be randomly initialized

        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",num_labels = num_classes,
                                                          ignore_mismatched_sizes = True)
        return model
    def forward(self,img):
        out = self.vit(img).logits
        return out



if __name__ == '__main__':
    random = torch.rand(1,3,224,224)
    model = VIT()
    output = model(random)
    print(output.shape)

