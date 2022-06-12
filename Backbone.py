import torch
import torchvision  

class Backbone(torch.nn.Module):
    def __init__(self, device, backbone_name="VGG19", ):
        super(Backbone, self).__init__()

        self.backbone_name = backbone_name
        self.__target_layers = [0, 5, 10, 19, 28]
       
        

        self.device = device

        if self.backbone_name == "VGG19":
            self.__backbone = torchvision.models.vgg19(pretrained=True)
            
            for param in self.__backbone.parameters():
                param.requires_grad = False

        self.__backbone.to(device)
        self.__backbone.eval()

    def forward(self, x):
        target_features = []
        
        for layer_num, layer in enumerate(self.__backbone.features):
            x = layer(x)
            if layer_num in self.__target_layers:
                target_features.append(x)
        return target_features


if __name__ == "__main__":
    data = torch.rand((2, 3, 256, 256)).to("cuda")
    st = Backbone(device="cuda", backbone_name="VGG19")
    out = st.forward(data)

    for o in out:
        print("out.shape", o.shape)

