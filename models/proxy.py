import torch
import torchvision
from torch import nn, Tensor
from torchvision.models.resnet import BasicBlock


class TinyResNet(torchvision.models.ResNet):
    def __init__(self, *args, **kwargs):
        super(TinyResNet, self).__init__(*args, **kwargs)
        num_features = self.layer2[-1].bn2.num_features
        self.fc = nn.Linear(num_features, kwargs['num_classes'])
        del self.layer3
        del self.layer4

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 
        

def _tinyresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = TinyResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def tinyresnet18(pretrained=False, progress=True, **kwargs):
    return _tinyresnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                       **kwargs)


if __name__ == "__main__":
    model = tinyresnet18(num_classes=2, pretrained=True)
    image = torch.randn(32, 3, 128, 128)
    output = model(image)
    print(output.shape)
