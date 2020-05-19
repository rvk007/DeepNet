from deepnet.model.models.resnet import ResNet, BasicBlock
from deepnet.model.models.customnet import CustomNet, CustomBlock
from deepnet.model.models.resmodnet import ResModNet, ModBasicBlock
from deepnet.model.models.masknet import MaskNet2

def ResNet18():
    """Create Resnet-18 architecture
    Returns:
        Resnet-18 architecture
    """
    return ResNet(BasicBlock, [2,2,2,2])

def CustomRes():
    """Create CustomNet architecture
    Returns:
        CustomNet architecture
    """
    return CustomNet(CustomBlock)

def ResModNet18():
    """Create ResModnet-18 architecture
    Returns:
        ResModnet-18 architecture
    """
    return ResModNet(ModBasicBlock, [2,2,2,2])

def MaskNet():

    return MaskNet2()

   