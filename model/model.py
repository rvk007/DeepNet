from deepnet.model.resnet import ResNet, BasicBlock
from deepnet.model.customnet import CustomNet, CustomBlock
from deepnet.model.resmodnet import ResModNet, ModBasicBlock

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

   