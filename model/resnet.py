from model import ResNet, BasicBlock


def ResNet18():
    """
    Create Resnet-18 architecture

    Returns:
        Resnet-18 architecture
    """
    return ResNet(BasicBlock, [2,2,2,2])