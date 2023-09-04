from src.models.abstract_learner import Supervised, PseudoLabel, FixMatch, FlexMatch
from src.models.backbone import ConvLinSeq, LinearSeq, LeNet5


def get_backbone(
    model_architecture: str,
):
    """ Create a model from a configuration. """
    if model_architecture == "conv":
        return ConvLinSeq
    elif model_architecture == "linear":
        return LinearSeq
    elif model_architecture in ["LeNet"]:
        return LeNet5


def get_learner(
    learner: str,
):
    """ Learner defines training procedure and loss function."""
    if learner == "supervised":
        return Supervised
    elif learner == "pseudolabel":
        return PseudoLabel
    elif learner == "fixmatch":
        return FixMatch
    elif learner == "flexmatch":
        return FlexMatch
