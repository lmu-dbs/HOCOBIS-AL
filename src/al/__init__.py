from .random import RandomSampling
from .uncertainty import MarginSampling, EntropySampling, LCSampling
from .kmeans import KMeansSampling
from .kcentergreedy import KCenterGreedy, BalanceSampling


def get_strategy(name, learner):
    instance = None
    strategy_params = {}
    klass = globals()[name]
    if name in [
        "RandomSampling",
        "EntropySampling",
        "MarginSampling",
        "KCenterGreedy",
        "KMeansSampling",
        "BalanceSampling",
        "LCSampling"
    ]:
        instance = klass(learner)
    else:
        print("Strategy not found")
    return instance, strategy_params
