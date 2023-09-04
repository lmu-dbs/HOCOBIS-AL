from torchvision import transforms

from src.data.randaugment import RandAugment


def get_augmentations(aug_type, appendix_augs, add_default):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    default_train_augs = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    # appendix_augs = [
    #     transforms.ToTensor(),
    #     normalize,
    # ]
    appendix_augs = appendix_augs.transforms if appendix_augs else []
    if aug_type == 'DefaultTrain':
        if add_default:
            augs = default_train_augs + appendix_augs
        else:
            augs = appendix_augs
    elif aug_type == 'RandAugment':
        if add_default:
            augs = default_train_augs + [RandAugment(n=2, m=10)] + appendix_augs
        else:
            augs = [RandAugment(n=2, m=10)] + appendix_augs
    else:
        raise NotImplementedError('augmentation type not found: {}'.format(aug_type))

    return augs


def get_transforms(aug_type, appendix_augs, add_default=False):
    augs = get_augmentations(aug_type, appendix_augs, add_default)
    return transforms.Compose(augs)


class TwoCropsTransform:
    """Take two random crops of one image."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        out1 = self.transform1(x)
        out2 = self.transform2(x)
        return out1, out2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        names = ['transform1', 'transform2']
        for idx, t in enumerate([self.transform1, self.transform2]):
            format_string += '\n'
            t_string = '{0}={1}'.format(names[idx], t)
            t_string_split = t_string.split('\n')
            t_string_split = ['    ' + tstr for tstr in t_string_split]
            t_string = '\n'.join(t_string_split)
            format_string += '{0}'.format(t_string)
        format_string += '\n)'
        return format_string
