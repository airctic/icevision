__all__ = [
    "notnone",
    "ifnotnone",
    "last",
    "lmap",
    "allequal",
    "cleandict",
    "mergeds",
    "zipsafe",
    "np_local_seed",
    "pbar",
    "IMAGENET_STATS",
    "normalize",
    "denormalize",
    "normalize_imagenet",
    "denormalize_imagenet",
    "patch_class_to_main",
    "sort_losses",
    "get_stats",
]

from icevision.imports import *


def notnone(x):
    return x is not None


def ifnotnone(x, f):
    return f(x) if notnone(x) else x


def last(x):
    return next(reversed(x))


def lmap(f, xs):
    return list(map(f, xs)) if notnone(xs) else None


def allequal(l):
    return l.count(l[0]) == len(l) if l else True


def cleandict(d):
    return {k: v for k, v in d.items() if notnone(v)}


def mergeds(ds):
    aux = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            aux[k].append(v)
    return dict(aux)


def zipsafe(*its):
    if not allequal(lmap(len, its)):
        raise ValueError("The elements have different leghts")
    return zip(*its)


def pbar(iter, show=True):
    return tqdm(iter) if show else iter


@contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def normalize(img, mean, std, max_pixel_value=255):
    img = img.astype(np.float32)
    img /= max_pixel_value

    mean, std = map(np.float32, [mean, std])
    return (img - mean) / std


def denormalize(img, mean, std, max_pixel_value=255):
    return np.around((img * std + mean) * max_pixel_value).astype(np.uint8)


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def denormalize_imagenet(img):
    mean, std = IMAGENET_STATS
    return denormalize(img=img, mean=mean, std=std)


def normalize_imagenet(img):
    mean, std = IMAGENET_STATS
    return normalize(img=img, mean=mean, std=std)


def patch_class_to_main(cls):
    import __main__

    setattr(__main__, cls.__name__, cls)
    cls.__module__ = "__main__"
    return cls


def sort_losses(
    samples: List[dict], preds: List[dict], by: str = "loss_total"
) -> Tuple[List[dict], List[dict], List[str]]:
    l = list(zip(samples, preds))
    l = sorted(l, key=lambda i: i[0][by], reverse=True)
    sorted_samples, sorted_preds = zip(*l)
    annotations = [el["text"] for el in sorted_samples]
    return list(sorted_samples), list(sorted_preds), annotations


def get_stats(l: List) -> dict:
    l = np.array(l)
    quants_names = ["1ile", "25ile", "50ile", "75ile", "99ile"]
    quants = np.quantile(l, [0.01, 0.25, 0.5, 0.75, 0.99])
    d = {
        "min": l.min(),
        "max": l.max(),
        "mean": l.mean(),
    }

    q = {k: v for k, v in zip(quants_names, quants)}
    d.update(q)
    return d
