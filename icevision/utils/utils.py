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
    "compute_weighted_sum",
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


def compute_weighted_sum(sample, weights):
    loss_weighted = 0
    for loss, weight in weights.items():
        loss_weighted += sample[loss] * weight
    sample["loss_weighted"] = loss_weighted
    return sample


def sort_losses(
    samples: List[dict], preds: List[dict], by: Union[str, dict] = "loss_total"
) -> Tuple[List[dict], List[dict], List[str]]:
    by_copy = deepcopy(by)
    losses_expected = [
        k for k in samples[0].keys() if "loss" in k and k != "loss_total"
    ]

    if isinstance(by, str):
        loss_check = losses_expected + ["loss_total"]
        assert (
            by in loss_check
        ), f"You must `sort_by` one of the losses. '{by}' is not among {loss_check}"

    if isinstance(by, dict):
        expected = ["weighted"]
        assert (
            by["method"] in expected
        ), f"`method` must be in {expected}, got {by['method']} instead."
        if by["method"] == "weighted":
            losses_passed = set(by["weights"].keys())
            losses_expected = set(losses_expected)
            assert (
                losses_passed == losses_expected
            ), f"You need to pass a weight for each of the losses in {losses_expected}, got {losses_passed} instead."
            samples = [compute_weighted_sum(s, by["weights"]) for s in samples]
            by = "loss_weighted"

    l = list(zip(samples, preds))
    l = sorted(l, key=lambda i: i[0][by], reverse=True)
    sorted_samples, sorted_preds = zip(*l)
    annotations = [el["text"] for el in sorted_samples]

    if isinstance(by_copy, dict):
        if by_copy["method"] == "weighted":
            annotations = [
                f"loss_weighted: {round(s['loss_weighted'], 5)}\n" + a
                for a, s in zip(annotations, sorted_samples)
            ]

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
