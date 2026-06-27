import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden


@DATASET_REGISTRY.register()
class BTMRI(DatasetBase):

    dataset_dir = "BTMRI"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Fallback: ROOT may point to repo root while images live under data/BTMRI
        alt_dataset_dir = os.path.join(os.path.dirname(root), "data", "BTMRI")
        if not os.path.exists(self.dataset_dir) and os.path.exists(alt_dataset_dir):
            self.dataset_dir = alt_dataset_dir
        alt_dataset_dir2 = os.path.join(root, "data", "BTMRI")
        if not os.path.exists(self.dataset_dir) and os.path.exists(alt_dataset_dir2):
            self.dataset_dir = alt_dataset_dir2

        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, "split_BTMRI.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    def generate_fewshot_dataset_(self, num_shots, split):
        print("num_shots is ", num_shots)

        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)

        return few_shot_data

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/") or impath.startswith("\\"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def _classname_from_relpath(rel_path):
        rel = rel_path.replace("\\", "/").strip("/")
        parts = [p for p in rel.split("/") if p]
        if not parts:
            return None
        if parts[0] in ("train", "val", "test") and len(parts) >= 2:
            return parts[1]
        return parts[0]

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath_norm = impath.replace("\\", "/")
                impath_full = os.path.normpath(os.path.join(path_prefix, impath_norm))
                cn = BTMRI._classname_from_relpath(impath) or classname
                item = Datum(impath=impath_full, label=int(label), classname=cn)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=None, new_cnames=None, **kwargs):
        if ignored is None:
            ignored = []
        ignored = set(ignored) | {"split_fewshot", "images"}

        images_sub = os.path.join(image_dir, "images")
        if os.path.isdir(images_sub) and os.path.isdir(os.path.join(images_sub, "train")):
            return BTMRI._read_nested_train_val_test(images_sub)

        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored and os.path.isdir(os.path.join(image_dir, c))]
        categories.sort()

        if BTMRI._flat_uses_split_prefixes(image_dir, categories):
            return BTMRI._read_flat_prefixed(image_dir, categories, new_cnames)

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test (flat class folders)")

        def _collate(ims, y, category):
            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]
            items = []
            for im in ims:
                items.append(Datum(impath=im, label=y, classname=category))
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        print(f"Loaded {len(train)} train, {len(val)} val, {len(test)} test items from flat layout.")
        return train, val, test

    @staticmethod
    def _flat_uses_split_prefixes(image_dir, categories):
        for cat in categories:
            d = os.path.join(image_dir, cat)
            for fn in listdir_nohidden(d):
                if fn.startswith(("train_", "val_", "test_")):
                    return True
        return False

    @staticmethod
    def _read_flat_prefixed(image_dir, categories, new_cnames=None):
        category_to_label = {c: i for i, c in enumerate(categories)}

        def _datum(impath, category):
            cname = category
            if new_cnames is not None and cname in new_cnames:
                cname = new_cnames[cname]
            return Datum(
                impath=impath,
                label=category_to_label[category],
                classname=cname,
            )

        train, val, test = [], [], []
        for category in categories:
            category_dir = os.path.join(image_dir, category)
            for fn in sorted(listdir_nohidden(category_dir)):
                impath = os.path.join(category_dir, fn)
                if not os.path.isfile(impath):
                    continue
                item = _datum(impath, category)
                if fn.startswith("train_"):
                    train.append(item)
                elif fn.startswith("val_"):
                    val.append(item)
                elif fn.startswith("test_"):
                    test.append(item)

        print(
            f"Loaded {len(train)} train, {len(val)} val, {len(test)} test items "
            f"from flat folders (split by train_/val_/test_ prefixes)."
        )
        return train, val, test

    @staticmethod
    def _read_nested_train_val_test(images_root):
        all_categories = set()
        for split_name in ("train", "val", "test"):
            split_dir = os.path.join(images_root, split_name)
            if os.path.exists(split_dir):
                all_categories.update(listdir_nohidden(split_dir))
        categories = sorted(all_categories)
        category_to_label = {c: i for i, c in enumerate(categories)}

        def _read_split(split_name):
            items = []
            split_dir = os.path.join(images_root, split_name)
            if not os.path.exists(split_dir):
                return []
            for category in categories:
                category_dir = os.path.join(split_dir, category)
                if not os.path.exists(category_dir):
                    continue
                for im in listdir_nohidden(category_dir):
                    impath = os.path.join(category_dir, im)
                    items.append(Datum(impath=impath, label=category_to_label[category], classname=category))
            return items

        train = _read_split("train")
        val = _read_split("val")
        test = _read_split("test")
        print(
            f"Loaded {len(train)} train, {len(val)} val, {len(test)} test items "
            f"from nested images/{{train,val,test}}."
        )
        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]
        else:
            selected = labels[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
