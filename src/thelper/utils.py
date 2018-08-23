import glob
import importlib
import inspect
import logging
import math
import json
import os
import re
import sys
import time
import itertools

import cv2 as cv
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def test_cuda_device():
    """
    work around for gpu identification on cluster
    :return:
    """
    device_id = -1
    try:
        device_id = 0
        torch.cuda.set_device(device_id)
        a = torch.cuda.FloatTensor(1)
        logger.info('Setting to device: 0')
    except:
        try:
            device_id = 1
            torch.cuda.set_device(device_id)
            a = torch.cuda.FloatTensor(1)
            logger.info('Setting to device: 1')
        except:
            device_id = -1
            logger.warning("No cuda device available!")

    torch.cuda.empty_cache()
    return device_id


def import_class(fullname):
    modulename, classname = fullname.rsplit('.', 1)
    module = importlib.import_module(modulename)
    return getattr(module, classname)


def get_class_logger(skip=0):
    """Shorthand to get logger for current class frame."""
    return logging.getLogger(get_caller_name(skip + 1).rsplit(".", 1)[0])


def get_func_logger(skip=0):
    """Shorthand to get logger for current function frame."""
    return logging.getLogger(get_caller_name(skip + 1))


def get_caller_name(skip=2):
    # source: https://gist.github.com/techtonik/2151727
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """

    def stack_(frame):
        framelist = []
        while frame:
            framelist.append(frame)
            frame = frame.f_back
        return framelist

    stack = stack_(sys._getframe(1))
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start]
    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe
    return ".".join(name)


def str2bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, (int, float)):
        return s != 0
    if isinstance(s, str):
        positive_flags = ["true", "1", "yes"]
        return s.lower() in positive_flags
    raise AssertionError("unrecognized input type")


def clipstr(s, size, fill=" "):
    if len(s) > size:
        s = s[:size]
    if len(s) < size:
        s = fill * (size - len(s)) + s
    return s


def lreplace(string, old_prefix, new_prefix):
    return re.sub(r'^(?:%s)+' % re.escape(old_prefix), lambda m: new_prefix * (m.end() // len(old_prefix)), string)


def keyvals2dict(keyvals):
    if not isinstance(keyvals, list):
        raise AssertionError("expected key-value pair vector")
    out = {}
    for idx in range(len(keyvals)):
        item = keyvals[idx]
        if not isinstance(item, dict):
            raise AssertionError("expected items to be dicts")
        elif "name" not in item or "value" not in item:
            raise AssertionError("expected 'name' and 'value' fields in each item")
        out[item["name"]] = item["value"]
    return out


def query_yes_no(question, default=None):
    """
    Asks the user a yes/no question and returns the answer.

    :param question: the string that is presented to the user.
    :param default: the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    :returns: True for "yes" or False for "no".
    """
    valid = {"yes": True, "ye": True, "y": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise AssertionError("invalid default answer: '%s'" % default)
    sys.stdout.flush()
    sys.stderr.flush()
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes/y' or 'no/n'.\n")


def query_string(question, default=None, allow_empty=False):
    """
    Asks the user a question and returns the answer (a generic string).

    :param question: the string that is presented to the user.
    :param default: the presumed answer if the user just hits <Enter>.
    :param allow_empty: defines if empty string should be accepted.
    :returns: The string entered by the user.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    while True:
        if default is not None:
            sys.stdout.write(question + " (default=" + default + ")")
        else:
            sys.stdout.write(question)
        answer = input()
        if answer == "" and default is not None:
            return default
        elif answer == "" and allow_empty:
            return answer
        else:
            sys.stdout.write("Please respond with a non-empty string.\n")


def get_save_dir(out_root, session_name, config, resume=False):
    logger = get_func_logger()
    save_dir = out_root
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, session_name)
    if not resume:
        overwrite = False
        if "overwrite" in config:
            overwrite = str2bool(config["overwrite"])
        old_session_name = session_name
        time.sleep(0.5)  # to make sure all debug/info prints are done, and we see the question
        while os.path.exists(save_dir) and not overwrite:
            overwrite = query_yes_no("Training session at '%s' already exists; overwrite?" % save_dir)
            if not overwrite:
                session_name = query_string("Please provide a new session name (old=%s):" % old_session_name)
                save_dir = os.path.join(save_dir, session_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        config_backup_path = os.path.join(save_dir, "config.json")
        json.dump(config, open(config_backup_path, "w"), indent=4, sort_keys=False)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        config_backup_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_backup_path):
            config_backup = json.load(open(config_backup_path, "r"))
            if config_backup != config:
                answer = query_yes_no("Config backup in '%s' differs from config loaded through checkpoint; overwrite?" % config_backup_path)
                if answer:
                    logger.warning("config mismatch with previous run; will overwrite backup in save directory")
                else:
                    logger.error("config mismatch with previous run; user aborted")
                    sys.exit(1)
        json.dump(config, open(config_backup_path, "w"), indent=4, sort_keys=False)
    logs_dir = os.path.join(save_dir, "logs")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    return save_dir


def draw_histogram(data, bins=50, xlabel="", ylabel="Proportion"):
    fig, ax = plt.subplots()
    ax.hist(data, density=True, bins=bins)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=0)
    fig.show()


def draw_popbars(labels, counts, xlabel="", ylabel="Pop. Count"):
    fig, ax = plt.subplots()
    xrange = range(len(labels))
    ax.bar(xrange, counts, align="center")
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    ax.set_xticks(xrange)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize="8", labelrotation=45)
    fig.show()


def draw_classifs(images, labels_gt, labels_pred=None, labels_map=None):
    nb_imgs = len(images) if isinstance(images, list) else images.shape[images.ndim - 1]
    if nb_imgs < 1:
        return
    grid_size_x = int(math.ceil(math.sqrt(nb_imgs)))
    grid_size_y = int(math.ceil(nb_imgs / grid_size_x))
    if grid_size_x * grid_size_y < nb_imgs:
        raise AssertionError("bad gridding for subplots")
    fig, axes = plt.subplots(grid_size_y, grid_size_x)
    plt.tight_layout()
    if nb_imgs == 1:
        axes = np.array(axes)
    for ax_idx, ax in enumerate(axes.reshape(-1)):
        if ax_idx < nb_imgs:
            if isinstance(images, list):
                ax.imshow(images[ax_idx], interpolation='nearest')
            else:
                ax.imshow(images[ax_idx, ...], interpolation='nearest')
            curr_label_gt = labels_map[labels_gt[ax_idx]] if labels_map else labels_gt[ax_idx]
            if labels_pred is not None:
                curr_label_pred = labels_map[labels_pred[ax_idx]] if labels_map else labels_pred[ax_idx]
                xlabel = "GT={0}\nPred={1}".format(curr_label_gt, curr_label_pred)
            else:
                xlabel = "GT={0}".format(curr_label_gt)
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.show()


def draw_sample(sample, pred=None, image_key="image", label_key="label", block=False):
    if not isinstance(sample, dict):
        raise AssertionError("expected dict-based sample")
    if image_key not in sample or label_key not in sample:
        if len(sample) == 2:
            get_func_logger().warning("bad or missing image/label keys, will try to guess them for visualization")
        else:
            raise AssertionError("missing classification-related fields in sample dict, and dict is multi-elem")
        key1, key2 = sample.keys()
        if ((isinstance(sample[key1], torch.Tensor) and sample[key1].dim() > 1) and
            (isinstance(sample[key2], list) or (isinstance(sample[key2], torch.Tensor) and sample[key2].dim() == 1))):
            image_key, label_key = key1, key2
        elif ((isinstance(sample[key2], torch.Tensor) and sample[key2].dim() > 1) and
              (isinstance(sample[key1], list) or (isinstance(sample[key1], torch.Tensor) and sample[key1].dim() == 1))):
            image_key, label_key = key2, key1
        else:
            raise AssertionError(
                "missing classification-related fields in sample dict, and could not find proper default types")
    labels = sample[label_key]
    if not isinstance(labels, list) and not (isinstance(labels, torch.Tensor) and labels.dim() == 1):
        raise AssertionError("expected classification labels to be in list or 1-d tensor format")
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    # here we assume the sample's data has been tensor'd (so images are 4D, BxCxHxW)
    images = sample[image_key]
    if not isinstance(images, torch.Tensor):
        raise AssertionError("expected classification images to be in 4-d tensor format")
    images = images.numpy().copy()
    if images.ndim != 4:
        raise AssertionError("unexpected dimension count for input images tensor")
    if images.shape[0] != len(labels):
        raise AssertionError("images/labels count mismatch")
    images = np.transpose(images, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
    masks = sample["mask"].numpy().copy() if "mask" in sample else None
    if masks is not None:
        # masks should have same dim count, but 2nd always equal to 1 (single channel)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise AssertionError("image/mask ndim mismatch")
        if (images.shape[0:3] != np.asarray(masks.shape)[[0, 2, 3]]).any():
            raise AssertionError("image/mask shape mismatch")
    image_list = []
    for batch_idx in range(images.shape[0]):
        image = images[batch_idx, ...]
        if image.ndim != 3:
            raise AssertionError("indexing should return a pre-squeezed array")
        if image.shape[2] == 2:
            image = np.dstack((image, image[:, :, 0]))
        elif image.shape[2] > 3:
            image = image[..., :3]
        image_normalized = np.empty_like(image, dtype=np.uint8).copy()  # copy needed here due to ocv 3.3 bug
        cv.normalize(image, image_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        image_list.append(image_normalized)
    draw_classifs(image_list, labels, labels_pred=pred)  # normalize & pass mask to draw func also? todo
    if block:
        plt.show(block=block)
    else:
        plt.pause(0.01)


def draw_errbars(labels, min, max, stddev, mean, xlabel="", ylabel="Raw Value"):
    if min.shape != max.shape or min.shape != stddev.shape or min.shape != mean.shape:
        raise AssertionError("input dim mismatch")
    if len(min.shape) != 1 and len(min.shape) != 2:
        raise AssertionError("input dim unexpected")
    if len(min.shape) == 1:
        np.expand_dims(min, 1)
        np.expand_dims(max, 1)
        np.expand_dims(stddev, 1)
        np.expand_dims(mean, 1)
    nb_subplots = min.shape[1]
    fig, axs = plt.subplots(nb_subplots)
    xrange = range(len(labels))
    for ax_idx in range(nb_subplots):
        ax = axs[ax_idx]
        ax.locator_params(nbins=nb_subplots)
        ax.errorbar(xrange, mean[:, ax_idx], stddev[:, ax_idx], fmt='ok', lw=3)
        ax.errorbar(xrange, mean[:, ax_idx], [mean[:, ax_idx] - min[:, ax_idx], max[:, ax_idx] - mean[:, ax_idx]],
                    fmt='.k', ecolor='gray', lw=1)
        ax.set_xticks(xrange)
        ax.set_xticklabels(labels, visible=(ax_idx == nb_subplots - 1))
        ax.set_title("Band %d" % (ax_idx + 1))
        ax.tick_params(axis="x", labelsize="6", labelrotation=45)
    plt.tight_layout()
    fig.show()


def draw_roc_curve(fpr, tpr, labels=None, size_inch=(5, 5), dpi=320):
    if not isinstance(fpr, np.ndarray) or not isinstance(tpr, np.ndarray):
        raise AssertionError("invalid inputs")
    if fpr.shape != tpr.shape:
        raise AssertionError("mismatched input sizes")
    if fpr.ndim == 1:
        fpr = np.expand_dims(fpr, 0)
    if tpr.ndim == 1:
        tpr = np.expand_dims(tpr, 0)
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if len(labels) != fpr.shape[0]:
            raise AssertionError("should have one label per curve")
    else:
        labels = [None] * fpr.shape[0]
    fig = plt.figure(num="roc", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    for idx,label in enumerate(labels):
        auc = sklearn.metrics.auc(fpr[idx, ...], tpr[idx, ...])
        if label is not None:
            ax.plot(fpr[idx, ...], tpr[idx, ...], "b", label=("%s [auc = %0.3f]" % (label, auc)))
        else:
            ax.plot(fpr[idx, ...], tpr[idx, ...], "b", label=("auc = %0.3f" % auc))
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    fig.set_tight_layout(True)
    return fig


def draw_confmat(confmat, class_list, size_inch=(5, 5), dpi=320, normalize=False):
    if not isinstance(confmat, np.ndarray) or not isinstance(class_list, list):
        raise AssertionError("invalid inputs")
    if normalize:
        confmat = confmat.astype("float") / confmat.sum(axis=1)[:, np.newaxis]
        confmat = np.nan_to_num(confmat)
    fig = plt.figure(num="confmat", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(confmat, cmap=plt.cm.hot)
    labels = [clipstr(label, 9) for label in class_list]
    tick_marks = np.arange(len(labels))
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=4, rotation=-90, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.set_ylabel("Real", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va="center")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        str = format(confmat[i, j], "d") if confmat[i, j] != 0 else "."
        color = "blue" if i != j else "green"
        ax.text(j, i, str, horizontalalignment="center", fontsize=3, verticalalignment="center", color=color)
    fig.set_tight_layout(True)
    return fig


def stringify_confmat(confmat, class_list, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    if not isinstance(confmat, np.ndarray) or not isinstance(class_list, list):
        raise AssertionError("invalid inputs")
    columnwidth = 9
    empty_cell = " " * columnwidth
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    res = "\t" + fst_empty_cell + " "
    for label in class_list:
        res += ("%{0}s".format(columnwidth) % clipstr(label, columnwidth)) + " "
    res += ("%{0}s".format(columnwidth) % "total") + "\n"
    for idx_true, label in enumerate(class_list):
        res += ("\t%{0}s".format(columnwidth) % clipstr(label, columnwidth)) + " "
        for idx_pred, _ in enumerate(class_list):
            cell = "%{0}d".format(columnwidth) % int(confmat[idx_true, idx_pred])
            if hide_zeroes:
                cell = cell if int(confmat[idx_true, idx_pred]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if idx_true != idx_pred else empty_cell
            if hide_threshold:
                cell = cell if confmat[idx_true, idx_pred] > hide_threshold else empty_cell
            res += cell + " "
        res += ("%{0}d".format(columnwidth) % int(confmat[idx_true, :].sum())) + "\n"
    res += ("\t%{0}s".format(columnwidth) % "total") + " "
    for idx_pred, _ in enumerate(class_list):
        res += ("%{0}d".format(columnwidth) % int(confmat[:, idx_pred].sum())) + " "
    res += ("%{0}d".format(columnwidth) % int(confmat.sum())) + "\n"
    return res


def fig2array(fig):  # will return figure as numpy-compatible RGBA array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def get_glob_paths(input_glob_pattern, can_be_dir=False):
    glob_file_paths = glob.glob(input_glob_pattern)
    if not glob_file_paths:
        raise AssertionError("invalid input glob pattern '%s'" % input_glob_pattern)
    for file_path in glob_file_paths:
        if not os.path.isfile(file_path) and not (can_be_dir and os.path.isdir(file_path)):
            raise AssertionError("invalid input file at globbed path '%s'" % file_path)
    return glob_file_paths


def get_dataset_file_paths(input_path, dataset_root, allow_glob=False, can_be_dir=False):
    if os.path.isabs(input_path):
        if '*' in input_path and allow_glob:
            return get_glob_paths(input_path)
        elif not os.path.isfile(input_path) and not (can_be_dir and os.path.isdir(input_path)):
            raise AssertionError("invalid input file at absolute path '%s'" % input_path)
    else:
        if not os.path.isdir(dataset_root):
            raise AssertionError("invalid dataset root directory at '%s'" % dataset_root)
        input_path = os.path.join(dataset_root, input_path)
        if '*' in input_path and allow_glob:
            return get_glob_paths(input_path)
        elif not os.path.isfile(input_path) and not (can_be_dir and os.path.isdir(input_path)):
            raise AssertionError("invalid input file at path '%s'" % input_path)
    return [input_path]


def check_key(key, tdict, tdict_name, msg=''):
    if key not in tdict:
        if msg == '':
            raise AssertionError("%s missing '%s' field" % (tdict_name, key))
        else:
            raise AssertionError(msg)


def get_table_from_classification_report(classification_report):
    lines = classification_report.splitlines()
    header = lines[1].split()
    data = []
    for line in lines[2:-1]:
        els = line.split()
        if len(els):
            data.append(els)
    data = np.vstack(data).transpose()
    avg = lines[len(lines) - 1].split()
    return header, data, avg
