"""Drawing/display utilities module.

These functions currently rely on OpenCV and/or matplotlib.
"""
import itertools
import logging
import math
from typing import TYPE_CHECKING

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

import thelper.typedefs  # noqa: F401

if TYPE_CHECKING:
    from typing import Any, AnyStr, List, Optional  # noqa: F401

logger = logging.getLogger(__name__)

warned_generic_draw = False


def safe_crop(image, tl, br, bordertype=cv.BORDER_CONSTANT, borderval=0, force_copy=False):
    """Safely crops a region from within an image, padding borders if needed.

    Args:
        image: the image to crop (provided as a numpy array).
        tl: a tuple or list specifying the (x,y) coordinates of the top-left crop corner.
        br: a tuple or list specifying the (x,y) coordinates of the bottom-right crop corner.
        bordertype: border copy type to use when the image is too small for the required crop size.
            See ``cv2.copyMakeBorder`` for more information.
        borderval: border value to use when the image is too small for the required crop size. See
            ``cv2.copyMakeBorder`` for more information.
        force_copy: defines whether to force a copy of the target image region even when it can be
            avoided.

    Returns:
        The cropped image.
    """
    if not isinstance(image, np.ndarray):
        raise AssertionError("expected input image to be numpy array")
    if isinstance(tl, tuple):
        tl = list(tl)
    if isinstance(br, tuple):
        br = list(br)
    if not isinstance(tl, list) or not isinstance(br, list):
        raise AssertionError("expected tl/br coords to be provided as tuple or list")
    if tl[0] < 0 or tl[1] < 0 or br[0] > image.shape[1] or br[1] > image.shape[0]:
        image = cv.copyMakeBorder(image, max(-tl[1], 0), max(br[1] - image.shape[0], 0),
                                  max(-tl[0], 0), max(br[0] - image.shape[1], 0),
                                  borderType=bordertype, value=borderval)
        if tl[0] < 0:
            br[0] -= tl[0]
            tl[0] = 0
        if tl[1] < 0:
            br[1] -= tl[1]
            tl[1] = 0
        return image[tl[1]:br[1], tl[0]:br[0], ...]
    if force_copy:
        return np.copy(image[tl[1]:br[1], tl[0]:br[0], ...])
    return image[tl[1]:br[1], tl[0]:br[0], ...]


def get_bgr_from_hsl(hue, sat, light):
    """Converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet."""
    # this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
    if hue < 0 or hue > 360:
        raise AssertionError("invalid hue")
    if sat < 0 or sat > 1:
        raise AssertionError("invalid saturation")
    if light < 0 or light > 1:
        raise AssertionError("invalid lightness")
    if sat == 0:
        return (int(np.clip(round(light * 255), 0, 255)),) * 3
    if light == 0:
        return 0, 0, 0
    if light == 1:
        return 255, 255, 255

    def h2rgb(_p, _q, _t):
        if _t < 0:
            _t += 1
        if _t > 1:
            _t -= 1
        if _t < 1 / 6:
            return _p + (_q - _p) * 6 * _t
        if _t < 1 / 2:
            return _q
        if _t < 2 / 3:
            return _p + (_q - _p) * (2 / 3 - _t) * 6
        return _p

    q = light * (1 + sat) if (light < 0.5) else light + sat - light * sat
    p = 2 * light - q
    h = hue / 360
    return (int(np.clip(round(h2rgb(p, q, h - 1 / 3) * 255), 0, 255)),
            int(np.clip(round(h2rgb(p, q, h) * 255), 0, 255)),
            int(np.clip(round(h2rgb(p, q, h + 1 / 3) * 255), 0, 255)))


def get_displayable_image(image,                # type: thelper.typedefs.ArrayType
                          grayscale=False,      # type: Optional[bool]
                          ):                    # type: (...) -> thelper.typedefs.ArrayType
    """Returns a 'displayable' image that has been normalized and padded to three channels."""
    if image.ndim != 3:
        raise AssertionError("indexing should return a pre-squeezed array")
    if image.shape[2] == 2:
        image = np.dstack((image, image[:, :, 0]))
    elif image.shape[2] > 3:
        image = image[..., :3]
    if grayscale and image.shape[2] != 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif not grayscale and image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image_normalized = np.empty_like(image, dtype=np.uint8).copy()  # copy needed here due to ocv 3.3 bug
    cv.normalize(image, image_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return image_normalized


def get_displayable_heatmap(array,              # type: thelper.typedefs.ArrayType
                            convert_rgb=True,   # type: Optional[bool]
                            ):                  # type: (...) -> thelper.typedefs.ArrayType
    """Returns a 'displayable' array that has been min-maxed and mapped to color triplets."""
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise AssertionError("indexing should return a pre-squeezed array")
    array_normalized = np.empty_like(array, dtype=np.uint8).copy()  # copy needed here due to ocv 3.3 bug
    cv.normalize(array, array_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmap = cv.applyColorMap(array_normalized, cv.COLORMAP_JET)
    if convert_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    return heatmap


def draw_histogram(data,                 # type: thelper.typedefs.ArrayType
                   bins=50,              # type: Optional[int]
                   xlabel="",            # type: Optional[thelper.typedefs.LabelType]
                   ylabel="Proportion",  # type: Optional[thelper.typedefs.LabelType]
                   show=False,           # type: Optional[bool]
                   block=False,          # type: Optional[bool]
                   ):                    # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns a histogram figure using pyplot."""
    fig, ax = plt.subplots()
    ax.hist(data, density=True, bins=bins)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=0)
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, ax


def draw_popbars(labels,                # type: thelper.typedefs.LabelList
                 counts,                # type: int
                 xlabel="",             # type: Optional[thelper.typedefs.LabelType]
                 ylabel="Pop. Count",   # type: Optional[thelper.typedefs.LabelType]
                 show=False,            # type: Optional[bool]
                 block=False,           # type: Optional[bool]
                 ):                     # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns a bar histogram figure using pyplot."""
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
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, ax


def draw_pascalvoc_curve(metrics, size_inch=(5, 5), dpi=160, show=False, block=False):
    """Draws and returns a precision-recall curve according to pascalvoc metrics."""
    # note: the 'metrics' must correspond to a single class output produced by pascalvoc evaluator
    assert isinstance(metrics, dict), "unexpected metrics format"
    class_name = metrics["class_name"]
    assert isinstance(class_name, str), "unexpected class name type"
    iou_threshold = metrics["iou_threshold"]
    assert 0 < iou_threshold <= 1, "invalid intersection over union value (should be in ]0,1])"
    method = metrics["eval_method"]
    assert method in ["all-points", "11-points"], "invalid method (should be 'all-points' or '11-points')"
    fig = plt.figure(num="pr", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["recall"], metrics["precision"],
            label=f"{class_name} (AP={metrics['AP'] * 100:.2f}%)")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PascalVOC PR Curve @ {iou_threshold} IoU")
    ax.legend(loc="upper right")
    ax.grid()
    fig.set_tight_layout(True)
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, ax


def draw_images(images,               # type: thelper.typedefs.OneOrManyArrayType
                captions=None,        # type: Optional[List[str]]
                redraw=None,          # type: Optional[thelper.typedefs.DrawingType]
                show=True,            # type: Optional[bool]
                block=False,          # type: Optional[bool]
                use_cv2=True,         # type: Optional[bool]
                cv2_flip_bgr=True,    # type: Optional[bool]
                img_shape=None,       # type: Optional[thelper.typedefs.ArrayShapeType]
                max_img_size=None,    # type: Optional[thelper.typedefs.ArrayShapeType]
                grid_size_x=None,     # type: Optional[int]
                grid_size_y=None,     # type: Optional[int]
                caption_opts=None,
                window_name=None,     # type: Optional[str]
                ):                    # type: (...) -> thelper.typedefs.DrawingType
    """Draws a set of images with optional captions."""
    nb_imgs = len(images) if isinstance(images, list) else images.shape[0]
    if nb_imgs < 1:
        return None
    assert captions is None or len(captions) == nb_imgs, "captions count mismatch with image count"
    # for display on typical monitors... (height, width)
    max_img_size = (800, 1600) if max_img_size is None else max_img_size
    grid_size_x = int(math.ceil(math.sqrt(nb_imgs))) if grid_size_x is None else grid_size_x
    grid_size_y = int(math.ceil(nb_imgs / grid_size_x)) if grid_size_y is None else grid_size_y
    assert grid_size_x * grid_size_y >= nb_imgs, f"bad gridding for subplots (need at least {nb_imgs} tiles)"
    if use_cv2:
        if caption_opts is None:
            caption_opts = {
                "org": (10, 40),
                "fontFace": cv.FONT_HERSHEY_SIMPLEX,
                "fontScale": 0.40,
                "color": (255, 255, 255),
                "thickness": 1,
                "lineType": cv.LINE_AA
            }
        if window_name is None:
            window_name = "images"
        img_grid_shape = None
        img_grid = None if redraw is None else redraw[1]
        for img_idx in range(nb_imgs):
            image = images[img_idx] if isinstance(images, list) else images[img_idx, ...]
            if img_shape is None:
                img_shape = image.shape
            if img_grid_shape is None:
                img_grid_shape = (img_shape[0] * grid_size_y, img_shape[1] * grid_size_x, img_shape[2])
            if img_grid is None or img_grid.shape != img_grid_shape:
                img_grid = np.zeros(img_grid_shape, dtype=np.uint8)
            if image.shape[2] != img_shape[2]:
                raise AssertionError(f"unexpected image depth ({image.shape[2]} vs {img_shape[2]})")
            if image.shape != img_shape:
                image = cv.resize(image, (img_shape[1], img_shape[0]), interpolation=cv.INTER_NEAREST)
            if captions is not None and str(captions[img_idx]):
                image = cv.putText(image.copy(), str(captions[img_idx]), **caption_opts)
            offsets = (img_idx // grid_size_x) * img_shape[0], (img_idx % grid_size_x) * img_shape[1]
            np.copyto(img_grid[offsets[0]:(offsets[0] + img_shape[0]), offsets[1]:(offsets[1] + img_shape[1]), :], image)
        win_name = str(window_name) if redraw is None else redraw[0]
        if img_grid is not None:
            display = img_grid[..., ::-1] if cv2_flip_bgr else img_grid
            if display.shape[0] > max_img_size[0] or display.shape[1] > max_img_size[1]:
                if display.shape[0] / max_img_size[0] > display.shape[1] / max_img_size[1]:
                    dsize = (max_img_size[0], int(round(display.shape[1] / (display.shape[0] / max_img_size[0]))))
                else:
                    dsize = (int(round(display.shape[0] / (display.shape[1] / max_img_size[1]))), max_img_size[1])
                display = cv.resize(display, (dsize[1], dsize[0]))
            if show:
                cv.imshow(win_name, display)
                cv.waitKey(0 if block else 1)
        return win_name, img_grid
    else:
        fig, axes = redraw if redraw is not None else plt.subplots(grid_size_y, grid_size_x)
        if nb_imgs == 1:
            axes = np.array(axes)
        for ax_idx, ax in enumerate(axes.reshape(-1)):
            if ax_idx < nb_imgs:
                image = images[ax_idx] if isinstance(images, list) else images[ax_idx, ...]
                if image.shape != img_shape:
                    image = cv.resize(image, (img_shape[1], img_shape[0]), interpolation=cv.INTER_NEAREST)
                ax.imshow(image, interpolation='nearest')
                if captions is not None and str(captions[ax_idx]):
                    ax.set_xlabel(str(captions[ax_idx]))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.set_tight_layout(True)
        if show:
            fig.show()
            if block:
                plt.show(block=block)
                return None
            plt.pause(0.5)
        return fig, axes


def draw_predicts(images, preds=None, targets=None, swap_channels=False, redraw=None, block=False, **kwargs):
    """Draws and returns a set of generic prediction results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    image_gray_list = [cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) for image in image_list]
    nb_imgs = len(image_list)
    caption_list = [""] * nb_imgs
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    if targets is not None:
        if not isinstance(targets, list) and not (isinstance(targets, torch.Tensor) and targets.shape[0] == nb_imgs):
            raise AssertionError("expected targets to be in list or tensor format (Bx...)")
        if isinstance(targets, list):
            if all([isinstance(t, list) for t in targets]):
                targets = list(itertools.chain.from_iterable(targets))  # merge all augmented lists together
            targets = torch.cat(targets, 0)  # merge all masks into a single tensor
        if targets.shape[0] != nb_imgs:
            raise AssertionError("images/targets count mismatch")
        targets = targets.numpy()
        if swap_channels:
            if not targets.ndim == 4:
                raise AssertionError("unexpected swap for targets tensor that is not 4-dim")
            targets = np.transpose(targets, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
        if ((targets.ndim == 4 and targets.shape[1] == 1) or targets.ndim == 3) and targets.shape[-2:] == images.shape[1:3]:
            target_list = [get_displayable_heatmap(targets[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            target_list = [cv.addWeighted(image_gray_list[idx], 0.3, target_list[idx], 0.7, 0) for idx in range(nb_imgs)]
            image_list += target_list
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        elif targets.shape == images.shape:
            image_list += [get_displayable_image(targets[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        else:
            for idx in range(nb_imgs):
                caption_list[idx] = f"GT={str(targets[idx])}"
    if preds is not None:
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.shape[0] == nb_imgs):
            raise AssertionError("expected preds to be in list or tensor shape (Bx...)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            preds = torch.cat(preds, 0)  # merge all preds into a single tensor
        if preds.shape[0] != nb_imgs:
            raise AssertionError("images/preds count mismatch")
        preds = preds.numpy()
        if swap_channels:
            if not preds.ndim == 4:
                raise AssertionError("unexpected swap for targets tensor that is not 4-dim")
            preds = np.transpose(preds, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
        if targets is not None and preds.shape != targets.shape:
            raise AssertionError("preds/targets shape mismatch")
        if ((preds.ndim == 4 and preds.shape[1] == 1) or preds.ndim == 3) and preds.shape[-2:] == images.shape[1:3]:
            pred_list = [get_displayable_heatmap(preds[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            pred_list = [cv.addWeighted(image_gray_list[idx], 0.3, pred_list[idx], 0.7, 0) for idx in range(nb_imgs)]
            image_list += pred_list
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        elif preds.shape == images.shape:
            image_list += [get_displayable_image(preds[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        else:
            for idx in range(nb_imgs):
                if len(caption_list[idx]) != 0:
                    caption_list[idx] += ", "
                caption_list[idx] = f"Pred={str(preds[idx])}"
    return draw_images(image_list, captions=caption_list, redraw=redraw, window_name="predictions",
                       block=block, grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def draw_segments(images, preds=None, masks=None, color_map=None, redraw=None, block=False,
                  segm_threshold=None, target_class=None, target_threshold=None, **kwargs):
    """Draws and returns a set of segmentation results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    image_gray_list = [cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) for image in image_list]
    nb_imgs = len(image_list)
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    if color_map is not None and isinstance(color_map, dict):
        assert len(color_map) <= 256, "too many indices for uint8 map"
        use_alpha = all([isinstance(val, np.ndarray) and val.dtype in (np.float32, np.float64)
                         for val in color_map.values()])
        color_map_new = np.zeros((256, 1, 3), dtype=np.float32 if use_alpha else np.uint8)
        for idx, val in color_map.items():
            color_map_new[idx, ...] = val
        color_map = color_map_new
    if masks is not None:
        if not isinstance(masks, list) and not (isinstance(masks, torch.Tensor) and masks.dim() == 3):
            if isinstance(masks, torch.Tensor) and masks.dim() == 4 \
                    and masks.shape[0] == images.shape[0] and masks.shape[1] == 1:
                masks = masks.squeeze(axis=1)
            else:
                raise AssertionError("expected segmentation masks to be in list or 3-d tensor format (BxHxW)")
        if isinstance(masks, list):
            if all([isinstance(m, list) for m in masks]):
                masks = list(itertools.chain.from_iterable(masks))  # merge all augmented lists together
            masks = torch.cat(masks, 0)  # merge all masks into a single tensor
        if masks.shape[0] != nb_imgs:
            raise AssertionError("images/masks count mismatch")
        if images.shape[0:3] != masks.shape:
            raise AssertionError("images/masks shape mismatch")
        masks = masks.numpy()
        if color_map is not None:
            masks = [apply_color_map(masks[idx], color_map) for idx in range(masks.shape[0])]
        image_list += [cv.addWeighted(image_gray_list[idx], 0.3, masks[idx], 0.7, 0)
                       if masks[idx].dtype == np.uint8 else (image_list[idx] * masks[idx]).astype(np.uint8)
                       for idx in range(nb_imgs)]
        grid_size_y += 1
    if preds is not None:
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.dim() == 4):
            raise AssertionError("expected segmentation preds to be in list or 3-d tensor format (BxCxHxW)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            preds = torch.cat(preds, 0)  # merge all preds into a single tensor
        with torch.no_grad():
            if target_class is not None:
                assert isinstance(target_class, int), "target class should be index (integer)"
                assert isinstance(target_threshold, float), "target threshold should be float"
                assert 0 < target_threshold < 1, "target threshold should be in [0,1]"
                preds_softmax = torch.nn.functional.softmax(preds, dim=1)
                preds = (preds_softmax[:, target_class, ...] > target_threshold).long()
            else:
                preds = torch.squeeze(preds.topk(k=1, dim=1)[1], dim=1)  # keep top prediction index only
        if preds.shape[0] != nb_imgs:
            raise AssertionError("images/preds count mismatch")
        if images.shape[0:3] != preds.shape:
            raise AssertionError("images/preds shape mismatch")
        preds = preds.numpy()
        if color_map is not None:
            preds = [apply_color_map(preds[idx], color_map) for idx in range(preds.shape[0])]
        image_list += [cv.addWeighted(image_gray_list[idx], 0.3, preds[idx], 0.7, 0)
                       if preds[idx].dtype == np.uint8 else (image_list[idx] * preds[idx]).astype(np.uint8)
                       for idx in range(nb_imgs)]
        grid_size_y += 1
    return draw_images(image_list, redraw=redraw, window_name="segments", block=block,
                       grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def draw_classifs(images, preds=None, labels=None, class_names_map=None, redraw=None, block=False, **kwargs):
    """Draws and returns a set of classification results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    caption_list = [""] * len(image_list)
    if labels is not None:  # convert labels to flat list, if available
        if not isinstance(labels, list) and not (isinstance(labels, torch.Tensor) and labels.dim() == 1):
            raise AssertionError("expected classification labels to be in list or 1-d tensor format")
        if isinstance(labels, list):
            if all([isinstance(lbl, list) for lbl in labels]):
                labels = list(itertools.chain.from_iterable(labels))  # merge all augmented lists together
            if all([isinstance(t, torch.Tensor) for t in labels]):
                labels = torch.cat(labels, 0)
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        if images.shape[0] != len(labels):
            raise AssertionError("images/labels count mismatch")
        if class_names_map is not None:
            labels = [class_names_map[lbl] if lbl in class_names_map else lbl for lbl in labels]
        for idx in range(len(image_list)):
            caption_list[idx] = f"GT={labels[idx]}"
    if preds is not None:  # convert predictions to flat list, if available
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.dim() == 2):
            raise AssertionError("expected classification predictions to be in list or 2-d tensor format (BxC)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            if all([isinstance(t, torch.Tensor) for t in preds]):
                preds = torch.cat(preds, 0)
        with torch.no_grad():
            preds = torch.squeeze(preds.topk(1, dim=1)[1], dim=1)
        if images.shape[0] != preds.shape[0]:
            raise AssertionError("images/predictions count mismatch")
        preds = preds.tolist()
        if class_names_map is not None:
            preds = [class_names_map[lbl] if lbl in class_names_map else lbl for lbl in preds]
        for idx in range(len(image_list)):
            if len(caption_list[idx]) != 0:
                caption_list[idx] += ", "
            caption_list[idx] += f"Pred={preds[idx]}"
    return draw_images(image_list, captions=caption_list, redraw=redraw, window_name="classifs", block=block, **kwargs)


def draw(task, input, pred=None, target=None, block=False, ch_transpose=True, flip_bgr=False, redraw=None, **kwargs):
    """Draws and returns a figure of a model input/predictions/targets using pyplot or OpenCV."""
    # note: this function actually dispatches the drawing procedure using the task interface
    import thelper.tasks
    if not isinstance(task, thelper.tasks.Task):
        raise AssertionError("invalid task object")
    if isinstance(input, list) and all([isinstance(t, torch.Tensor) for t in input]):
        # if we have a list, it must be due to a augmentation stage
        if not all([image.shape == input[0].shape for image in input]):
            raise AssertionError("image shape mismatch throughout list")
        input = torch.cat(input, 0)  # merge all images into a single tensor
    if not isinstance(input, torch.Tensor) or input.dim() != 4:
        raise AssertionError("expected input images to be in 4-d tensor format (BxCxHxW or BxHxWxC)")
    input = input.numpy().copy()
    if ch_transpose:
        input = np.transpose(input, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
    if flip_bgr:
        input = input[..., ::-1]  # BGR to RGB
    if pred is not None and isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach()  # avoid latency for preprocessing on gpu
    if target is not None and isinstance(target, torch.Tensor):
        target = target.cpu().detach()  # avoid latency for preprocessing on gpu
    if isinstance(task, thelper.tasks.Classification):
        class_names_map = {idx: name for name, idx in task.class_indices.items()}
        return draw_classifs(images=input, preds=pred, labels=target,
                             class_names_map=class_names_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Segmentation):
        color_map = task.color_map if task.color_map else {idx: get_label_color_mapping(idx + 1) for idx in task.class_indices.values()}
        if task.dontcare is not None and task.dontcare not in color_map:
            color_map[task.dontcare] = np.asarray([0, 0, 0])
        return draw_segments(images=input, preds=pred, masks=target, color_map=color_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Detection):
        color_map = task.color_map if task.color_map else {idx: get_label_color_mapping(idx) for idx in task.class_indices.values()}
        return draw_bboxes(images=input, preds=pred, bboxes=target, color_map=color_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Regression):
        swap_channels = isinstance(task, thelper.tasks.SuperResolution)  # must update BxCxHxW to BxHxWxC in targets/preds
        # @@@ todo: cleanup swap_channels above via flag in superres task?
        return draw_predicts(images=input, preds=pred, targets=target,
                             swap_channels=swap_channels, redraw=redraw, block=block, **kwargs)
    else:
        global warned_generic_draw
        if not warned_generic_draw:
            logger.warning("unhandled drawing mode, defaulting to input display only")
            warned_generic_draw = True
        image_list = [get_displayable_image(input[batch_idx, ...]) for batch_idx in range(input.shape[0])]
        return draw_images(image_list, redraw=redraw, window_name="inputs", block=block, **kwargs)


# noinspection PyUnusedLocal
def draw_errbars(labels,                # type: thelper.typedefs.LabelList
                 min_values,            # type: thelper.typedefs.ArrayType
                 max_values,            # type: thelper.typedefs.ArrayType
                 stddev_values,         # type: thelper.typedefs.ArrayType
                 mean_values,           # type: thelper.typedefs.ArrayType
                 xlabel="",             # type: thelper.typedefs.LabelType
                 ylabel="Raw Value",    # type: thelper.typedefs.LabelType
                 show=False,            # type: Optional[bool]
                 block=False,           # type: Optional[bool]
                 ):                     # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns an error bar histogram figure using pyplot."""
    if min_values.shape != max_values.shape \
            or min_values.shape != stddev_values.shape \
            or min_values.shape != mean_values.shape:
        raise AssertionError("input dim mismatch")
    if len(min_values.shape) != 1 and len(min_values.shape) != 2:
        raise AssertionError("input dim unexpected")
    if len(min_values.shape) == 1:
        np.expand_dims(min_values, 1)
        np.expand_dims(max_values, 1)
        np.expand_dims(stddev_values, 1)
        np.expand_dims(mean_values, 1)
    nb_subplots = min_values.shape[1]
    fig, axs = plt.subplots(nb_subplots)
    xrange = range(len(labels))
    for ax_idx in range(nb_subplots):
        ax = axs[ax_idx]
        ax.locator_params(nbins=nb_subplots)
        ax.errorbar(xrange, mean_values[:, ax_idx], stddev_values[:, ax_idx], fmt='ok', lw=3)
        ax.errorbar(xrange, mean_values[:, ax_idx], [mean_values[:, ax_idx] - min_values[:, ax_idx],
                                                     max_values[:, ax_idx] - mean_values[:, ax_idx]],
                    fmt='.k', ecolor='gray', lw=1)
        ax.set_xticks(xrange)
        ax.set_xticklabels(labels, visible=(ax_idx == nb_subplots - 1))
        ax.set_title("Band %d" % (ax_idx + 1))
        ax.tick_params(axis="x", labelsize="6", labelrotation=45)
    fig.set_tight_layout(True)
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, axs


def draw_roc_curve(fpr, tpr, labels=None, size_inch=(5, 5), dpi=160, show=False, block=False):
    """Draws and returns an ROC curve figure using pyplot."""
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
    import sklearn.metrics
    for idx, label in enumerate(labels):
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
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, ax


def draw_confmat(confmat, class_list, size_inch=(5, 5), dpi=160, normalize=False,
                 keep_unset=False, show=False, block=False):
    """Draws and returns an a confusion matrix figure using pyplot."""
    if not isinstance(confmat, np.ndarray) or not isinstance(class_list, list):
        raise AssertionError("invalid inputs")
    if confmat.ndim != 2:
        raise AssertionError("invalid confmat shape")
    if not keep_unset and "<unset>" in class_list:
        unset_idx = class_list.index("<unset>")
        del class_list[unset_idx]
        np.delete(confmat, unset_idx, 0)
        np.delete(confmat, unset_idx, 1)
    if normalize:
        row_sums = confmat.sum(axis=1)[:, np.newaxis]
        confmat = np.nan_to_num(confmat.astype(np.float) / np.maximum(row_sums, 0.0001))
    fig = plt.figure(num="confmat", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(confmat, cmap=plt.cm.Blues, aspect="equal", interpolation="none")
    import thelper.utils
    labels = [thelper.utils.clipstr(label, 9) for label in class_list]
    tick_marks = np.arange(len(labels))
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=4, rotation=-90, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.set_ylabel("Real", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va="center")
    ax.set_ylim(confmat.shape[0] - 0.5, -0.5)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        if not normalize:
            txt = ("%d" % confmat[i, j]) if confmat[i, j] != 0 else "."
        else:
            if confmat[i, j] >= 0.01:
                txt = "%.02f" % confmat[i, j]
            else:
                txt = "~0" if confmat[i, j] > 0 else "."
        color = "white" if confmat[i, j] > thresh else "black"
        ax.text(j, i, txt, horizontalalignment="center", fontsize=4, verticalalignment="center", color=color)
    fig.set_tight_layout(True)
    if show:
        fig.show()
        if block:
            plt.show(block=block)
            return fig
        plt.pause(0.5)
    return fig, ax


def draw_bbox(image, tl, br, text, color, box_thickness=2, font_thickness=1,
              font_scale=0.4, show=False, block=False, win_name="bbox"):
    """Draws a single bounding box on a given image (used in :func:`thelper.draw.draw_bboxes`)."""
    tl, br = (round(float(tl[0])), round(float(tl[1]))), \
             (round(float(br[0])), round(float(br[1])))
    text_size, baseline = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                         fontScale=font_scale, thickness=font_thickness)
    text_bl = (tl[0] + box_thickness + 1, tl[1] + text_size[1] + box_thickness + 1)
    # note: text will overflow if box is too small
    text_box_br = (text_bl[0] + text_size[0] + box_thickness, text_bl[1] + box_thickness * 2)
    cv.rectangle(image, (tl[0] - 1, tl[1] - 1), (text_box_br[0] + 1, text_box_br[1] + 1),
                 color=(0, 0, 0), thickness=-1)
    cv.rectangle(image, tl, br, color=(0, 0, 0), thickness=(box_thickness + 1))
    cv.rectangle(image, tl, br, color=color, thickness=box_thickness)
    cv.rectangle(image, tl, text_box_br, color=color, thickness=-1)
    cv.putText(image, text, text_bl, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
               color=(0, 0, 0), thickness=font_thickness + 1)
    cv.putText(image, text, text_bl, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
               color=(255, 255, 255), thickness=font_thickness)
    if show:
        cv.imshow(win_name, image)
        cv.waitKey(0 if block else 1)
    return win_name, image


def draw_bboxes(images,                 # type: thelper.typedefs.ImageArray
                preds=None,             # type: Optional[thelper.typedefs.BoundingBoxArray]
                bboxes=None,            # type: Optional[thelper.typedefs.BoundingBoxArray]
                color_map=None,         # type: Optional[thelper.typedefs.ClassColorMap]
                redraw=None,            # type: Optional[thelper.typedefs.DrawingType]
                block=False,            # type: Optional[bool]
                min_confidence=0.5,     # type: thelper.typedefs.Number
                class_map=None,         # type: Optional[thelper.typedefs.ClassIdType, AnyStr]
                **kwargs                # type: Any
                ):
    """Draws a set of bounding box prediction results on images.

    Args:
        images: images with first dimension as list index, and other dimensions are each image's content
        preds: predicted bounding boxes per image to be displayed, must match images count if provided
        bboxes: ground truth (targets) bounding boxes per image to be displayed, must match images count if provided
        color_map: mapping of class-id to color to be applied to drawn bounding boxes on the image
        redraw: existing figure and axes to reuse for drawing the new images and bounding boxes
        block: indicate whether to block execution until all figures have been closed or not
        min_confidence: ignore display of bounding boxes that have a confidence below this value, if available
        class_map: alternative class-id to class-name mapping to employ for display.
            This overrides the default class names retrieved from each bounding box's attributed task.
            Useful for displaying generic bounding boxes obtained from raw input values without a specific task.
        kwargs: other arguments to be passed down to further drawing functions or drawing settings
            (amongst other settings, box_thickness, font_thickness and font_scale can be provided)
    """
    def get_class_name(_bbox):
        if isinstance(class_map, dict):
            return class_map[_bbox.class_id]
        elif bbox.task is not None:
            return _bbox.task.class_names[_bbox.class_id]
        else:
            raise RuntimeError("could not find class name from either class mapping or bbox task definition")

    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    if color_map is not None and isinstance(color_map, dict):
        assert len(color_map) <= 256, "too many indices for uint8 map"
        color_map_new = np.zeros((256, 3), dtype=np.uint8)
        for idx, val in color_map.items():
            color_map_new[idx, ...] = val
        color_map = color_map_new.tolist()
    nb_imgs = len(image_list)
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    box_thickness = thelper.utils.get_key_def("box_thickness", kwargs, default=2, delete=True)
    font_thickness = thelper.utils.get_key_def("font_thickness", kwargs, default=1, delete=True)
    font_scale = thelper.utils.get_key_def("font_scale", kwargs, default=0.4, delete=True)
    if preds is not None:
        assert len(image_list) == len(preds)
        for preds_list, image in zip(preds, image_list):
            for bbox_idx, bbox in enumerate(preds_list):
                assert isinstance(bbox, thelper.data.BoundingBox), "unrecognized bbox type"
                if bbox.confidence is not None and bbox.confidence < min_confidence:
                    continue
                color = get_bgr_from_hsl(bbox_idx / len(preds_list) * 360, 1.0, 0.5) \
                    if color_map is None else color_map[bbox.class_id]
                conf = ""
                if thelper.utils.is_scalar(bbox.confidence):
                    conf = f" ({bbox.confidence:.3f})"
                elif isinstance(bbox.confidence, (list, tuple, np.ndarray)):
                    conf = f" ({bbox.confidence[bbox.class_id]:.3f})"
                draw_bbox(image, bbox.top_left, bbox.bottom_right, f"{get_class_name(bbox)} {conf}",
                          color, box_thickness=box_thickness, font_thickness=font_thickness, font_scale=font_scale)
    if bboxes is not None:
        assert len(image_list) == len(bboxes), "mismatched bboxes list and image list sizes"
        clean_image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
        for bboxes_list, image in zip(bboxes, clean_image_list):
            for bbox_idx, bbox in enumerate(bboxes_list):
                assert isinstance(bbox, thelper.data.BoundingBox), "unrecognized bbox type"
                color = get_bgr_from_hsl(bbox_idx / len(bboxes_list) * 360, 1.0, 0.5) \
                    if color_map is None else color_map[bbox.class_id]
                draw_bbox(image, bbox.top_left, bbox.bottom_right, f"GT: {get_class_name(bbox)}",
                          color, box_thickness=box_thickness, font_thickness=font_thickness, font_scale=font_scale)
        grid_size_y += 1
        image_list += clean_image_list
    return draw_images(image_list, redraw=redraw, window_name="detections", block=block,
                       grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""
    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0
    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def get_label_html_color_code(idx):
    """Returns the PASCAL VOC HTML color code for a given label index."""
    color_array = get_label_color_mapping(idx)
    return f"#{color_array[0]:02X}{color_array[1]:02X}{color_array[2]:02X}"


def apply_color_map(image, colormap, dst=None):
    """Applies a color map to an image of 8-bit color indices; works similarly to cv2.applyColorMap (v3.3.1)."""
    assert isinstance(image, np.ndarray) and image.ndim == 2 and \
        np.issubdtype(image.dtype, np.integer), "invalid input image"
    assert isinstance(colormap, np.ndarray) and colormap.shape == (256, 1, 3) and \
        (colormap.dtype == np.uint8 or colormap.dtype == np.float32), "invalid color map"
    out_shape = (image.shape[0], image.shape[1], 3)
    if dst is None:
        dst = np.empty(out_shape, dtype=colormap.dtype)
    else:
        assert isinstance(dst, np.ndarray) and dst.shape == out_shape and \
            dst.dtype == colormap.dtype, "invalid output image"
    # using np.take might avoid an extra allocation...
    np.copyto(dst, colormap.squeeze()[image.ravel(), :].reshape(out_shape))
    return dst


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf
