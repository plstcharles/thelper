"""Graphical User Interface (GUI) annotator module.

This module contains various annotators that define interactive ways to visualize and annotate
data loaded via a dataset parser.
"""

import abc
import collections
import json
import logging
import os

import cv2 as cv
import numpy as np

import thelper.utils

logger = logging.getLogger(__name__)


class Annotator:
    """Abstract annotation tool used to define common functions for all GUI-based I/O.

    Example configuration file::

        # ...
        "annotator": {
            # type of annotator to instantiate
            "type": "thelper.gui.ImageSegmentAnnotator",
            # ...
            # provide all extra parameters to the specialized anntator here
            "params": {
                # ...
            }
        },
        # ...

    .. seealso::
        | :class:`thelper.gui.annotators.ImageSegmentAnnotator`
        | :func:`thelper.gui.utils.create_annotator`
    """

    def __init__(self, session_name, config, save_dir, datasets):
        """Receives the annotator configuration dictionary, parses it, and sets up the basic session attributes."""
        if config is None or not isinstance(config, dict):
            raise AssertionError("invalid input config type")
        if "annotator" not in config or not config["annotator"]:
            raise AssertionError("config missing 'annotator' field")
        if save_dir is None or not os.path.isdir(save_dir):
            raise AssertionError("invalid output directory '%s'" % save_dir)
        if datasets is None or not isinstance(datasets, dict):
            raise AssertionError("invalid input dataset parser dictionary")
        self.config = config
        self.annotator_config = thelper.utils.get_key_def(["params", "parameters"], config["annotator"], {})
        self.logger = thelper.utils.get_class_logger()
        self.name = session_name
        self.save_dir = save_dir
        self.datasets = datasets
        self.logs_dir = os.path.join(self.save_dir, "logs")
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.annotations_dir = os.path.join(self.save_dir, "annotations")
        if not os.path.isdir(self.annotations_dir):
            os.makedirs(self.annotations_dir)
        annot_logger_path = os.path.join(self.save_dir, "logs", "annotator.log")
        annot_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        annot_logger_fh = logging.FileHandler(annot_logger_path)
        annot_logger_fh.setFormatter(annot_logger_format)
        self.logger.addHandler(annot_logger_fh)
        self.logger.info("created annotation log for session '%s'" % self.name)
        logstamp = thelper.utils.get_log_stamp()
        repover = thelper.__version__ + ":" + thelper.utils.get_git_stamp()
        self.logger.debug("logstamp = %s" % logstamp)
        self.logger.debug("version = %s" % repover)

    @abc.abstractmethod
    def run(self):
        """Displays the GUI tool and blocks until it it closed by the user."""
        raise NotImplementedError


class ImageSegmentAnnotator(Annotator):
    """Annotator interface specialized for image segmentation annotation generation.

    This interface will create a GUI tool with a brush and a zoomed tooltip window that allows images
    to be painted over with predetermined class labels. The generated masks will then be saved in the
    session directory as PNG images.

    The configuration is expected to provide values for at least the following parameters:

    - ``sample_input_key``: specifies the key to use when extracting images from loaded samples. This
      is typically a string defined by the dataset parser.
    - ``labels``: provides a list of labels that will be available to use in the GUI. These labels are
      expected to be given as dictionaries that each define an ``id`` (uint8 value used in output masks),
      a ``name`` (string used for display/lookup purposes), and a color (3-element integer tuple).

    Other parameters can also be provided to alter the GUI's default behavior:

    - ``default_brush_radius``: the size of the brush at startup (default=10).
    - ``background_id``: the integer id to use for the background label (default=0).
    - ``window_zoom_crop_size``: the crop size displayed in the zoom tooltip (default=250x250).
    - ``window_zoom_size``: the size of the zoom tooltip window (default=500x500).
    - ``zoom_interp_type``: the interpolation type to use when zooming (default=cv2.INTER_NEAREST).
    - ``start_sample_idx``: the index of the first sample to display (default=0).
    - ``window_name``: the name of the main display window (default=image-segm-annotator).
    - ``window_size``: the size of the main display window (default=1000).
    - ``brush_thickness``: the size of the GUI brush tooltip border display (default=2).
    - ``gui_bar_size``: the width of the GUI bar displayed on top of the main window (default=50).
    - ``default_mask_opacity``: the default opacity of the segmentation mask (default=0.3).
    - ``default_fill_id``: the label id to fill all new masks with (default=0).

    .. seealso::
        | :class:`thelper.gui.annotators.Annotator`
    """

    # static variables used for callbacks and comms with other classes
    LATEST_PT = (-1, -1)
    LATEST_RAW_PT = (-1, -1)
    GUI_BAR_SIZE = None
    WINDOW_SIZE = None
    MOUSE_FLAGS = 0
    BRUSH_SIZE = None
    CURRENT_KEY = -1
    MASK_DIRTY = True
    GUI_DIRTY = True

    @staticmethod
    def on_press(key):
        """Callback entrypoint for pynput to register keyboard presses."""
        ImageSegmentAnnotator.CURRENT_KEY = key

    @staticmethod
    def on_mouse(event, x, y, flags, param):
        """Callback entrypoint for opencv to register mouse movement/clicks."""
        cls = ImageSegmentAnnotator
        cls.LATEST_RAW_PT = (x, y)
        cls.LATEST_PT = (x / cls.WINDOW_SIZE[0], (y - cls.GUI_BAR_SIZE) / cls.WINDOW_SIZE[1])
        cls.MOUSE_FLAGS = flags
        if event == cv.EVENT_MOUSEWHEEL:
            delta = flags >> 16
            if delta > 0:
                cls.BRUSH_SIZE = min(max(int(cls.BRUSH_SIZE * 1.1), cls.BRUSH_SIZE + 1), 100)
            else:
                cls.BRUSH_SIZE = max(min(int(cls.BRUSH_SIZE * 0.9), cls.BRUSH_SIZE - 1), 3)
        cls.GUI_DIRTY = True

    class Brush:
        """Brush manager used to refresh/draw mask contents based on mouse input."""

        def __init__(self, config):
            """Parses the input config and extracts brush-related parameters."""
            ImageSegmentAnnotator.BRUSH_SIZE = int(thelper.utils.get_key_def("default_brush_radius", config, 10))
            self.background_id = int(thelper.utils.get_key_def("background_id", config, 0))
            self.last_coords = collections.deque()

        def refresh(self, mask, label):
            """Fetches the latest mouse state and updates the mask if necessary."""
            cls = ImageSegmentAnnotator
            if not cls.GUI_DIRTY:
                return
            if 0 <= cls.LATEST_PT[0] <= 1 and 0 <= cls.LATEST_PT[1] <= 1 and \
               (cls.MOUSE_FLAGS & cv.EVENT_FLAG_LBUTTON or cls.MOUSE_FLAGS & cv.EVENT_FLAG_RBUTTON):
                coords = (int(cls.LATEST_PT[0] * mask.shape[1]), int(cls.LATEST_PT[1] * mask.shape[0]))
                self.last_coords.append(coords)
                if len(self.last_coords) == 1:
                    self.last_coords.append(coords)
                for start, stop in zip(list(self.last_coords)[0:], list(self.last_coords)[1:]):
                    if cls.MOUSE_FLAGS & cv.EVENT_FLAG_LBUTTON:
                        self.draw_stroke(mask, label["id"], start, stop)
                    else:
                        self.draw_stroke(mask, self.background_id, start, stop)
                if len(self.last_coords) > 2:
                    self.last_coords.popleft()
                cls.MASK_DIRTY = True
            else:
                self.last_coords = collections.deque()

        @staticmethod
        def draw_stroke(mask, label_id, start, end):
            """Draws a brush stroke on the mask with a given label id between two points."""
            cls = ImageSegmentAnnotator
            drag_len = cv.norm(start, end)
            brush_step_size = 1  # min(max(BRUSH_SIZE / 4, 1), 5)  # try increasing this if drawing lags
            brush_steps = int(round(max(drag_len / brush_step_size, 1)))
            coords_diff = (end[0] - start[0], end[1] - start[1])
            for step in range(brush_steps):
                alpha = step / brush_steps
                brush_coords = (int(round(start[0] + coords_diff[0] * alpha)),
                                int(round(start[1] + coords_diff[1] * alpha)))
                cv.circle(mask, brush_coords,
                          int(round((mask.shape[1] / cls.WINDOW_SIZE[0]))) * cls.BRUSH_SIZE,
                          label_id, thickness=-1, lineType=-1)

    class ZoomTooltip:
        """Zoom tooltip manager used to visualize image details based on mouse location."""

        def __init__(self, config):
            """Parses the input config and extracts zoom-related parameters."""
            self.window_zoom_crop_size = thelper.utils.str2size(thelper.utils.get_key_def("window_zoom_crop_size", config, "250x250"))
            self.window_zoom_size = thelper.utils.str2size(thelper.utils.get_key_def("window_zoom_size", config, "500x500"))
            self.window_zoom_name = "zoom"
            self.image_crop = np.empty((self.window_zoom_crop_size[1], self.window_zoom_crop_size[0], 3), dtype=np.uint8)
            self.image_zoom = np.empty((self.window_zoom_size[1], self.window_zoom_size[0], 3), dtype=np.uint8)
            self.mask_crop = np.empty((self.window_zoom_crop_size[1], self.window_zoom_crop_size[0]), dtype=np.uint8)
            self.mask_crop_color = np.empty((self.window_zoom_crop_size[1], self.window_zoom_crop_size[0], 3), dtype=np.uint8)
            self.mask_zoom = np.empty((self.window_zoom_size[1], self.window_zoom_size[0], 3), dtype=np.uint8)
            self.display_zoom = np.empty((self.window_zoom_size[1], self.window_zoom_size[0], 3), dtype=np.uint8)
            self.zoom_interp_type = thelper.utils.import_class(thelper.utils.get_key_def("zoom_interp_type", config, "cv2.INTER_NEAREST"))
            cv.namedWindow(self.window_zoom_name, cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)

        def refresh(self, image, mask, mask_colormap, mask_opacity, coords):
            """Fetches the latest mouse position and updates the zoom window tooltip."""
            if coords is not None:
                if self.window_zoom_crop_size[0] > image.shape[1] or self.window_zoom_crop_size[1] > image.shape[0]:
                    logger.warning("crop size too large for input image, skipping zoom refresh")
                    return
                tl = (max(coords[0] - self.window_zoom_crop_size[0] // 2, 0),
                      max(coords[1] - self.window_zoom_crop_size[1] // 2, 0))
                br = (min(tl[0] + self.window_zoom_crop_size[0], image.shape[1]),
                      min(tl[1] + self.window_zoom_crop_size[1], image.shape[0]))
                tl = (max(br[0] - self.window_zoom_crop_size[0], 0),
                      max(br[1] - self.window_zoom_crop_size[1], 0))
                if (br[0] - tl[0], br[1] - tl[1]) != self.window_zoom_crop_size:
                    logger.warning("bad zoom crop size, should never happen...?")
                else:
                    np.copyto(self.image_crop, image[tl[1]:br[1], tl[0]:br[0], ...])
                    cv.resize(self.image_crop, dsize=self.window_zoom_size, dst=self.image_zoom, interpolation=self.zoom_interp_type)
                    np.copyto(self.mask_crop, mask[tl[1]:br[1], tl[0]:br[0], ...])
                    thelper.utils.apply_color_map(self.mask_crop, mask_colormap, dst=self.mask_crop_color)
                    cv.resize(self.mask_crop_color, dsize=self.window_zoom_size, dst=self.mask_zoom, interpolation=self.zoom_interp_type)
                    cv.addWeighted(self.image_zoom, (1 - mask_opacity), self.mask_zoom, mask_opacity, 0.0, dst=self.display_zoom)
                    cv.imshow(self.window_zoom_name, self.display_zoom)

    def __init__(self, session_name, config, save_dir, datasets):
        """Parses the input samples and initializes the anntator GUI elements."""
        super().__init__(session_name, config, save_dir, datasets)
        config = self.annotator_config  # cheat for conciseness
        self.sample_input_key = thelper.utils.get_key("sample_input_key", config)
        self.logger.info("parsing datasets...")
        self.sample_count = 0
        self.sample_idx_offsets = {}
        for dataset_name, dataset in self.datasets.items():
            self.sample_idx_offsets[dataset_name] = self.sample_count
            self.sample_count += len(dataset)
            annot_dir = os.path.join(self.annotations_dir, dataset_name)
            if not os.path.isdir(annot_dir):
                os.mkdir(annot_dir)
            logstamp = thelper.utils.get_log_stamp()
            repover = thelper.__version__ + ":" + thelper.utils.get_git_stamp()
            log_content = {
                "session_name": session_name,
                "logstamp": logstamp,
                "version": repover,
                "dataset": str(dataset),
            }
            if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
                log_content["samples"] = [str(sample) for sample in dataset.samples]
            dataset_log_file = os.path.join(annot_dir, "metadata.log")
            with open(dataset_log_file, "w") as fd:
                json.dump(log_content, fd, indent=4, sort_keys=False)
        self.logger.info("datasets possess a total of %d samples" % self.sample_count)
        self.curr_sample_idx = int(thelper.utils.get_key_def("start_sample_idx", config, 0))
        if self.curr_sample_idx >= self.sample_count:
            raise AssertionError("file index out-of-range (curr=%d, max=%d)" % (self.curr_sample_idx, self.sample_count - 1))
        self.window_name = thelper.utils.get_key_def("window_name", config, "image-segm-annotator")
        self.window_size = thelper.utils.get_key_def("window_size", config, 1000)
        self.brush_thickness = thelper.utils.get_key_def("brush_thickness", config, 2)
        ImageSegmentAnnotator.GUI_BAR_SIZE = thelper.utils.get_key_def("gui_bar_size", config, 50)
        self.mask_opacity = float(thelper.utils.get_key_def("default_mask_opacity", config, 0.3))
        if not (0 <= self.mask_opacity <= 1):
            raise AssertionError("unexpected opacity setting, should be in [0,1]")
        self.background_id = int(thelper.utils.get_key_def("background_id", config, 0))
        if not (0 <= self.background_id <= 255):
            raise AssertionError("background id '%s' out of 8-bit int range" % self.background_id)
        self.labels = thelper.utils.get_key("labels", config)
        self.curr_label_idx = 0
        self.label_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        for label_idx, label in enumerate(self.labels):
            if not isinstance(label, dict):
                raise AssertionError("invalid label type")
            if "id" not in label or "name" not in label or "color" not in label:
                raise AssertionError("missing some fields in label dict")
            label_id = label["id"]
            if not isinstance(label_id, int):
                raise AssertionError("unexpected label type (must be int)")
            if label_id == self.background_id or sum([label_id == lbl["id"] for lbl in self.labels]) != 1:
                raise AssertionError("duplicate label id found (%s)" % label_id)
            if not (0 <= label_id <= 255):
                raise AssertionError("label id '%s' out of 8-bit int range" % label_id)
            color = label["color"]
            if not isinstance(color, list) or len(color) != 3 or not all([isinstance(c, int) for c in color]):
                raise AssertionError("invalid label color, 3-elem integer list expected")
            self.label_colormap[label_id, 0, :] = color
        self.default_fill_id = int(thelper.utils.get_key_def("default_fill_id", config, 0))
        if self.default_fill_id not in self.labels and self.default_fill_id != self.background_id:
            raise AssertionError("unknown fill id '%s'" % self.default_fill_id)
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
        self.brush = ImageSegmentAnnotator.Brush(config)
        self.zoom = ImageSegmentAnnotator.ZoomTooltip(config)
        self.sample = None
        self.image, self.mask = self.load(self.curr_sample_idx)
        self.image_display, self.image_display_base, self.mask_display = self.refresh_layers()
        self.gui_display = None
        self.listener = thelper.gui.create_key_listener(ImageSegmentAnnotator.on_press)
        self.listener.start()
        cv.setMouseCallback(self.window_name, ImageSegmentAnnotator.on_mouse)

    def refresh_layers(self):
        """Updates the image, mask, and tool display layers based on the latest changes."""
        cls = ImageSegmentAnnotator
        if self.image_display_base is None:
            self.image_display_base = cv.resize(self.image, dsize=cls.WINDOW_SIZE)
            cls.MASK_DIRTY = True
        if self.mask_display is None or cls.MASK_DIRTY:
            self.mask_display = cv.resize(self.mask, dsize=cls.WINDOW_SIZE, interpolation=cv.INTER_NEAREST)
            self.mask_display = thelper.utils.apply_color_map(self.mask_display, self.label_colormap)
            cls.MASK_DIRTY = True
        if self.image_display is None or cls.MASK_DIRTY:
            self.image_display = cv.addWeighted(self.image_display_base, (1 - self.mask_opacity), self.mask_display, self.mask_opacity, 0.0)
            cls.GUI_DIRTY = True
        cls.MASK_DIRTY = False
        return self.image_display, self.image_display_base, self.mask_display

    def refresh_gui(self):
        """Updates and displays the main window based on the latest changes."""
        cls = ImageSegmentAnnotator
        if self.gui_display is None or cls.GUI_DIRTY:
            gui_shape = (self.image_display.shape[0] + cls.GUI_BAR_SIZE, self.image_display.shape[1], self.image_display.shape[2])
            if self.gui_display is None or self.gui_display.shape != gui_shape:
                self.gui_display = np.empty(gui_shape, dtype=np.uint8)
            np.copyto(self.gui_display[cls.GUI_BAR_SIZE:, ...], self.image_display)
            if cls.LATEST_PT[0] >= 0 and cls.LATEST_PT[1] >= 0:
                coords_abs = (int(cls.LATEST_PT[0] * self.image.shape[1]), int(cls.LATEST_PT[1] * self.image.shape[0]))
                self.zoom.refresh(self.image, self.mask, self.label_colormap, self.mask_opacity, coords_abs)
            curr_label = self.labels[self.curr_label_idx]
            cv.circle(self.gui_display, cls.LATEST_RAW_PT, cls.BRUSH_SIZE, curr_label["color"], self.brush_thickness)
            cv.rectangle(self.gui_display, (0, 0), (self.gui_display.shape[1], cls.GUI_BAR_SIZE), (0, 0, 0), -1)
            gui_str = "sample #%d" % self.curr_sample_idx
            # todo: update to use meta keys?
            if "path" in self.sample and isinstance(self.sample["path"], str):
                gui_str += "   path: %s" % self.sample["path"].replace('\\', '/')
            cv.putText(self.gui_display, gui_str, (10, int(cls.GUI_BAR_SIZE * 2 / 5)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv.LINE_AA)
            gui_str = "current brush: %s" % curr_label["name"]
            cv.putText(self.gui_display, gui_str, (10, int(cls.GUI_BAR_SIZE * 3 / 4)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.40, curr_label["color"], 1, cv.LINE_AA)
            cls.GUI_DIRTY = False
        cv.imshow(self.window_name, self.gui_display)
        cv.waitKey(1)

    def handle_keys(self):
        """Fetches the latest keyboard press and updates the annotator state accordingly."""
        import pynput.keyboard
        cls = ImageSegmentAnnotator
        nb_labels = len(self.labels)
        if cls.CURRENT_KEY != -1:
            if cls.CURRENT_KEY == pynput.keyboard.Key.down or cls.CURRENT_KEY == pynput.keyboard.KeyCode(char='s'):
                cv.imwrite(self.get_mask_path(self.curr_sample_idx), self.mask)
                self.curr_sample_idx = min(self.curr_sample_idx + 1, self.sample_count - 1)
                logger.debug("loading image+mask #%d... (max=%d)" % (self.curr_sample_idx, self.sample_count - 1))
                if self.curr_sample_idx == self.sample_count - 1:
                    logger.debug("(reached last)")
                self.image, self.mask = self.load(self.curr_sample_idx)
            elif cls.CURRENT_KEY == pynput.keyboard.Key.up or cls.CURRENT_KEY == pynput.keyboard.KeyCode(char='w'):
                cv.imwrite(self.get_mask_path(self.curr_sample_idx), self.mask)
                self.curr_sample_idx = max(self.curr_sample_idx - 1, 0)
                logger.debug("loading image+mask #%d... (max=%d)" % (self.curr_sample_idx, self.sample_count - 1))
                self.image, self.mask = self.load(self.curr_sample_idx)
            elif cls.CURRENT_KEY == pynput.keyboard.Key.right:
                self.curr_label_idx = min(self.curr_label_idx + 1, nb_labels - 1)
                curr_label = self.labels[self.curr_label_idx]
                logger.debug("swapping to label #%s : '%s'" % (curr_label["id"], curr_label["name"]))
                cls.GUI_DIRTY = True
            elif cls.CURRENT_KEY == pynput.keyboard.Key.left:
                self.curr_label_idx = max(self.curr_label_idx - 1, 0)
                curr_label = self.labels[self.curr_label_idx]
                logger.debug("swapping to label #%s : '%s'" % (curr_label["id"], curr_label["name"]))
                cls.GUI_DIRTY = True
            elif cls.CURRENT_KEY == pynput.keyboard.Key.page_up:
                self.mask_opacity = max(self.mask_opacity - 0.1, 0.0)
                logger.debug("decreasing display mask opacity to %0.1f..." % self.mask_opacity)
                self.image_display = None
            elif cls.CURRENT_KEY == pynput.keyboard.Key.page_down:
                self.mask_opacity = min(self.mask_opacity + 0.1, 1.0)
                logger.debug("increasing display mask opacity to %0.1f..." % self.mask_opacity)
                self.image_display = None
            elif cls.CURRENT_KEY == pynput.keyboard.Key.esc:
                cv.imwrite(self.get_mask_path(self.curr_sample_idx), self.mask)
                logger.debug("breaking off!")
                return True
            elif cls.CURRENT_KEY == pynput.keyboard.Key.f12:
                self.mask = np.full((self.image.shape[0], self.image.shape[1]), self.default_fill_id, dtype=np.uint8)
                self.mask_display = None
            elif cls.CURRENT_KEY == pynput.keyboard.Key.enter:
                cv.imwrite(self.get_mask_path(self.curr_sample_idx), self.mask)
            else:
                for label_idx in range(min(9, nb_labels)):
                    if cls.CURRENT_KEY == pynput.keyboard.KeyCode(char=str(label_idx + 1)):
                        self.curr_label_idx = label_idx
                        curr_label = self.labels[self.curr_label_idx]
                        logger.debug("swapping to label #%s : '%s'" % (curr_label["id"], curr_label["name"]))
                        break
            cls.CURRENT_KEY = -1
        return False

    def get_mask_path(self, index):
        """Returns the path where the mask of a specific sample should be located."""
        dataset_name = None
        for name, offset in reversed(list(self.sample_idx_offsets.items())):
            if 0 <= index - offset < len(self.datasets[name]):
                index -= offset
                dataset_name = name
                break
        if dataset_name is None:
            raise AssertionError("bad logic somewhere")
        return os.path.join(self.annotations_dir, dataset_name, "%06d.png" % index)

    def load(self, index):
        """Loads the image and mask associated to a specific sample."""
        self.image = None
        for name, offset in reversed(list(self.sample_idx_offsets.items())):
            if 0 <= index - offset < len(self.datasets[name]):
                self.sample = self.datasets[name][index - offset]
                if self.sample_input_key not in self.sample:
                    raise AssertionError("could not locate value #%d for input key '%s'" % (index, self.sample_input_key))
                self.image = self.sample[self.sample_input_key]
                break
        if self.image is None or not isinstance(self.image, np.ndarray):
            raise AssertionError("invalid input image for index #%d" % index)
        self.image_display_base, self.image_display = None, None
        mask_path = self.get_mask_path(index)
        if os.path.isfile(mask_path):
            self.mask = cv.imread(mask_path, flags=cv.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise AssertionError("could not open mask at '%s'" % mask_path)
        else:
            self.mask = np.full((self.image.shape[0], self.image.shape[1]), self.default_fill_id, dtype=np.uint8)
        self.mask_display = None
        if self.image.shape[0:2] != self.mask.shape:
            raise AssertionError("mismatched image/mask shapes")
        cls = ImageSegmentAnnotator
        if isinstance(self.window_size, str):
            cls.WINDOW_SIZE = thelper.utils.str2size(self.window_size)
        elif isinstance(self.window_size, int):
            max_scale_factor = self.window_size / max(self.image.shape[0], self.image.shape[1])
            cls.WINDOW_SIZE = (int(round(max_scale_factor * self.image.shape[1])), int(round(max_scale_factor * self.image.shape[0])))
        else:
            raise AssertionError("unexpected window size type")
        return self.image, self.mask

    def run(self):
        """Displays the main window and other GUI elements in a loop until it is closed by the user."""
        while cv.getWindowProperty(self.window_name, 0) != -1:
            if self.handle_keys():
                break
            self.brush.refresh(self.mask, self.labels[self.curr_label_idx])
            self.refresh_layers()
            self.refresh_gui()
        cv.destroyAllWindows()
