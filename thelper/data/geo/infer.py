import json
import logging
import os
from typing import TYPE_CHECKING

import gdal
import numpy as np
import torch

import thelper.concepts
import thelper.data.geo
from thelper.infer.base import Tester

if TYPE_CHECKING:
    from typing import AnyStr, Optional, Tuple  # noqa: F401

logger = logging.getLogger(__name__)


@thelper.concepts.classification
class SlidingWindowTester(Tester):
    """Tester that satisfies the requirements of the :class:`Tester` in order to run classification inference

    """

    def __init__(self,
                 session_name,    # type: AnyStr
                 session_dir,     # type: AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                 ):
        super(SlidingWindowTester, self).__init__(session_name, session_dir, model,
                                                  task, loaders, config, ckptdata=ckptdata)
        # because 'tester' gets called explicitly during inference, check for tester/runner before trainer key
        # this way we can favor using a detailed config which specified both trainer/tester simultaneously and
        # use the correct one with all corresponding CLI modes
        runner_config = thelper.utils.get_key(["runner", "tester", "trainer"], config)
        self.normalize_loss = thelper.utils.get_key_def("normalize_loss", runner_config, True)

    def eval_epoch(self, model, epoch, dev, loader, metrics, output_path):
        """Computes the pixelwise prediction on an image.

        It does the prediction per batch size of N pixels. It returns the class predicted and its probability.
        The results are saved into two images created with the same size and projection info as the input rasters.

        The ``class`` image gives the class id, a number between 1 and the number of classes for corresponding pixels.
        Class id 0 is reserved for ``nodata``.

        The ``probs`` image contains N-class channels with the probability
        values of the pixels for each class. The probabilities by default are normalised.

        Also, a ``config-classes.json`` file is created listing the ``name-to-class-id`` mapping that was used to
        generate the values in the ``class`` image (i.e.: class names defined by the pre-trained ``model``).

        Args:
            model: the model with which to run inference that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based, and should normally only be 0 for single test pass).
            dev: the target device that tensors should be uploaded to (corresponding to model's device(s)).
            loader: the data loader used to get transformed test samples.
            metrics: the dictionary of metrics/consumers to report inference results (mostly loggers and basic report
                generator in this case since there shouldn't be ground truth labels to validate against).
            output_path: directory where output files should be written, if necessary.
        """

        ds_type = thelper.data.geo.parsers.SlidingWindowDataset
        if not isinstance(loader.dataset, ds_type):
            raise AssertionError(f"Only dataset '{ds_type.__module__}.{ds_type.__name__}' is supported by "
                                 f"test session runner {SlidingWindowTester.__module__}.{SlidingWindowTester.__name__}")
        output_path = os.path.abspath(output_path)
        class_count = len(model.task.class_names)
        class_ds, probs_ds = self._prepare_output_rasters(loader.dataset, output_path, class_count)

        class_indices = model.task.class_indices
        for key in class_indices.keys():
            class_indices[key] += 1
        class_indices['no_data'] = 0

        class_indices_file = "config-classes.json"
        class_indices_file_path = os.path.join(output_path, class_indices_file)
        logger.debug("Writing class indices: [%s]", class_indices_file_path)
        with open(class_indices_file_path, 'w') as f:
            json.dump(class_indices, f, indent=4)

        normalize = torch.nn.Softmax(dim=1) if self.normalize_loss else lambda _: _  # Normalizing/pass-through
        model.eval()
        with torch.no_grad():
            n_batches = len(loader)
            n_patches = loader.batch_size
            logger.debug("Starting inference of %s batches each composed of %s patch samples", n_batches, n_patches)
            for k, sample in enumerate(loader):
                logger.info(f"Batch {k+1} of {n_batches}: {(k+1)/n_batches:4.1%}")
                center_x0 = self._move_tensor(sample[loader.dataset.center_key][0], dev="cpu", detach=True).data.numpy()
                center_y0 = self._move_tensor(sample[loader.dataset.center_key][1], dev="cpu", detach=True).data.numpy()
                n_data = center_x0.shape[0]  # batch-size
                x_data = sample[loader.dataset.image_key]
                x_data = self._move_tensor(x_data, dev=dev)
                y_prob = model(x_data)
                y_prob = normalize(y_prob)
                y_class_indices = torch.argmax(y_prob, dim=1)
                y_class_indices = self._move_tensor(y_class_indices, dev="cpu", detach=True).data.numpy()
                y_prob = self._move_tensor(y_prob, dev="cpu", detach=True).data.numpy()
                # loop each patch from the batch
                for j in range(n_data):
                    class_id = np.array([[y_class_indices[j] + 1]])
                    x0 = int(center_x0[j])
                    y0 = int(center_y0[j])
                    class_ds.GetRasterBand(1).WriteArray(class_id, x0, y0)
                    for p in range(y_prob.shape[1]):
                        probs_ds.GetRasterBand(p + 1).WriteArray(np.array([[y_prob[j, p]]], dtype='float32'),
                                                                 int(center_x0[j]), int(center_y0[j]))
                # save writen changes to disk
                class_ds.FlushCache()
                probs_ds.FlushCache()
        logger.debug("Closing output rasters")
        class_ds = None  # noqa # close file
        probs_ds = None  # noqa # close file

    @staticmethod
    def _prepare_output_rasters(raster_loader, output_path, class_count):
        # type: (thelper.data.geo.parsers.SlidingWindowDataset, AnyStr, int) -> Tuple[gdal.Dataset, gdal.Dataset]
        """
        Generates the ``class`` and ``probs`` datasets to be filed by inference results.
        """
        logger.info("Preparing output rasters")
        logger.debug("using output name: [%s]", raster_loader.raster["name"])

        xsize = raster_loader.raster["xsize"]
        ysize = raster_loader.raster["ysize"]
        georef = raster_loader.raster["georef"]
        affine = raster_loader.raster["affine"]
        raster_name = raster_loader.raster["name"]
        raster_class_name = f"{raster_name}_class.tif"
        raster_class_path = os.path.join(output_path, raster_class_name)
        # Create the class raster output
        class_ds = gdal.GetDriverByName('GTiff').Create(raster_class_path, xsize, ysize, 1, gdal.GDT_Byte)
        if class_ds is None:
            raise IOError(f"Unable to create: [{raster_class_path}]")
        else:
            logger.debug(f"Creating: [{raster_class_path}]")
        class_ds.SetGeoTransform(affine)
        class_ds.SetProjection(georef)
        class_band = class_ds.GetRasterBand(1)
        class_band.SetNoDataValue(0)
        class_ds.FlushCache()   # save to disk
        class_ds = None     # noqa # need to close before open-update
        class_band = None   # noqa # also close band (remove ptr)
        class_ds = gdal.Open(raster_class_path, gdal.GA_Update)
        # Create the probabilities raster output
        raster_prob_name = f"{raster_name}_probs.tif"
        raster_prob_path = os.path.join(output_path, raster_prob_name)
        probs_ds = gdal.GetDriverByName('GTiff').Create(raster_prob_path, xsize, ysize, class_count, gdal.GDT_Float32)
        if probs_ds is None:
            raise IOError(f"Unable to create: [{raster_prob_path}]")
        else:
            logger.debug(f"Creating: [{raster_prob_path}]")
        probs_ds.SetGeoTransform(affine)
        probs_ds.SetProjection(georef)
        probs_ds.FlushCache()   # save to disk
        probs_ds = None  # noqa # need to close before open-update
        probs_ds = gdal.Open(raster_prob_path, gdal.GA_Update)
        return class_ds, probs_ds
