import io
import math
import warnings
import itertools

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from wotan import flatten
from .common_utils import warning_on_one_line

warnings.formatwarning = warning_on_one_line  # Raise my own warns


def detrend_light_curve(lc_object, window_length=0.5, edge_cutoff=0.5, break_tolerance=0.5, cval=5.0, sigma_upper=3,
                        sigma_lower=20):
    """
    Detrend a light curve for upcoming transit searching with wotan biweight method and sigma clipping
    https://github.com/hippke/wotan

    Parameters
    ----------
    lc_object: `~lightkurve.LightCurve` instance
                Input light curve object
    window_length: float
                The length of the filter window in units of ``time``, default is 0.5
    edge_cutoff: float
                Length (in units of time) to be cut off each edge, default is 0.5
    break_tolerance: float
                Split into segments at breaks longer than that, default is 0.5
    cval: float
                Tuning parameter for the robust estimators, default is 5.0
    sigma_upper: float
                Upper limit of standard deviations for sigma clipping
    sigma_lower: float
                Lower limit of standard deviations for sigma clipping
    Returns
    -------
    flatten_lc : `~lightkurve.LightCurve` instance
    """
    _, trend_flux = flatten(
        lc_object.time.value,
        lc_object.flux.value,
        method='biweight',
        window_length=window_length,
        edge_cutoff=edge_cutoff,
        break_tolerance=break_tolerance,
        return_trend=True,  # Return trend and flattened light curve
        cval=cval
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flatten_flux = sigma_clip(lc_object.flux.value / trend_flux, sigma_upper=sigma_upper, sigma_lower=sigma_lower,
                                  cenfunc=np.nanmedian, stdfunc=np.nanstd, masked=False, axis=0)

    return lk.LightCurve(time=lc_object.time.value, flux=flatten_flux, meta=lc_object.meta)


def smooth_light_curve(lc_object, N_points):
    """
    Smooth light curve with moving average
    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
    N_points : int
                Window size of moving average

    Returns
    -------
    smoothed_lc : `~lightkurve.LightCurve` instance
    """
    df = lc_object.to_pandas()
    t = df.index.values
    y = df.flux.rolling(N_points, min_periods=N_points // 2).mean().values
    return lk.LightCurve(time=t, flux=y)


def _light_curve_to_image_array(lc_object, flux_range):
    """
    Convert light curve slice to image array
    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
    flux_range : tuple
                Flux range in 30-day window in format: (flux_min, flux_max)
    Returns
    -------
    img_arr : np.ndarray
                Numpy image array
    """
    exp_time = np.nanmin(np.diff(lc_object.time.value))

    with plt.rc_context({'backend': 'agg'}):
        io_buf = io.BytesIO()
        io_buf.seek(0)
        plt.ioff()
        fig, ax = plt.subplots(1, figsize=(4.16, 4.16), dpi=100, frameon=False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_facecolor('white')
        ax.axis('off')
        ax.margins(0, 0)
        if exp_time > 0.00417:
            ax.plot(lc_object.time.value, lc_object.flux.value, ls='-', marker='o', lw=72 / fig.dpi / 2,
                    ms=72 / fig.dpi * 2,
                    color='black')
        else:
            ax.plot(lc_object.time.value, lc_object.flux.value, '.', ms=72 / fig.dpi, color='black', mew=1.0)
            smoothed_lc = smooth_light_curve(lc_object.remove_nans(),
                                             int(0.0204 / np.nanmin(np.diff(lc_object.time.value))))
            ax.plot(smoothed_lc.time.value, smoothed_lc.flux.value, 'grey', lw=1)

        ax.set_ylim([flux_range[0], flux_range[1]])
        fig.savefig(io_buf, format='raw', dpi=100)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4))
        io_buf.close()
        plt.close()
    return img_arr


def _bounding_box_to_time_flux(lc_object, bboxes, flux_range):
    """
    Convert image pixel value to time and flux

    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
    bboxes : list
                List of bounding boxes.
    flux_range : tuple
                Flux range in 30-day window in format: (flux_min, flux_max)

    Returns
    -------
    transit_masks : list
                List of bounding box position, in format: [[confidence, x_time, y_flux, w_time, h_flux], ...]
    """
    lc_object = lc_object.remove_nans()
    time, flux = lc_object.time.value, lc_object.flux.value
    t_min, t_max, f_min, f_max = np.min(time), np.max(time), flux_range[0], flux_range[1]
    transit_masks = []
    for bbox in bboxes:
        confidence, x, y, w, h = bbox[:]
        x_time = x * (t_max - t_min) + t_min  # x_time is the middle time of a box
        y_flux = f_max - y * (f_max - f_min)  # y_flux is the middle flux of a box
        w_time = w * (t_max - t_min)
        h_flux = h * (f_max - f_min)
        transit_masks.append([confidence, x_time, y_flux, w_time, h_flux])
    return transit_masks


def _split_light_curve(lc_object, split_time=10, back_step=3):
    """
    Split a light curve to some slices with indicated time interval and step

    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
                Input light curve object
    split_time : float
                The length of each light curve slice in units of ``time``
    back_step : float
                The backward step of each light curve slice in units of ``time``

    Yields
    -------
    time_start : float
                The begin time of each light curve slice.
    time_stop : float
                The end time of each light curve slice.
    """
    if (lc_object.time.value[-1] - lc_object.time.value[0]) < split_time:
        yield lc_object.time.value[0], lc_object.time.value[-1]
    else:
        time_start = lc_object.time.value[0]
        while time_start < lc_object.time.value[-1]:
            time_stop = time_start + split_time
            if time_stop > lc_object.time.value[-1]:
                yield lc_object.time.value[-1] - split_time, lc_object.time.value[-1]
                break
            else:
                yield time_start, time_stop
                time_start = time_stop - back_step


class DeepTransit:
    """
    The core class of transit detection.
    """
    def __init__(self, lc_object=None, time=None, flux=None, flux_err=None, is_flat=False,
                 lk_kwargs={}, flatten_kwargs={}):
        """
        Initial function for receiving an light curve object or a time series.

        Parameters
        ----------
        lc_object : `~lightkurve.LightCurve` instance
                    Input light curve object
        time : `~astropy.time.Time` or iterable
                    Time values.  They can either be given directly as a `~astropy.time.Time` array
                    or as any iterable that initializes the `~astropy.time.Time` class.
        flux : `~astropy.units.Quantity` or iterable
                    Flux values for every time point.
        flux_err : `~astropy.units.Quantity` or iterable
                    Uncertainty on each flux data point.
        is_flat : bool
                    True when receiving a flattened light, False will use the built-in flatten method.
        lk_kwargs : dict
                    Keyword arguments of `~lightkurve.LightCurve`.
        flatten_kwargs : dict
                    Keyword arguments of `detrend_light_curve` method.
        """
        if lc_object is None:
            if time is None and flux is not None:
                time = np.arange(len(flux))
                lc_object = lk.LightCurve(time=time, flux=flux, flux_err=flux_err, **lk_kwargs).remove_nans()
            # We are tolerant of missing time format
            if time is not None and flux is not None:
                lc_object = lk.LightCurve(time=time, flux=flux, flux_err=flux_err, **lk_kwargs).remove_nans()
        elif lc_object:
            if isinstance(lc_object, lk.LightCurve):
                lc_object = lc_object.remove_nans()
            else:
                raise TypeError(f"'lc_obj' should be a `lightkurve.LightCurve object`")

        lc_object.sort('time')

        if is_flat is True:
            self.lc = lc_object
        else:
            self.lc = detrend_light_curve(lc_object, **flatten_kwargs)



    def _splited_lc_generator(self, backend):
        time_initial_index = 0
        for time_block_index in np.append((np.diff(self.lc.time.value) > 10).nonzero()[0], len(self.lc) - 1):
            lc_block = self.lc[time_initial_index:time_block_index]
            time_initial_index = time_block_index + 1
            for start_time, stop_time in list(_split_light_curve(lc_block.remove_nans(), split_time=30, back_step=5)):
                mask = (lc_block.time.value >= start_time) & (lc_block.time.value <= stop_time)
                selected_lc = lc_block[mask]
                exp_time = np.nanmin(np.diff(selected_lc.time.value))
                if len(selected_lc) < 5 / exp_time:
                    continue
                flux_min, flux_max = np.nanmin(selected_lc.flux.value) * 1.02 - 0.02 * np.nanmax(
                    selected_lc.flux.value), np.nanmax(selected_lc.flux.value)
                for t0, t1 in _split_light_curve(selected_lc.remove_nans(), split_time=10, back_step=3):
                    mask = (selected_lc.time.value >= t0) & (selected_lc.time.value <= t1)
                    splited_flatten_lc = selected_lc[mask]
                    exp_time = np.nanmin(np.diff(splited_flatten_lc.time.value))
                    if len(splited_flatten_lc) < 1 / exp_time:
                        continue
                    img_arr = _light_curve_to_image_array(splited_flatten_lc, (flux_min, flux_max))
                    image = np.array(Image.fromarray(img_arr).convert("L"))
                    image = backend.trans(image)

                    yield splited_flatten_lc, image, flux_min, flux_max

    def _data_loader(self, batch_size=1, backend=None):
        it = iter(self._splited_lc_generator(backend))
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunk = np.array(list(itertools.islice(it, batch_size)))
            if chunk.size == 0:
                return
            yield chunk


    def transit_detection(self, local_model_path, batch_size=2, confidence_threshold=0.6, nms_iou_threshold=0.1, device_str='auto', backend='pytorch'):
        """
        Searching transit signals from a given light curve.

        Parameters
        ----------
        model_path : str
                    The path of the model file.
        batch_size : int
                    Batch size for increasing detection speed, especially useful for GPU
                    default value is 2, if using GPU, it can be higher depending on the limitation of the GPU memory.
        confidence_threshold : float
                    Confidence threshold for transit detection. If None, the value will be obtained from config.
                    Default value is 0.6.
        nms_iou_threshold : float
                    IOU threshold for NMS algorithm. If None, the value will be obtained from config.
                    Default value is 0.1.
        device_str : str
                    Device name. If "cuda", it will use GPU. Default is "auto".
        backend : str
                    Backend of the model. You can choose between "pytorch" or "megengine".
                    Default is "pytorch".
        Returns
        -------
        final_bboxes : np.ndarray
                    An (N, 5) shape numpy.ndarray of bounding boxes.
        """

        if backend == 'pytorch':
            from .backend import PytorchBackend
            backend = PytorchBackend(device_str)
        else:
            assert backend == 'megengine'
            from.mge.backend import MegengineBackend
            backend = MegengineBackend()
        backend.load_model(local_model_path)

        real_unit_bboxes = []
        exp_time = np.nanmedian(np.diff(self.lc.time.value))
        rough_length = math.ceil(math.ceil((len(self.lc) * exp_time - 30) / 25 + 1) * 4 / batch_size)
        warnings.warn('The total number of progress bar is the upper limit.')
        for data in tqdm(self._data_loader(batch_size=batch_size, backend=backend), total=rough_length):
            lc_data = data[:, 0]
            flux_min, flux_max = data[:, 2], data[:, 3]
            predicted_bboxes = backend.inference(
                data[:, 1], nms_iou_threshold=nms_iou_threshold, confidence_threshold=confidence_threshold)
            for index, bboxes in enumerate(predicted_bboxes):
                predicted_bboxes_in_real_unit = _bounding_box_to_time_flux(lc_data[index], bboxes,
                                                                           (flux_min[index], flux_max[index]))
                real_unit_bboxes += predicted_bboxes_in_real_unit

        final_bboxes = np.array(backend.nms(real_unit_bboxes, nms_iou_threshold=nms_iou_threshold,  confidence_threshold=confidence_threshold))

        return final_bboxes


def plot_lc_with_bboxes(lc_object, bboxes, ax=None, **kwargs):
    """
    Plot light curve with bounding boxes

    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance

    bboxes : list or np.ndarray
                Bounding boxes in shape (N, 5)

    ax : `~matplotlib.pyplot.axis` instance
                Axis to plot to. If None, create a new one.
    kwargs : dict
                Additional arguments to be passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax : `~matplotlib.pyplot.axis` instance
                The matplotlib axes object.
    """
    with plt.style.context('grayscale'):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 7), constrained_layout=False)
            ax.plot(lc_object.time.value, lc_object.flux.value, **kwargs)
        else:
            ax.plot(lc_object.time.value, lc_object.flux.value, **kwargs)
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        recs = []
        for real_mask in bboxes:
            rec = Rectangle((real_mask[1] - real_mask[3] / 2, real_mask[2] - real_mask[4] / 2),
                            real_mask[3],
                            real_mask[4], fill=False, color='lime')
            recs.append(rec)

            ax.text(
                real_mask[1] - real_mask[3] / 2,
                real_mask[2] + real_mask[4] / 2,
                s=f"{real_mask[0]:.2f}",
                color="white",
                verticalalignment="top",
                bbox=dict(alpha=0.5, color='blue'),
                clip_on=True
            )

        pc = PatchCollection(recs, facecolor='none', edgecolor='lime', lw=1, zorder=3)
        ax.add_collection(pc)
        ax.set_xlabel('Time (day)')
        ax.set_ylabel('Normalized Flux')
    return ax


def select_lc_from_bboxes(lc_object, bboxes, fill=1):
    """

    Parameters
    ----------
    bboxes : list
    lc_object : `~lightkurve.LightCurve` instance
    fill : float
        If None, return light curve in the bounding boxes.
        Otherwise the outer region will be filled with a given value.
        Default value is 1.
    Returns
    -------
    """
    range_logic = False
    for bbox in bboxes:
        t0 = bbox[1] - bbox[3] / 2
        t1 = bbox[1] + bbox[3] / 2
        y0 = bbox[2] - bbox[4] / 2
        y1 = bbox[2] + bbox[4] / 2
        range_logic = range_logic | (lc_object.time.value >= t0) & (lc_object.time.value <= t1) & (
                lc_object.flux.value >= y0) & (lc_object.flux.value <= y1)

    if fill is not None:
        notransiting_lc = lc_object[~range_logic]
        notransiting_lc['flux'] = fill
        filled_lc = lc_object[range_logic].append(notransiting_lc)
        filled_lc.sort('time')
        return filled_lc
    else:
        return lc_object[range_logic]


def main():
    parser = argparse.ArgumentParser(description='demo for lc detection')
    parser.add_argument('-lc', type=str, default='11446443', help='light curve number of KIC, used as src')
    parser.add_argument('-m', '--model_path', type=str, help='model path, will download if empty' )
    parser.add_argument('-b', '--batch', type=int, help='batchsize used to inference', default=3)
    parser.add_argument('--backend', type=str, help='backend of model, use pytorch/megengine', default='pytorch')
    parser.add_argument('-d', '--device', type=str, help='runtime device of backend', default=None)
    parser.add_argument('--nms_iou_threshold', type=float, help='nms iou  threshold', default=None)
    parser.add_argument('--confidence_threshold', type=float, help='confidence threshold', default=None)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    search_result = lk.search_lightcurve('KIC {}'.format(args.lc), author='Kepler')
    lc = search_result.download_all().stitch()
    lc = lc[lc.time.value < 135]
    dt = DeepTransit(lc, is_flatten=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper':3})
    flat_lc = detrend_light_curve(lc, window_length=0.5)
    bboxes = dt.transit_detection(args.model_path, batch_size=args.batch,
                                  nms_iou_threshold=args.nms_iou_threshold, confidence_threshold=args.confidence_threshold, device_str=args.device, backend=args.backend)
    fig, ax = plt.subplots()
    ax = plot_lc_with_bboxes(flat_lc, bboxes, ax=ax, lw=1)
    plt.show()



if __name__ == '__main__':
    main()
