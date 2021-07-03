import io

import torch
from deep_transit import config
import warnings
import itertools

import numpy as np
from PIL import Image
from tqdm import tqdm

import lightkurve as lk
from deep_transit.model import YOLOv3
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from wotan import flatten

from deep_transit.utils import (
    warning_on_one_line,
    predict_bboxes,
    non_max_suppression,
    load_model)

torch.set_flush_denormal(True)  # Fixing a bug caused by Intel CPU

warnings.formatwarning = warning_on_one_line  # Raise my own warns


def detrend_light_curve(lc_object, window_length=0.5, edge_cutoff=0.5, break_tolerance=0.5, cval=5.0, sigma_upper=3,
                        sigma_lower=20):
    """
    Detrend a light curve for upcoming transit searching with wotan biweight method and sigma clipping
    https://github.com/hippke/wotan
    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
                Input light curve object
    window_length : float
                The length of the filter window in units of ``time``, default is 0.5
    edge_cutoff : float
                Length (in units of time) to be cut off each edge, default is 0.5
    break_tolerance : float
                Split into segments at breaks longer than that, default is 0.5
    cval : float
                Tuning parameter for the robust estimators, default is 5.0
    sigma_upper : float

    sigma_lower : float
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


def _light_curve_to_image_array(lc_object, flux_range, exp_time):
    """
    Convert light curve slice to image array
    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance
    flux_range : tuple
                Flux range in 30-day window in format: (flux_min, flux_max)
    exp_time : float
                Exposure time of the input light curve
    Returns
    -------
    img_arr : np.ndarray
                Numpy image array
    """
    with plt.rc_context({'backend': 'agg'}):
        io_buf = io.BytesIO()
        io_buf.seek(0)
        fig, ax = plt.subplots(1, figsize=(4.16, 4.16), dpi=100, frameon=False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.margins(0, 0)
        if exp_time > 0.00417:
            ax.plot(lc_object.time.value, lc_object.flux.value, ls='-', marker='o', lw=72 / fig.dpi / 2,
                    ms=72 / fig.dpi * 2,
                    color='black')
        else:
            ax.plot(lc_object.time.value, lc_object.flux.value, '.', ms=72 / fig.dpi, color='black')
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
    def __init__(self, lc_object=None, time=None, flux=None, flux_err=None, is_flatten=False, exp_time='auto',
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
        flux : `~astropy.time.Time` or iterable
                    Flux values for every time point.
        flux_err : `~astropy.time.Time` or iterable
                    Uncertainty on each flux data point.
        is_flatten : bool
                    True when receiving a flatten light, False will use the built-in flatten method.
        exp_time : float
                    exposure time in unit of ``time``.
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
        if is_flatten is True:
            self.lc = lc_object
        else:
            self.lc = detrend_light_curve(lc_object, **flatten_kwargs)

        if exp_time == 'auto':
            self.exp_time = np.nanmin(np.diff(self.lc.time.value))
        elif not isinstance(exp_time, float):
            raise TypeError("`exp_time` should be a float type value")

    def save_sample(self, transit_masks, file_name_base=None):
        if file_name_base is None:
            # ToDo: save sample to training Data Folder
            import string
            import random
            file_name_base = ''.join(
                random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))

        for start_time, stop_time in tqdm(list(_split_light_curve(self.lc))):
            mask = (self.lc.time.value >= start_time) & (self.lc.time.value <= stop_time)
            selected_lc = self.lc[mask]
            if len(selected_lc) < 5 / self.exp_time:
                continue
            flatten_lc = detrend_light_curve(selected_lc)

    def _splited_lc_generator(self):
        time_initial_index = 0
        for time_block_index in np.append((np.diff(self.lc.time.value) > 10).nonzero()[0], len(self.lc) - 1):
            lc_block = self.lc[time_initial_index:time_block_index]
            time_initial_index = time_block_index + 1
            for start_time, stop_time in list(_split_light_curve(lc_block, split_time=30, back_step=5)):
                mask = (lc_block.time.value >= start_time) & (lc_block.time.value <= stop_time)
                selected_lc = lc_block[mask]
                if len(selected_lc) < 20 / self.exp_time:
                    continue
                flux_min, flux_max = np.nanmin(selected_lc.flux.value) * 1.02 - 0.02 * np.nanmax(
                    selected_lc.flux.value), np.nanmax(selected_lc.flux.value)
                for t0, t1 in _split_light_curve(selected_lc.remove_nans(), split_time=10, back_step=3):
                    mask = (selected_lc.time.value >= t0) & (selected_lc.time.value <= t1)
                    splited_flatten_lc = selected_lc[mask]
                    if len(splited_flatten_lc) < 5 / self.exp_time:
                        continue
                    img_arr = _light_curve_to_image_array(splited_flatten_lc, (flux_min, flux_max),
                                                          exp_time=self.exp_time)
                    image = config.data_transforms(np.array(Image.fromarray(img_arr).convert("L")))
                    # image = torch.unsqueeze(image, 0)
                    # image = np.expand_dims(image, 0)
                    yield splited_flatten_lc, image, flux_min, flux_max

    def _data_loader(self, batch_size=1):
        it = iter(self._splited_lc_generator())
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunk = np.array(list(itertools.islice(it, batch_size)))
            if chunk.size == 0:
                return
            yield chunk

    def transit_detection(self, local_model_path, batch_size=2):
        """
        Searching transit signals from a given light curve.
        Parameters
        ----------
        batch_size : int
                    batch size for increasing detection speed, especially useful for GPU
                    default value is 2, if using GPU, it can be higher depending on the limitation of the GPU memory.
        exp_time : str or float
                    exposer time (cadence) of the light curve in unit of days.
                    For example, Kepler long cadence is 0.0204.
                    Default value is 'auto', which is calculated from the minimum separation of time.

        return_real_unit : bool
                    if True, this will return a list of bounding boxes,
                    with unit of real time and flux. The format is [[confidence, x0, y0, width, height]]

        Returns
        -------
        final_bboxes : np.ndarray
                    An (N, 5) shape numpy.ndarray of bounding boxes.
        """

        model = YOLOv3().to(config.DEVICE)
        load_model(local_model_path, model)
        model.eval()

        real_unit_bboxes = []
        rough_length = int(len(self.lc) * self.exp_time / 25 * 4 / batch_size)
        warnings.warn('The total number of progress bar is roughly estimated')
        for data in tqdm(self._data_loader(batch_size=batch_size), total=rough_length):
            lc_data = data[:, 0]
            flux_min, flux_max = data[:, 2], data[:, 3]
            image_data = torch.tensor(np.stack(data[:, 1]), device=config.DEVICE)

            predicted_bboxes = predict_bboxes(image_data,
                                              model=model,
                                              iou_threshold=config.NMS_IOU_THRESH,
                                              anchors=config.ANCHORS,
                                              threshold=config.CONF_THRESHOLD
                                              )

            for index, bboxes in enumerate(predicted_bboxes):
                predicted_bboxes_in_real_unit = _bounding_box_to_time_flux(lc_data[index], bboxes,
                                                                           (flux_min[index], flux_max[index]))
                real_unit_bboxes += predicted_bboxes_in_real_unit

                # fig, ax = plt.subplots(1, constrained_layout=True)
                # lc_data[index].scatter(ax=ax)
                # from matplotlib.patches import Rectangle
                # from matplotlib.collections import PatchCollection
                # recs = []
                # for real_mask in predicted_bboxes_in_real_unit:
                #     # ax.axvspan(real_mask[1] - real_mask[3] / 2,
                #     #            real_mask[1] + real_mask[3] / 2,
                #     #            ec='k', fc='none')
                #     # print(real_mask)
                #     rec = Rectangle((real_mask[1] - real_mask[3] / 2, real_mask[2] - real_mask[4] / 2),
                #                     real_mask[3],
                #                     real_mask[4], fill=False, color='lime')
                #     recs.append(rec)
                # pc = PatchCollection(recs, facecolor='none', edgecolor='lime', lw=1)
                # ax.add_collection(pc)
                # plt.show()
        # break
        # final_bboxes = sorted(non_max_suppression(real_unit_bboxes, config.NMS_IOU_THRESH, config.CONF_THRESHOLD),
        #                       key=lambda x: x[1], reverse=False)
        final_bboxes = np.array(non_max_suppression(real_unit_bboxes, config.NMS_IOU_THRESH, config.CONF_THRESHOLD))
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
                bbox=dict(alpha=0.5),
                clip_on=True
            )

        pc = PatchCollection(recs, facecolor='none', edgecolor='lime', lw=1, zorder=3)
        ax.add_collection(pc)
        ax.set_xlabel('Time (day)')
        ax.set_ylabel('Normalized Flux')
    return ax


def select_lc_from_bboxes(lc_object, bboxes, bboxes_format='LC'):
    """

    Parameters
    ----------
    bboxes
    lc_object
    bboxes_format

    Returns
    -------

    """
    range_logic = False
    for bbox in bboxes:
        if bboxes_format == 'LC':
            t0 = bbox[1] - bbox[3] / 2
            t1 = bbox[1] + bbox[3] / 2
            y0 = bbox[2] - bbox[4] / 2
            y1 = bbox[2] + bbox[4] / 2
            range_logic = range_logic | (lc_object.time.value >= t0) & (lc_object.time.value <= t1) & (
                    lc_object.flux.value >= y0) & (lc_object.flux.value <= y1)
    return lc_object[range_logic]


def __show_light_curve_image(image_array):
    """
    Private debugging function
    """
    from PIL import Image
    img_arr = np.array(Image.fromarray(image_array).convert("L"))
    fig, ax = plt.subplots(1)
    ax.imshow(img_arr, cmap='binary_r', origin='upper')
    plt.show()


def kepler_id_to_lc(kicid):
    from glob import iglob
    # prepare the kicid

    # fits file location
    lc_fits_folder_path = f'/home/ckm/Data2/.lightkurve-cache/mastDownload/Kepler/*{kicid}*'

    lc_collection = []
    for i in iglob(lc_fits_folder_path + '/*llc.fits'):
        lc_collection.append(lk.read(i).remove_nans())
    return lk.LightCurveCollection(lc_collection)


def tess_id_to_lc(ticid, product='spoc'):
    from glob import glob
    lc_collection = []
    if product == 'spoc':
        lc_file_paths = glob(f'/home/ckm/.lightkurve-cache/mastDownload/TESS/*{ticid}*s/*lc.fits')
    elif product == 'qlp':
        lc_file_paths = glob(f'/home/ckm/.lightkurve-cache/mastDownload/HLSP/*qlp*{ticid}*/*lc.fits')

    for i in lc_file_paths:
        lc_collection.append(lk.read(i).remove_nans())
    return lk.LightCurveCollection(lc_collection)


def main():
    import matplotlib.pyplot as plt

    # search_result = lk.search_lightcurve('KIC 7047824', author='Kepler')
    # lc = search_result.download_all().stitch()
    # print(lc.time.value[-1])
    # lc = kepler_id_to_lc(4178606).stitch()
    # lc = kepler_id_to_lc(757076).stitch()

    # lc = kepler_id_to_lc(1025494).stitch()
    # lc = kepler_id_to_lc(11446443).stitch() # Kepeler-1b

    # lc = kepler_id_to_lc(11554435).stitch()
    # lc = kepler_id_to_lc(10874614).stitch()
    # lc = kepler_id_to_lc(8692861).stitch()

    # lc = tess_id_to_lc(100014359).stitch()
    # lc.sort('time')

    # lc = lc[lc.time.value > 1500]

    import pandas as pd
    data = pd.read_csv('../../1809.05967/pimensatesslightcurve.csv')
    lc = lk.LightCurve(time=data.time, flux=data.flux)

    # t = lc.time.value[0]
    # sub_sample_time = []
    # sub_sample_flux = []
    # while t <= lc.time.value[-1]:
    #     sub_lc = lc[(lc.time.value > t) & (lc.time.value < t + 0.0204)]
    #     sub_sample_time.append(t + 0.0204 / 2)
    #     sub_sample_flux.append(np.nanmean(sub_lc.flux.value))
    #     t += 0.0204
    # lc = lk.LightCurve(time=sub_sample_time, flux=sub_sample_flux)

    # lc.plot()
    # print(lc.meta)
    # lc.plot()
    # plt.show()
    dt = DeepTransit(lc, is_flatten=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper':3})
    flat_lc = detrend_light_curve(lc, window_length=0.5)

    # flat_lc.plot()
    # plt.show()
    # print(list(dt_lc._split_light_curve()))

    # bboxes = dt.transit_detection('models/Model_Kepler.pth', batch_size=3)
    bboxes = dt.transit_detection('models/Model_TESS.pth', batch_size=2)
    # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=False)

    fig, ax = plt.subplots()
    ax = plot_lc_with_bboxes(flat_lc, bboxes, ax=ax, lw=1)
    # ax.set_xlim(1560, 1570)
    plt.show()

    # import pickle
    # with open('bboxes.list', 'wb') as f:
    #     pickle.dump(bboxes, f)

    # with open('bboxes.list', 'rb') as f:
    #     bboxes = pickle.load(f)

    # x = []
    # y = []
    # for bbox in bboxes:
    #     t0 = bbox[1] - bbox[3] / 2
    #     t1 = bbox[1] + bbox[3] / 2
    #     y0 = bbox[2] - bbox[4] / 2
    #     y1 = bbox[2] + bbox[4] / 2
    #     x.append([t0, t0, t1, t1])
    #     y.append([y1, y0, y0, y1])
    # plt.plot(np.array(x).flatten(), np.array(y).flatten(), '.')
    # plt.show()

    # new_selected_lc = select_lc_from_bboxes(bboxes, flat_lc)
    # new_selected_lc.scatter()
    # plt.show()

    # from astropy.timeseries import BoxLeastSquares
    # x, y = np.array(x).flatten(), np.array(y).flatten()
    # from scipy import interpolate
    # f = interpolate.interp1d(x, y)
    # xnew = np.arange(x[0], x[-1], 0.1)
    # models = BoxLeastSquares(xnew, f(xnew))
    # periodogram = models.autopower(0.28)
    # plt.plot(periodogram.period, periodogram.power)
    # plt.show()

    # from pdmpy import pdm
    # freq, theta = pdm(new_selected_lc.time.value, new_selected_lc.flux.value, f_min=0.1, f_max=1, delf=1e-3, nbin=5)
    # plt.plot(freq, theta)

    # plt.show()
    # print(freq[np.argmin(theta)])
    # flat_lc.fold(1/0.309, normalize_phase=True).scatter()
    # plt.show()

    # dt_lc.transit_detection()
    # dt_lc.estimate_cdpp()
    # print(dt_lc.meta)

    # dataset = LightCurveDataset()
    # loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    # for i in loader:
    #     print(i)


if __name__ == '__main__':
    main()
