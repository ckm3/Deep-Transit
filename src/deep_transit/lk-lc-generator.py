import lightkurve as lk
import pandas as pd
from tqdm.autonotebook import tqdm
from glob import glob, iglob
import numpy as np
import matplotlib.pyplot as plt
from wotan import slide_clip, flatten
from astropy.stats import sigma_clip
import warnings

warnings.filterwarnings("ignore")
import os

df = pd.read_csv('data/kepler_confirmed_transits.csv')
kepler_tce = pd.read_csv('data/kepler_tce.csv')


def kepler_id_to_lc(kicid):
    # prepare the kicid 

    # fits file location
    lc_fits_folder_path = f'/home/ckm/Data2/.lightkurve-cache/mastDownload/Kepler/*{kicid}*'

    lc_collection = []
    for i in iglob(lc_fits_folder_path + '/*llc.fits'):
        lc_collection.append(lk.read(i).remove_nans())
    return lk.LightCurveCollection(lc_collection)


def split_light_curve(lc_object, split_time=11):
    time_start = lc_object.time.value[0]
    while time_start < lc_object.time.value[-1]:
        time_stop = time_start + split_time
        yield time_start, time_stop
        time_start = time_stop - 3


def check_has_transits(lc_object, tce_results):
    for _, row_data in tce_results.iterrows():
        transit_period, transit_time0, transit_duration = row_data.at['tce_period'], row_data.at['tce_time0bk'], row_data.at['tce_duration']/24
        n = np.ceil((lc_object.time.value[0] - transit_time0 +  transit_duration/2)/transit_period).astype('int')
        m = np.floor((lc_object.time.value[-1] - transit_time0 - transit_duration/2)/transit_period).astype('int')

        for i in range(n, m+1):
            t0 = transit_time0 +  transit_period*i - transit_duration/2
            t1 = t0 + transit_duration

            sub_lc_obj = lc_object[(lc_object.time.value>=t0)&(lc_object.time.value<=t1)]

    #         if np.any(sub_lc_obj.flux.value-1<=-3*np.nanmedian(np.abs(lc_object.flux.value - np.nanmedian(lc_object.flux.value)))):
            if np.any(sub_lc_obj.flux.value-1<=-3*np.nanstd(lc_object.flux.value)):
    #             print(sub_lc_obj.flux.value-1, -3*np.nanstd(lc_object.flux.value))
    #         if len(sub_lc_obj) > 1:
                yield (t0, t1)
            else:
                yield False


def plot_light_curve_to_image(kicid, light_curve_object, check_results, save_box=True):
    time_begin, time_end = light_curve_object.remove_nans().time.value[0], light_curve_object.remove_nans().time.value[-1]

    temp_path = '/home/ckm/PycharmProjects/Deep-LC/YOLO-LC-transit/Data'

    # file_name = f'{temp_path}/transit-images/{kicid}_{time_begin:.0f}_{time_end:.0f}.png'
    #
    # if os.path.exists(file_name):
    #     return None
    #
    # fig, ax = plt.subplots(1, figsize=(4.16, 4.16), dpi=100, frameon=False)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax.axis('off')
    # ax.margins(0, 0)
    # ax.plot(light_curve_object.time.value, light_curve_object.flux.value, 'k.-', lw=1)
    # fig.savefig(file_name, dpi=100)
    # plt.close()

    if save_box:
        for check_value in check_results:
            if check_value:
                t0, t1 = check_value
                duration = t1 - t0
                time_length = time_end - time_begin

                x0 = t0 + duration / 2
                x0_scale = (x0 - time_begin) / time_length

                width = duration * 2
                width_scale = width / time_length
                if width_scale < 0.02:
                    width_scale = 0.02

                flux_in_transit_range = light_curve_object[
                    (light_curve_object.time.value >= (x0 - width_scale / 2 * time_length)) & (
                                light_curve_object.time.value <= (x0 + width_scale / 2 * time_length))].flux.value
                light_curve_object_flux_range = np.nanmax(light_curve_object.flux.value) - np.nanmin(
                    light_curve_object.flux.value)

                # height = np.nanmax(flux_in_transit_range) - np.nanmin(flux_in_transit_range)
                height = np.nanpercentile(light_curve_object.flux.value, 95) - np.nanmin(flux_in_transit_range)
                height_scale = height / light_curve_object_flux_range

                y0 = np.nanmin(flux_in_transit_range) + height / 2
                y0_scale = (np.nanmax(light_curve_object.flux.value) - y0) / light_curve_object_flux_range
                # print(x0_scale, y0_scale, width_scale, height_scale)
                pd.DataFrame([[x0_scale, y0_scale, width_scale, height_scale]]).to_csv(
                    f'{temp_path}/transit-labels/{kicid}_{time_begin:.0f}_{time_end:.0f}.txt',
                    mode='a', index=False, header=None)


def main():
    for kicid in tqdm(df.KIC.drop_duplicates(), position=0, desc='Outter'):
    # for kicid in [8415200, 9886361]:
        # kicid = 9886361
        # if glob(f'/home/ckm/PycharmProjects/Deep-LC/data/labels/{kicid}*'):
        #     continue
        lc_stitched = kepler_id_to_lc(kicid).stitch()
        for start_time, stop_time in tqdm(split_light_curve(lc_stitched), position=1, leave=False, desc='Inner'):
            mask = (lc_stitched.time.value >= start_time) & (lc_stitched.time.value <= stop_time)
            lc_selected = lc_stitched[mask]
            if len(lc_selected) < 5 / 0.0204:
                continue

            tce_results = kepler_tce[kepler_tce.kepid == kicid][
                ['tce_period', 'tce_duration', 'tce_time0bk', 'tce_duration_err', 'tce_time0bk_err', 'tce_period_err']]
            if tce_results.size == 0:
                continue

            clipped_flux = slide_clip(
                lc_selected.time.value,
                lc_selected.flux.value,
                window_length=0.5,
                low=3,
                high=3,
                method='std',  # mad or std
                center='mean'  # median or mean
            )

            _, trend_flux = flatten(
                lc_selected.time.value,  # Array of time values
                clipped_flux,  # Array of flux values
                method='biweight',
                window_length=0.5,  # The length of the filter window in units of ``time``
                edge_cutoff=0.5,  # length (in units of time) to be cut off each edge.
                break_tolerance=0.5,  # Split into segments at breaks longer than that
                return_trend=True,  # Return trend and flattened light curve
                cval=5.0  # Tuning parameter for the robust estimators
            )

            flatten_flux = sigma_clip(lc_selected.flux.value / trend_flux, sigma_upper=3, sigma_lower=20,
                                      cenfunc=np.nanmedian, stdfunc=np.nanstd)

            flatten_lc = lk.LightCurve(time=lc_selected.time.value, flux=flatten_flux)

            check_results = list(check_has_transits(flatten_lc.remove_nans(), tce_results))
            if np.any(check_results):
                plot_light_curve_to_image(kicid, flatten_lc, check_results)
        # break


if __name__ == "__main__":
    main()
