import pytest
import lightkurve as lk
import deep_transit as dt


def kepler_id_to_lc(kicid):
    from glob import iglob
    lc_fits_folder_path = f'docs/samples/*{kicid}*'
    lc_collection = []
    for i in iglob(lc_fits_folder_path + '/*llc.fits'):
        lc_collection.append(lk.read(i).remove_nans())
    return lk.LightCurveCollection(lc_collection)


def test_detection():
    import os
    lc = kepler_id_to_lc(11446443).stitch()
    lc.sort('time')
    lc = lc[lc.time.value < 150]

    print(len(lc))


    dt_obj = dt.DeepTransit(lc, is_flatten=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper': 3})
    if not os.path.exists('models/kepler_snr_model.pth'):
        os.system("wget http://paperdata.china-vo.org/ckm/kepler_snr_model.pth -P models/")
    bboxes = dt_obj.transit_detection('models/kepler_snr_model.pth', batch_size=3)

    assert len(bboxes) > 0
