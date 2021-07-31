import pytest
import lightkurve as lk
import deep_transit as dt
import numpy as np


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

    dt_obj = dt.DeepTransit(lc, is_flat=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper': 3})
    if not os.path.exists('models/model_Kepler.pth'):
        os.system("wget http://paperdata.china-vo.org/ckm/model_Kepler.pth -P models/")
    bboxes = dt_obj.transit_detection('models/model_Kepler.pth', batch_size=3)

    assert len(bboxes) > 0, "number should more than 1"
    assert np.all((bboxes[:, (0,2,3,4)]<=1)&(bboxes[:, (0,2,3,4)]>=0)), "0, 2-4 shold be 0 to 1"
    assert np.all((bboxes[:,0]<=1)&(bboxes[:,0]>=0.99)), "confidence score shold be 0 to 1"
   
def test_detection_mge():
    import os
    lc = kepler_id_to_lc(11446443).stitch()
    lc.sort('time')
    lc = lc[lc.time.value < 150]

    dt_obj = dt.DeepTransit(lc, is_flat=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper': 3})
    if not os.path.exists('models/ckpt_deep_transit_5_v2.pkl'):
        os.system("wget http://paperdata.china-vo.org/ckm/ckpt_deep_transit_5_v2.pkl -P models/")
    bboxes = dt_obj.transit_detection('models/ckpt_deep_transit_5_v2.pkl', batch_size=3, backend='megengine')

    assert len(bboxes) > 0, "number should more than 1"
    assert np.all((bboxes[:, (0,2,3,4)]<=1)&(bboxes[:, (0,2,3,4)]>=0)), "0, 2-4 shold be 0 to 1"
    assert np.all((bboxes[:,0]<=1)&(bboxes[:,0]>=0.99)), "confidence score shold be 0 to 1"
   

def test_detection_on_GPU():
    import torch
    if not torch.cuda.is_available():
        return
    import os
    lc = kepler_id_to_lc(11446443).stitch()
    lc.sort('time')
    lc = lc[lc.time.value < 150]

    dt_obj = dt.DeepTransit(lc, is_flat=False, flatten_kwargs={'window_length': 0.5, 'sigma_upper': 3})
    if not os.path.exists('models/model_Kepler.pth'):
        os.system("wget http://paperdata.china-vo.org/ckm/model_Kepler.pth -P models/")
    bboxes = dt_obj.transit_detection('models/model_Kepler.pth', batch_size=3, device_str='cuda')

    assert len(bboxes) > 0, "number should more than 1"
    assert np.all((bboxes[:, (0,2,3,4)]<=1)&(bboxes[:, (0,2,3,4)]>=0)), "0, 2-4 shold be 0 to 1"
    assert np.all((bboxes[:,0]<=1)&(bboxes[:,0]>=0.99)), "confidence score shold be 0 to 1"