from os import path

from misc import gpu_utils

from torch import hub

cvlab = lambda x: path.join('https://cv.snu.ac.kr/research/hub', x + '.pth')
_lookup_table = {
    'edsr-baseline-x2': cvlab('edsr_baseline_x2-2d2cf8ed'),
    'edsr-baseline-x4': cvlab('edsr_baseline_x4-6ecee39d'),
    'edsr-baseline-face-x8': cvlab('edsr_baseline_face_x8-051c6d7d'),
    'rrdb-x4': cvlab('rrdb_x4-59414275'),
    'rrdb-x4-new': cvlab('rrdb_x4_new-9d40f7f7'),
    'esrgan-x4': cvlab('esrgan_x4-fc7e3794'),
    'ddbpn-x4': cvlab('ddbpn_res_mr64_3_x4-abe50572'),
}

def download(name=None, url=None):
    map_location = gpu_utils.get_device()
    if name is not None:
        try:
            url = _lookup_table[name]
        except KeyError as k:
            print('Pre-trained model {} is not downloadable!'.format(k))

    state = hub.load_state_dict_from_url(
        url,
        # Use the default directory
        model_dir=None,
        map_location=map_location,
        check_hash=True,
    )

    if not isinstance(state, dict):
        state = {'model': state}

    if 'metadata' in state:
        print(state['metadata'])

    return state
