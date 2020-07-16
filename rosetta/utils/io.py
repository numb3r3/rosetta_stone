import hashlib
import os
import shutil
import tarfile

import requests
from tqdm import tqdm

CHUNK_SIZE = 4096


def md5file(fname: str):
    hash_md5 = hashlib.md5()
    f = open(fname, 'rb')
    for chunk in iter(lambda: f.read(CHUNK_SIZE), b''):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def touch_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def download(url: str, md5sum: str, target_dir: str, save_name: str = None):
    """Download file from url to target_dir, and check md5sum."""

    touch_dir(target_dir)
    filepath = os.path.join(
        target_dir,
        url.split('/')[-1] if save_name is None else save_name)

    md5sum_checked = None
    if os.path.exists(filepath):
        md5sum_checked = md5file(filepath)

    if md5sum_checked == md5sum:
        print('Skip downloading existed %s!' % filepath)
        return filepath
    else:
        if os.path.exists(filepath):
            print('file md5 %s vs %s' % (md5sum_checked, md5sum))

        print('Begin to download ...')
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filepath, 'wb') as f:
                for data in tqdm(r.iter_content(chunk_size=CHUNK_SIZE)):
                    f.write(data)

        print('Download finished!')
    return filepath


def unpack(filepath: str, target_dir: str, rm_tar: bool = False):
    """Unpack the file to the target_dir."""
    print('Unpacking %s ...' % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar is True:
        os.remove(filepath)
