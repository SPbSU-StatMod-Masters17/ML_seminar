import os
import errno


def ensure_dir_exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
