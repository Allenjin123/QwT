import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


parent_dir = osp.abspath(os.path.join(__file__, '..', '..'))
print(parent_dir)
add_path(parent_dir)