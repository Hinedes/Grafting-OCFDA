import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from nirvana_utils import copy_out_to_snapshot

output_dir = sys.argv[1]
if not int(os.environ.get("LOCAL_RANK") or 0):
    copy_out_to_snapshot(output_dir)
