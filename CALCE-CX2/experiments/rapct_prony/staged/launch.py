"""Select a fresh experiment root before importing stage modules (no historical overwrite)."""
import argparse
import importlib
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('stage',choices=['prepare','diagnostics','stage1_run','stage2_run','stage3_run',
        'single_memory_run','memory_run','finalize','report','validate'])
    args,remaining=parser.parse_known_args()
    from . import prepare
    prepare.OUT=args.output.resolve()
    module=importlib.import_module('.'+args.stage,package=__package__)
    sys.argv=[sys.argv[0]]+remaining
    if args.stage=='report':module.generate()
    else:module.main()


if __name__=='__main__':main()
