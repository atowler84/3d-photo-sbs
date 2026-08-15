"""Download the checkpoints the portable build ships with.

Straight into a plain folder rather than a Hugging Face cache: `from_pretrained`
on a directory never looks at the network, which is what makes the packaged app
start the same on a machine that has no internet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from stereocraft.depth import MODELS  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    parser.add_argument("-o", "--output", required=True, help="folder to fill with <model>/")
    args = parser.parse_args(argv)

    import shutil

    from huggingface_hub import snapshot_download

    for name in args.models:
        target = os.path.join(args.output, name)
        print(f"fetching {MODELS[name]} -> {target}")
        snapshot_download(
            MODELS[name],
            local_dir=target,
            # The repos carry the same weights twice, as .bin and .safetensors;
            # only one of them is wanted, and it is the one that loads faster.
            allow_patterns=["*.json", "*.safetensors"],
        )
        # Bookkeeping for a re-download that will never happen here, and the app
        # would ship it to every machine.
        shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
