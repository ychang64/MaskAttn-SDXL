#!/usr/bin/env python3
"""Print benchmark setup requirements without silently downloading data or evaluators."""

from __future__ import annotations

import argparse

COMMANDS = {
    "t2i_compbench_spatial": "Clone the official T2I-CompBench++ repository, prepare its Spatial/UniDet prompts, then configure evaluator_command with {images} {prompts} {output}.",
    "geneval": "Clone the official GenEval repository and data, then configure evaluator_command with {images} {prompts} {output}.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(COMMANDS), required=True)
    args = parser.parse_args()
    print(COMMANDS[args.benchmark])


if __name__ == "__main__":
    main()
