#!/usr/bin/env python
"""合成蓝底矿石数据生成器（CLI 薄封装，实现见 ``oresorter/vision/synthetic.py``）。

用法::

    python scripts/make_synthetic.py --out /tmp/synth --classes A,B,C --per-class 50 \\
        --frames 5 --width 4096 --frame-height 2048 --seed 0

产物：``{out}/samples/{cls}/*.png|.json``（RGBA 小图样本）与 ``{out}/frames/{i:04d}.png`` + ``labels.json``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ 不是包：把仓库根目录加进 sys.path 以便直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oresorter.vision.synthetic import main

if __name__ == "__main__":
    sys.exit(main())
