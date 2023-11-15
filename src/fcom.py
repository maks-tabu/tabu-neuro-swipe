import os.path
import typing as tp
import numpy as np
from dataclasses import dataclass, field
from src.constants import DATA_PATH
from typing import List
import os


@dataclass
class Hitbox:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self):
        self.cx, self.cy = (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def is_in(self, x, y):
        return self.left <= x < self.right and self.top <= y < self.bottom

    @property
    def central_coords(self):
        return [self.cx, self.cy]


class Grid:
    def __init__(self, grid_info: tp.Dict):
        self.hitboxes = self._build_hitboxes(grid_info)

    def get_the_nearest_hitbox(self, x, y) -> str:
        for label, hitbox in self.hitboxes.items():
            if hitbox.is_in(x, y):
                return label
        return "a"

    def get_centered_curve(self, word: str) -> np.ndarray:
        curve = []
        for idx, l in enumerate(word):
            if l not in self.hitboxes:
                continue
            curve.append(self.hitboxes[l].central_coords)
        return np.array(curve, dtype=np.float32)

    @staticmethod
    def _build_hitboxes(grid_info: tp.Dict) -> tp.Dict[str, Hitbox]:
        def hitbox_from_key(key) -> tp.Optional[tp.Tuple[str, Hitbox]]:
            h_name = key.get("label") or key.get("action")
            if h_name is None or len(h_name) > 1:
                return None, None

            h = key["hitbox"]
            x, y, w, h = h["x"], h["y"], h["w"], h["h"]

            return h_name, Hitbox(top=y, bottom=y + h, left=x, right=x + w)

        hitboxes = {
            h_name: hitbox
            for (h_name, hitbox) in map(hitbox_from_key, grid_info["keys"])
            if h_name
        }
        return hitboxes


class Fcom:
    def __init__(self):
        self.grids = self.extract_grid_info()

    def __call__(self, words: List[str], grid_name: str) -> List[np.ndarray]:
        return [self.grids[grid_name].get_centered_curve(cand) for cand in words]

    @staticmethod
    def extract_grid_info():
        grids = {}
        assert os.path.isfile(DATA_PATH / "train.jsonl")
        with open(DATA_PATH / "train.jsonl", "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if len(grids) == 2:
                    break
                res = eval(line)
                grid_name = res["curve"]["grid"]["grid_name"]
                if grid_name not in grids:
                    grids[grid_name] = Grid(res["curve"]["grid"])
        return grids
