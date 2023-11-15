import numpy as np
from PIL import Image, ImageDraw
from src.constants import GRID_TO_ID, DATA_PATH
import polars as pl
from joblib import Parallel, delayed


def save_images(row: pl.Series) -> None:
    coord_x = row["x"]
    coord_y = row["y"]
    time = row["t"]

    points = np.array(
        [
            [x, y, t]
            for x, y, t in zip(coord_x, coord_y, time)
            if (x is not None) and (y is not None) and (t is not None)
        ]
    )

    image = Image.new("RGB", tuple(row["grid_size"]))
    draw = ImageDraw.Draw(image)

    if len(points) >= 2:
        coord_x = points[:, 0]
        coord_y = points[:, 1]
        time = points[:, 2]

        start_point = [coord_x[0], coord_y[0]]

        # add thickness of lines
        dt = time[1:] - time[:-1] + 0.5
        v = (
            np.linalg.norm(
                np.array([coord_x, coord_y]).T[1:]
                - np.array([coord_x, coord_y]).T[:-1],
                axis=1,
            )
            / dt
        )
        max_v = np.max(v)

        lwidths = 1 + (v / max_v) ** 0.5 * 10
        lwidths = np.nan_to_num(lwidths, nan=1)
        points = np.array([coord_x, coord_y]).T.reshape(-1, 1, 2)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        for segment, width in zip(segments, lwidths):
            if (segment[0][0] != segment[1][0]) or (segment[0][1] != segment[1][1]):
                draw.line([tuple(x) for x in segment], width=int(np.rint(width)))

        draw.ellipse(
            (
                start_point[0] - 15,
                start_point[1] - 15,
                start_point[0] + 15,
                start_point[1] + 15,
            ),
            fill="white",
            outline="white",
        )

    image = image.resize((224, 224))
    draw = ImageDraw.Draw(image)
    grid_number = GRID_TO_ID[row["grid_name"]]
    draw.text((0, 0), str(grid_number), (255, 255, 255))
    image.save(row["path"])


if __name__ == "__main__":
    for name_dataset in ["train", "val", "test", "global_test", "accepted"]:
        data = pl.read_parquet(DATA_PATH / f"{name_dataset}.pa")
        _ = Parallel(n_jobs=-1)(
            delayed(save_images)(row) for row in data.iter_rows(named=True)
        )
