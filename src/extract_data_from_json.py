import polars as pl
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from src.constants import DATA_PATH, IMAGES_PATH, AllWords


def extract_data(data_path: str, with_target: bool = False) -> pl.DataFrame:
    if with_target:
        target_column = [("target", pl.Utf8)]
    else:
        target_column = []

    schema = target_column + [
        ("x", pl.List(pl.UInt16)),
        ("y", pl.List(pl.UInt16)),
        ("t", pl.List(pl.UInt16)),
        ("grid_size", pl.List(pl.UInt16)),
        ("grid_name", pl.Utf8),
    ]

    data = []
    total_data = pl.DataFrame([], schema=schema)
    rng = range(0, 1_000_000_000, 500_000)
    j = 0
    with open(data_path, "r", encoding="utf-8") as file:
        for i, line in tqdm(enumerate(file)):
            js = json.loads(line)
            grid_name = js["curve"]["grid"]["grid_name"]
            width = js["curve"]["grid"]["width"]
            height = js["curve"]["grid"]["height"]
            x = js["curve"]["x"]
            y = js["curve"]["y"]
            t = js["curve"]["t"]
            if with_target:
                word = js["word"]
                data.append((word, x, y, t, [width, height], grid_name))
            else:
                data.append((x, y, t, [width, height], grid_name))

            if i == rng[j]:
                data = pl.DataFrame(data, schema=schema)
                total_data = pl.concat([total_data, data], how="vertical")
                data = []
                j += 1

    data = pl.DataFrame(data, schema=schema)
    total_data = pl.concat([total_data, data], how="vertical")
    return total_data


if __name__ == "__main__":
    test_data = extract_data(DATA_PATH / "valid.jsonl")
    test_target = pl.read_csv(
        DATA_PATH / "valid.ref", has_header=False, new_columns=["target"]
    )
    test_data = pl.concat([test_data, test_target], how="horizontal")
    test_data = test_data.with_row_count("index")
    test_data = test_data.with_columns(
        pl.col("index")
        .map_elements(lambda row: IMAGES_PATH / f"test/{str(row).zfill(9)}.jpg")
        .alias("path")
    )
    test_data.write_parquet(DATA_PATH / "test.pa")

    train_data = extract_data(DATA_PATH / "train.jsonl", with_target=True)
    train_data, val_data = train_test_split(
        train_data, test_size=0.0016666666666666668, random_state=42, shuffle=True
    )

    train_data = train_data.with_row_count("index").with_columns(
        pl.col("index")
        .map_elements(lambda row: IMAGES_PATH / f"train/{str(row).zfill(9)}.jpg")
        .alias("path")
    )
    train_data.write_parquet(DATA_PATH / "train.pa")

    val_data = val_data.with_row_count("index").with_columns(
        pl.col("index")
        .map_elements(lambda row: IMAGES_PATH / f"val/{str(row).zfill(9)}.jpg")
        .alias("path")
    )
    val_data.write_parquet(DATA_PATH / "val.pa")

    global_test_data = extract_data(DATA_PATH / "test.jsonl")
    global_test_data = global_test_data.with_row_count("index").with_columns(
        pl.col("index")
        .map_elements(
            lambda row: IMAGES_PATH / f"global_test_data/{str(row).zfill(9)}.jpg"
        )
        .alias("path")
    )
    global_test_data.write_parquet(DATA_PATH / "global_test.pa")

    accepted_data = extract_data(DATA_PATH / "accepted", with_target=True)
    accepted_data = accepted_data.filter(
        pl.col("grid_name") == "android_ru_east_slavic_separate_comma"
    )
    accepted_data = accepted_data.filter(pl.col("target").str.lengths() > 4)
    accepted_data = accepted_data.filter(pl.col("target").is_in(AllWords.word_list))
    accepted_data = accepted_data.with_row_count("index").with_columns(
        pl.col("index")
        .map_elements(lambda row: IMAGES_PATH / f"accepted/{str(row).zfill(9)}.jpg")
        .alias("path")
    )
    accepted_data.write_parquet(DATA_PATH / "accepted.pa")
