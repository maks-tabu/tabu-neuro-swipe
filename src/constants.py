from pathlib import Path
import polars as pl

GRID_TO_ID = {"default": 1, "extra": 2, "android_ru_east_slavic_separate_comma": 3}

# DATA_PATH = Path("/data")
# IMAGES_PATH = Path("/images")
MODEL_FILES_PATH = Path("src/model_files")

DATA_PATH = Path("/home/tabolin.maksim2/dataset/sw_task/result_noctx_10k")
IMAGES_PATH = Path("/home/tabolin.maksim2/dataset/images")


class AllWords:
    word_list = pl.read_csv(DATA_PATH / "voc.txt", new_columns=["words"], has_header=False)[
        "words"
    ].to_list()
