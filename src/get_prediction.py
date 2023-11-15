from tslearn.metrics import dtw_path_from_metric
from src.model import TrOcrModel
from src.constants import DATA_PATH
from src.fcom import Fcom
import polars as pl
import numpy as np
import argparse
from typing import List


def swipe_mrr(row: pl.Series) -> float:
    koef = [1.0, 0.1, 0.09, 0.08]
    for i in range(min(len(row["new_predict"]), 4)):
        if row["new_predict"][i] == row["target"]:
            return koef[i]
    return 0.0


def save_submission(df: pl.DataFrame) -> None:
    results = []
    for predict in df["predict"].to_list():
        results.append(predict[:4] + (4 - len(predict[:4])) * [" "])
    pl.DataFrame(results, schema=["1", "2", "3", "4"]).write_csv(
        "submission.csv", include_header=False
    )


def get_ts_dtw(row: pl.Series, metric_name: str, fcom: Fcom) -> List[str]:
    predict = row["predict"][:8]
    grid_name = row["grid_name"]
    curve_source = [
        [x, y] for x, y in zip(row["x"], row["y"]) if (x is not None) and (y is not None)
    ]

    curves_predict = fcom(predict, grid_name)
    weights = []
    for curve in curves_predict:
        _, cost = dtw_path_from_metric(curve, curve_source, metric=metric_name)
        weights.append(cost)
    return weights


def weights_res(row: pl.Series) -> List[str]:
    predict = row["predict"][:8]
    score = -1 * np.array(row["score"])[: len(predict)]
    l2_dtw_path = np.array(row["l2"])
    cosine_dtw_path = np.array(row["cosine"])
    coef = [1, 0.15, 0.0009]
    if cosine_dtw_path is np.nan:
        weights = coef[0] * score + coef[1] * cosine_dtw_path + coef[2] * l2_dtw_path
    else:
        weights = coef[0] * score + coef[2] * l2_dtw_path
    return [predict[i] for i in np.argsort(weights)][:4]


def get_prediction(
    test_data: pl.DataFrame, global_test_mode: bool = True, batch_size: int = 16
) -> pl.DataFrame:
    trocr_model = TrOcrModel("result_model.pth", "tokenizer")
    test_data = trocr_model.predict(test_data, global_test_mode, batch_size)
    fcom = Fcom()

    for metric_name in ["l2", "cosine"]:
        test_data = test_data.with_columns(
            pl.struct(["predict", "score", "grid_name", "x", "y"])
            .map_elements(lambda row: get_ts_dtw(row, metric_name, fcom))
            .alias(metric_name)
        )
    test_data = test_data.with_columns(
        pl.struct(["predict", "score", "l2", "cosine"])
        .map_elements(weights_res)
        .alias("new_predict")
    )
    return test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_file", type=str, default="global_test")
    parser.add_argument("--batch_size", type=int, default=16)

    args, _ = parser.parse_known_args()
    name_file = args.name_file
    test_data = pl.read_parquet(DATA_PATH / f"{args.name_file}.pa")
    test_data = get_prediction(
        test_data, True if args.name_file == "global_test" else False, args.batch_size
    )
    if args.name_file == "global_test":
        save_submission(test_data)
    else:
        test_data = test_data.with_columns(
            pl.struct(["new_predict", "target"])
            .map_elements(lambda row: swipe_mrr(row))
            .alias("mrr")
        )
        print("SWIPE_MRR: ", test_data["mrr"].mean())
