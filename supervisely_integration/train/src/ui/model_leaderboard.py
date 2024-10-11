import pandas as pd
from src.utils import parse_yaml_metafile

from supervisely.app.widgets import Table, Card, Container


columns = [
    "Method",
    "Name",
    "Dataset",
    "Year",
    "box AP",
    "mask AP",
    "Inference Time (ms/im)",
    "Training Memory (GB)",
]


def _get_table_data(models: list, arch_name: str, year: int):
    keys = [
        # "method",
        "name",
        "dataset",
        # "year"
        "box AP",
        "mask AP",
        "inference_time",
        "train_memory",
    ]

    # collect rows
    rows = []
    for model in models:
        row = [model.get(k, (0 if i in [4, 5] else " ")) for i, k in enumerate(keys)]
        row.insert(0, arch_name)
        row.insert(3, year)
        row[-1] = float(row[-1])
        row[-2] = float(row[-2])
        rows.append(row)

    return rows


def get_rows_for_arch(model_meta: dict, task: str):
    metafile_path = "configs/" + model_meta["yml_file"]
    exclude = model_meta.get("exclude")
    _, models = parse_yaml_metafile(metafile_path, exclude)

    # filter models by task
    if "segmentation" in task.lower():
        task_name = "Instance Segmentation"
    else:
        task_name = "Object Detection"
    models = [m for m in models if task_name in m["tasks"]]

    # only COCO dataset
    models = [m for m in models if m["dataset"] == "COCO"]

    rows = _get_table_data(models, model_meta["model_name"], model_meta["year"])
    return rows


def get_models_table(models_meta: dict, task: str) -> pd.DataFrame:
    rows = []
    for model_meta in models_meta.values():
        rows += get_rows_for_arch(model_meta, task)
    if "segmentation" in task.lower():
        c_id = 5
    else:
        c_id = 4
    rows = sorted(rows, key=lambda x: x[c_id], reverse=True)
    df = pd.DataFrame(rows, columns=columns)
    df.index.name = "#"
    df = df.reset_index()
    df.iloc[:, -1] = df.iloc[:, -1].astype(float)
    return df


table = Table(per_page=30)

card = Card(
    title="Model Leaderboard 🏆",
    description="Compare all the models by metrics and performance",
    content=table,
    collapsable=True,
)

card.lock("Select task to update the table.")
card.collapse()


def update_table(models_meta: dict, task: str):
    df = get_models_table(models_meta, task)
    table.read_pandas(df)
    # if "segmentation" in task.lower():
    #     table.sort(6)
    # else:
    #     table.sort(5)


@table.click
def _getrid(x):
    pass
