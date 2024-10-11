import yaml

TASK_NAMES = ["Object Detection", "Instance Segmentation"]


def is_exclude(name: str, exclude):
    if isinstance(exclude, str):
        if is_exclude_pattern(name, exclude):
            return True
    elif isinstance(exclude, list):
        for e in exclude:
            if is_exclude_pattern(name, e):
                return True
    return False


def is_exclude_pattern(name: str, exclude: str):
    if exclude.endswith("*"):
        if name.startswith(exclude[:-1]):
            return True
    elif exclude.startswith("*"):
        if name.endswith(exclude[1:]):
            return True
    else:
        raise NotImplementedError(f"can't parse the exculde pattern: {exclude}")
    return False


def parse_yaml_metafile(yaml_file, exclude: str = None):
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)

    collections = {}  # Name: metadata
    yaml_models = []
    if isinstance(yaml_content, dict):
        if yaml_content.get("Collections"):
            if isinstance(yaml_content["Collections"], list):
                for c in yaml_content["Collections"]:
                    collections[c["Name"]] = c
            else:
                raise NotImplementedError()
        else:
            print(f"Has not collections: {yaml_file}.")
        if yaml_content.get("Models"):
            yaml_models = yaml_content["Models"]
    elif isinstance(yaml_content, list):
        yaml_models = yaml_content
        print(f"Only list: {yaml_file}.")
    else:
        raise NotImplementedError()

    models = []
    for model in yaml_models:
        # skip by exclude regexp
        name: str = model["Name"]
        if is_exclude(name, exclude):
            continue

        tasks = [r["Task"] for r in model["Results"]]
        # skip if task not in TASK_NAMES
        if all([task not in TASK_NAMES for task in tasks]):
            continue

        # skip if has not weights
        if not model.get("Weights"):
            print(f"skip {model['Name']} in {yaml_file}, weights don't exists.")
            continue

        # collect metrics
        metrics = {}
        for result in model["Results"]:
            for metric_name, metric_val in result["Metrics"].items():
                metrics[metric_name] = metric_val
            metrics["dataset"] = result["Dataset"]

        # collect metadata
        metadata = {}
        if model["Metadata"].get("Training Memory (GB)"):
            metadata["train_memory"] = model["Metadata"].get("Training Memory (GB)")
        if model["Metadata"].get("inference time (ms/im)"):
            if isinstance(model["Metadata"].get("inference time (ms/im)"), list):
                metadata["inference_time"] = model["Metadata"].get("inference time (ms/im)")[0][
                    "value"
                ]
            else:
                metadata["inference_time"] = model["Metadata"].get("inference time (ms/im)")
        if model["Metadata"].get("Epochs"):
            metadata["train_epochs"] = model["Metadata"].get("Epochs")
        if model["Metadata"].get("Iterations"):
            metadata["train_iters"] = model["Metadata"].get("Iterations")

        model_item = {
            "name": model["Name"],
            "method": model["In Collection"],
            "config": model["Config"],
            "tasks": tasks,
            "weights": model["Weights"],
            **metrics,
            **metadata,
        }
        models.append(model_item)

    return collections, models
