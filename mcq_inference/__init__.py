from mcq_inference.common import MCQExample, MCQPrediction


def __getattr__(name):
    if name in {"DATASET_ORDER", "build_smoke_set"}:
        from mcq_inference.benchmark import DATASET_ORDER, build_smoke_set

        return {
            "DATASET_ORDER": DATASET_ORDER,
            "build_smoke_set": build_smoke_set,
        }[name]
    if name == "run_inference":
        from mcq_inference.runner import run_inference

        return run_inference
    raise AttributeError(name)


__all__ = [
    "DATASET_ORDER",
    "MCQExample",
    "MCQPrediction",
    "build_smoke_set",
    "run_inference",
]
