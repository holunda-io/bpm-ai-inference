import functools
import logging
import os
import time
from pathlib import Path
from typing import Type

import cpuinfo
from huggingface_hub import HfFileSystem
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer, \
    ORTModelForQuestionAnswering, ORTModel
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig, AutoOptimizationConfig
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

FILENAME_ONNX = "model.onnx"
FILENAME_OPTIMIZED_ONNX = "model_optimized.onnx"
FILENAME_QUANTIZED_ONNX = "model_quantized.onnx"


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        rounded = round(run_time, 3)
        logger.debug("Finished {} in {} secs".format(repr(func.__name__), rounded))
        return value
    return wrapper


def _holisticon_onnx_repository_id(model_name: str) -> str:
    return f"holisticon/{model_name.split('/')[1]}-onnx"


def get_optimized_model(model: str, task: str, optimization_level: int = None, push_to_hub: bool = False):
    model_name = model
    model_dir = os.environ['HF_HOME'] + "/onnx/" + model.replace("/", "--")
    tokenizer = AutoTokenizer.from_pretrained(model)

    optimization_level = optimization_level or int(os.environ.get("OPTIMIZATION_LEVEL", "2"))

    onnx = optimization_level >= 1
    optimize = optimization_level == 2
    quantize = optimization_level == 3

    if onnx:
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        if optimize:
            model = _optimize(model, model_dir, task, push_to_hub=push_to_hub)
        elif quantize:
            model = _quantize(model, model_dir, task, push_to_hub=push_to_hub)
        else:
            model = _export_to_onnx(model, model_dir, task)

    if push_to_hub:
        model.push_to_hub(model.model_save_dir, _holisticon_onnx_repository_id(model_name))

    return model, tokenizer


def _check_exists_on_hub(repository_id: str, filename: str) -> str | None:
    fs = HfFileSystem()
    hub_models = fs.glob(f"{repository_id}/**/{filename}", maxdepth=3)
    if len(hub_models) == 1:
        model_path = str(Path(hub_models[0].removeprefix(repository_id + "/")).parent)
    else:
        model_path = None
    return model_path


def _try_load_from_hub(repository_id: str, filename: str, model_class: Type[ORTModel]) -> ORTModel | None:
    # check original repo
    hub_folder = _check_exists_on_hub(repository_id, filename)
    if hub_folder:
        logger.debug(f"Loading existing onnx model from: {hub_folder} on {repository_id}")
        return model_class.from_pretrained(repository_id, subfolder=hub_folder, file_name=filename)
    # check if the amazing people at holisticon conveniently shared a pre-optimized model
    holi_repo = _holisticon_onnx_repository_id(repository_id)
    hub_folder = _check_exists_on_hub(holi_repo, filename)
    if hub_folder:
        logger.debug(f"Loading existing onnx model from: {hub_folder} on {holi_repo}")
        return model_class.from_pretrained(holi_repo, subfolder=hub_folder, file_name=filename)
    return None


def _task_to_model(task: str):
    match task:
        case "zero-shot-classification":
            return ORTModelForSequenceClassification
        case "question-answering":
            return ORTModelForQuestionAnswering


@timer
def _export_to_onnx(repository_id: str, model_dir, task):
    model_class = _task_to_model(task)

    # try to load from hub
    pre_optimized_model = _try_load_from_hub(repository_id, FILENAME_ONNX, model_class)
    if pre_optimized_model:
        return pre_optimized_model

    # export to onnx
    return model_class.from_pretrained(repository_id, export=True)


@timer
def _optimize(repository_id: str, model_dir, task, push_to_hub=False):
    model_class = _task_to_model(task)

    # try to load from hub or cache
    pre_optimized_model = _try_load_from_hub(repository_id, FILENAME_OPTIMIZED_ONNX, model_class)
    if pre_optimized_model:
        return pre_optimized_model

    # try to load existing local file
    model_file = model_dir + "/" + FILENAME_OPTIMIZED_ONNX
    if os.path.exists(model_file):
        return model_class.from_pretrained(model_dir, file_name=FILENAME_OPTIMIZED_ONNX)

    # optimize
    model = _export_to_onnx(repository_id, model_dir, task)
    optimizer = ORTOptimizer.from_pretrained(model)
    if push_to_hub:
        config = AutoOptimizationConfig.O2()
    else:
        config = OptimizationConfig(optimization_level=99)  # enable all optimizations
    optimizer.optimize(
        optimization_config=config,
        save_dir=model_dir
    )
    size = os.path.getsize(model_file) / (1024 * 1024)
    logger.debug(f"Optimized Onnx Model file size: {size:.2f} MB")
    return model_class.from_pretrained(model_dir, file_name=FILENAME_OPTIMIZED_ONNX)


def _get_quantization_config():
    cpu_info = cpuinfo.get_cpu_info()
    if 'arch' in cpu_info and cpu_info['arch'] == 'ARM_8':
        return AutoQuantizationConfig.arm64
    if 'flags' in cpu_info:
        flags = cpu_info['flags']
        if 'AVX512_VNNI' in flags:
            return AutoQuantizationConfig.avx512_vnni
        elif 'AVX512F' in flags:
            return AutoQuantizationConfig.avx512
        elif 'AVX2' in flags:
            return AutoQuantizationConfig.avx2
    return None


@timer
def _quantize(repository_id: str, model_dir, task, push_to_hub=False):
    model_class = _task_to_model(task)

    # try to load from hub or cache
    pre_optimized_model = _try_load_from_hub(repository_id, FILENAME_QUANTIZED_ONNX, model_class)
    if pre_optimized_model:
        return pre_optimized_model

    # try to load existing local file
    model_file = model_dir + "/" + FILENAME_QUANTIZED_ONNX
    if os.path.exists(model_file):
        return model_class.from_pretrained(model_dir, file_name=FILENAME_QUANTIZED_ONNX)

    # quantize
    model = _export_to_onnx(repository_id, model_dir, task)
    quantizer = ORTQuantizer.from_pretrained(model)
    config = _get_quantization_config()(is_static=False, per_channel=True)
    quantizer.quantize(
        quantization_config=config,
        save_dir=model_dir
    )
    size = os.path.getsize(model_file) / (1024 * 1024)
    logger.debug(f"Quantized Onnx Model file size: {size:.2f} MB")
    return model_class.from_pretrained(model_dir, file_name=FILENAME_QUANTIZED_ONNX)