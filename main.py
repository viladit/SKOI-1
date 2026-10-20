import argparse
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_LOWER = (0, 140, 0)
DEFAULT_UPPER = (135, 255, 145)
SIZES = ((320, 240), (640, 480), (1280, 960))
LOWER = (0, 150, 0)
UPPER = (100, 255, 100)
SEED = 368925
REPEATS = 10
WARMUPS = 2
TARGET_BATCH_SECONDS = 0.01


def validate_inputs(foreground, background, lower, upper):
    for image in (foreground, background):
        if not isinstance(image, np.ndarray):
            raise TypeError("Изображения должны быть массивами NumPy")
        if image.dtype != np.uint8:
            raise TypeError("Изображения должны иметь тип uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Изображения должны содержать три канала RGB")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Изображения не должны быть пустыми")
    if foreground.shape != background.shape:
        raise ValueError("Размеры изображений должны совпадать")

    bounds = []
    for bound in (lower, upper):
        try:
            values = tuple(bound)
        except TypeError:
            raise TypeError("Граница должна содержать три целых числа") from None
        if len(values) != 3:
            raise ValueError("Граница должна содержать три значения RGB")
        if any(
            not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
            for value in values
        ):
            raise TypeError("Значения RGB должны быть целыми числами")
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("Значения RGB должны лежать от 0 до 255")
        bounds.append(tuple(int(value) for value in values))
    if any(low > high for low, high in zip(bounds[0], bounds[1])):
        raise ValueError("Нижняя граница не должна превышать верхнюю")
    return bounds[0], bounds[1]


def chroma_key_opencv(foreground, background, lower, upper):
    lower, upper = validate_inputs(foreground, background, lower, upper)
    mask = cv2.inRange(foreground, lower, upper)
    remaining = cv2.bitwise_and(foreground, foreground, mask=cv2.bitwise_not(mask))
    replacement = cv2.bitwise_and(background, background, mask=mask)
    return cv2.bitwise_or(remaining, replacement)


def chroma_key_python(foreground, background, lower, upper):
    lower, upper = validate_inputs(foreground, background, lower, upper)
    result = np.empty_like(foreground)
    height, width, _ = foreground.shape
    for row in range(height):
        for column in range(width):
            red = foreground[row, column, 0]
            green = foreground[row, column, 1]
            blue = foreground[row, column, 2]
            if (
                lower[0] <= red <= upper[0]
                and lower[1] <= green <= upper[1]
                and lower[2] <= blue <= upper[2]
            ):
                source = background
            else:
                source = foreground
            for channel in range(3):
                result[row, column, channel] = source[row, column, channel]
    return result


def load_rgb(path):
    path = Path(path)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Не удалось прочитать изображение {path}") from error
    if not content:
        raise ValueError(f"Файл изображения пуст {path}")
    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Не удалось распознать изображение {path}") from error
    if image is None:
        raise ValueError(f"Не удалось распознать изображение {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    path = Path(path)
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Для сохранения нужен непустой массив RGB uint8")
    if not path.suffix:
        raise ValueError("У выходного файла должно быть расширение изображения")
    try:
        success, encoded = cv2.imencode(
            path.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
    except cv2.error as error:
        raise ValueError(f"Не поддерживается формат результата {path.suffix}") from error
    if not success:
        raise ValueError("Не удалось закодировать результат")
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".chroma-", suffix=path.suffix, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(encoded.tobytes())
        os.replace(temporary, path)
    except OSError as error:
        raise ValueError(f"Не удалось сохранить изображение {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_inputs(width, height, seed=SEED):
    rng = np.random.default_rng(seed)
    foreground = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    background = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return foreground, background


def measure_batch(function, foreground, background, batch):
    started = time.perf_counter()
    for _ in range(batch):
        function(foreground, background, LOWER, UPPER)
    return (time.perf_counter() - started) * 1000 / batch


def summarize(samples, batch):
    return {
        "batch": batch,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_benchmark(sizes=SIZES, repeats=REPEATS, warmups=WARMUPS):
    if repeats < 1 or warmups < 1:
        raise ValueError("Число повторений и прогревов должно быть положительным")
    measurements = []
    for width, height in sizes:
        foreground, background = make_inputs(width, height)
        expected = chroma_key_python(foreground, background, LOWER, UPPER)
        actual = chroma_key_opencv(foreground, background, LOWER, UPPER)
        if not np.array_equal(expected, actual):
            raise ValueError("Результаты двух реализаций не совпали")
        functions = {
            "opencv": chroma_key_opencv,
            "python": chroma_key_python,
        }
        for _ in range(warmups):
            for function in functions.values():
                function(foreground, background, LOWER, UPPER)
        calibration_ms = measure_batch(
            chroma_key_opencv, foreground, background, 1
        )
        opencv_batch = min(
            1000,
            max(1, math.ceil(TARGET_BATCH_SECONDS * 1000 / max(calibration_ms, 1e-9))),
        )
        batches = {"opencv": opencv_batch, "python": 1}
        samples = {"opencv": [], "python": []}
        for repeat in range(repeats):
            order = ("opencv", "python") if repeat % 2 == 0 else ("python", "opencv")
            for name in order:
                samples[name].append(
                    measure_batch(
                        functions[name], foreground, background, batches[name]
                    )
                )
        opencv = summarize(samples["opencv"], batches["opencv"])
        native = summarize(samples["python"], batches["python"])
        measurements.append(
            {
                "width": width,
                "height": height,
                "pixels": width * height,
                "opencv": opencv,
                "python": native,
                "speedup": native["median_ms"] / opencv["median_ms"],
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        },
        "configuration": {
            "repeats": repeats,
            "warmups": warmups,
            "seed": SEED,
            "lower_rgb": list(LOWER),
            "upper_rgb": list(UPPER),
            "target_batch_ms": TARGET_BATCH_SECONDS * 1000,
            "alternating_order": True,
            "timed_scope": "Полный вызов ядра с проверками и выделением массивов",
            "excluded_scope": "Подготовка данных, ввод и вывод, проверка равенства",
        },
        "measurements": measurements,
    }


def plot_results(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError("Для отчёта установите зависимости из requirements.txt") from error

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    foreground = load_rgb(ROOT / "data" / "foreground.png")
    background = load_rgb(ROOT / "data" / "background.png")
    cases = [
        ("Узкий интервал", (0, 170, 0), (90, 255, 100), "narrow.png"),
        ("Выбранный интервал", DEFAULT_LOWER, DEFAULT_UPPER, None),
        ("Широкий интервал", (0, 80, 0), (200, 255, 200), "wide.png"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10, 6), layout="constrained")
    for axis, (label, lower, upper, filename) in zip(axes, cases):
        result = chroma_key_opencv(foreground, background, lower, upper)
        if not np.array_equal(result, chroma_key_python(foreground, background, lower, upper)):
            raise ValueError("Результаты реализаций не совпадают")
        if filename:
            save_rgb(results / filename, result)
        mask = np.all((foreground >= lower) & (foreground <= upper), axis=2)
        print(f"{label}. Заменено {np.count_nonzero(mask)} из {mask.size} пикселей")
        axis.imshow(result)
        axis.set_title(f"{label}\n{lower}\n{upper}", fontsize=10)
        axis.set_axis_off()
    figure.savefig(results / "ranges.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    measurements = report["measurements"]
    positions = np.arange(len(measurements))
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for method, label, offset, color in [
        ("opencv", "OpenCV", -0.19, "#187ca3"),
        ("python", "Python", 0.19, "#dc8a36"),
    ]:
        medians = np.array([row[method]["median_ms"] for row in measurements])
        minimum = np.array([row[method]["min_ms"] for row in measurements])
        maximum = np.array([row[method]["max_ms"] for row in measurements])
        bars = axis.bar(
            positions + offset, medians, width=0.36, label=label, color=color,
            yerr=np.vstack((medians - minimum, maximum - medians)), capsize=4,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in medians], padding=8)
    axis.set_yscale("log")
    axis.set_ylim(top=max(row["python"]["max_ms"] for row in measurements) * 5)
    axis.set_ylabel("Время, мс. Логарифмическая шкала")
    axis.set_xticks(positions, [f"{row['width']} x {row['height']}" for row in measurements])
    axis.set_xlabel("Размер изображения")
    axis.set_title(f"Медиана {report['configuration']['repeats']} замеров. Усы показывают минимум и максимум")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.savefig(results / "benchmark.png", dpi=170)
    plt.close(figure)
    print(f"Иллюстрации сохранены в {results}")


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"Ошибка. {message}\n")


def build_parser():
    parser = ArgumentParser(description="Замена диапазона RGB новым фоном")
    parser.add_argument("--foreground", type=Path, default=ROOT / "data" / "foreground.png", help="исходное изображение")
    parser.add_argument("--background", type=Path, default=ROOT / "data" / "background.png", help="новый фон")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "result.png", help="файл результата")
    parser.add_argument("--lower", nargs=3, type=int, metavar=("R", "G", "B"), default=DEFAULT_LOWER, help="нижняя включённая граница RGB")
    parser.add_argument("--upper", nargs=3, type=int, metavar=("R", "G", "B"), default=DEFAULT_UPPER, help="верхняя включённая граница RGB")
    parser.add_argument("--method", choices=("opencv", "python", "both"), default="both", help="реализация алгоритма")
    parser.add_argument("--report", action="store_true", help="выполнить замеры и построить графики")
    return parser


def check_output_path(output, inputs):
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("Результат нельзя сохранять поверх входного изображения")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        check_output_path(args.output, (args.foreground, args.background))
        if args.report:
            demo_inputs = (ROOT / "data" / "foreground.png", ROOT / "data" / "background.png")
            check_output_path(args.output, demo_inputs)
            protected = (
                args.foreground, args.background, args.output,
                *demo_inputs,
            )
            for filename in ("benchmark.json", "benchmark.png", "ranges.png", "narrow.png", "wide.png"):
                check_output_path(ROOT / "results" / filename, protected)
        foreground = load_rgb(args.foreground)
        background = load_rgb(args.background)
        function = chroma_key_python if args.method == "python" else chroma_key_opencv
        result = function(foreground, background, args.lower, args.upper)
        if args.method == "both":
            native = chroma_key_python(foreground, background, args.lower, args.upper)
            if not np.array_equal(result, native):
                raise ValueError("Результаты двух реализаций не совпали")
            print("Результаты OpenCV и Python совпадают для каждого канала")
        save_rgb(args.output, result)
        print(f"Результат сохранён в {args.output.resolve()}")
        print(f"Размер {result.shape[1]} x {result.shape[0]}")
        if args.report:
            report = run_benchmark()
            destination = ROOT / "results" / "benchmark.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("Размер        OpenCV, мс    Python, мс    Ускорение")
            for row in report["measurements"]:
                size = f"{row['width']} x {row['height']}"
                print(f"{size:<13} {row['opencv']['median_ms']:>10.3f} {row['python']['median_ms']:>13.3f} {row['speedup']:>11.1f}")
            print(f"Замеры сохранены в {destination}")
            plot_results(report)
    except (ValueError, TypeError, OSError, cv2.error) as error:
        print(f"Ошибка. {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
