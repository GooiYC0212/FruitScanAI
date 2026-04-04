# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
"""

import argparse
import csv
import os
import platform
import sys
from collections import Counter
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# =========================================================
# PRICE LIST
# 只会对这里面的物体进行计价
# 你可以改成你自己的商品和价格
# =========================================================
PRICE_LIST = {
    "bottle": 3.50,
    "cup": 2.00,
    "banana": 1.50,
    # "apple": 2.00,
    # "orange": 3.00,
}


def calculate_bill(detected_items):
    """
    根据检测到的物体列表，计算数量、小计和总价。
    只统计 PRICE_LIST 中存在的物体。
    """
    filtered_items = [item for item in detected_items if item in PRICE_LIST]
    item_counts = Counter(filtered_items)

    bill_rows = []
    total_price = 0.0

    for item, qty in item_counts.items():
        unit_price = PRICE_LIST[item]
        subtotal = unit_price * qty
        total_price += subtotal
        bill_rows.append(
            {
                "item": item,
                "qty": qty,
                "unit_price": unit_price,
                "subtotal": subtotal,
            }
        )

    return item_counts, bill_rows, total_price


def draw_bill_on_image(im0, bill_rows, total_price):
    """
    在画面左上角显示账单信息。
    """
    x = 20
    y = 30
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    if bill_rows:
        cv2.putText(im0, "Detected Items:", (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += line_height

        for row in bill_rows:
            text = f"{row['item']} x{row['qty']} = RM{row['subtotal']:.2f}"
            cv2.putText(im0, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            y += line_height

        total_text = f"Total = RM{total_price:.2f}"
        cv2.putText(im0, total_text, (x, y), font, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(
            im0,
            "No billable items detected",
            (x, y),
            font,
            font_scale,
            (0, 165, 255),
            thickness,
            cv2.LINE_AA,
        )


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source="0",  # 默认 webcam
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=True,  # 默认显示画面
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")

    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize_path).unsqueeze(0)
                    else:
                        pred = torch.cat(
                            (pred, model(image, augment=augment, visualize=visualize_path).unsqueeze(0)), dim=0
                        )
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize_path)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence, quantity="", subtotal="", total=""):
            data = {
                "Image Name": image_name,
                "Prediction": prediction,
                "Confidence": confidence,
                "Quantity": quantity,
                "Subtotal": subtotal,
                "Total": total,
            }
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1

            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            detected_items = []

            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print detection summary
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write detection results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    item_name = names[c]
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    detected_items.append(item_name)

                    if save_csv:
                        write_to_csv(p.name, item_name, confidence_str)

                    if save_txt:
                        if save_format == 0:
                            coords = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:
                        display_label = None if hide_labels else (
                            item_name if hide_conf else f"{item_name} {conf:.2f}"
                        )
                        annotator.box_label(xyxy, display_label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / item_name / f"{p.stem}.jpg", BGR=True)

            # Bill calculation
            item_counts, bill_rows, total_price = calculate_bill(detected_items)

            LOGGER.info("-" * 50)
            LOGGER.info(f"Bill Summary for: {p.name}")
            if bill_rows:
                for row in bill_rows:
                    LOGGER.info(
                        f"{row['item']} x{row['qty']} @ RM{row['unit_price']:.2f} = RM{row['subtotal']:.2f}"
                    )
                LOGGER.info(f"TOTAL = RM{total_price:.2f}")
            else:
                LOGGER.info("No billable items detected.")
            LOGGER.info("-" * 50)

            if save_csv and bill_rows:
                for row in bill_rows:
                    write_to_csv(
                        p.name,
                        row["item"],
                        "",
                        quantity=row["qty"],
                        subtotal=f"{row['subtotal']:.2f}",
                        total=f"{total_price:.2f}",
                    )

            # 先取得已经画好 bbox 的结果图
            im0 = annotator.result()

            # 再把账单和提示画上去
            draw_bill_on_image(im0, bill_rows, total_price)
            cv2.putText(
                im0,
                "Press Q to exit",
                (10, im0.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Show result
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2.imshow(str(p), im0)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Exit webcam by pressing Q")
                    cv2.destroyAllWindows()
                    return

            # Save result
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()

                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Final results
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

    if save_txt or save_img or save_csv:
        extra = []
        if save_txt:
            extra.append(f"{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}")
        if save_csv:
            extra.append(f"CSV saved to {csv_path}")
        extra_text = "\n" + "\n".join(extra) if extra else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{extra_text}")

    if update:
        strip_optimizer(weights[0] if isinstance(weights, list) else weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)