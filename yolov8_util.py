import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import os
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from bytetrack.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT

# Fallback palette to support older supervision versions without ColorPalette.VIVID
try:
    VIVID_PALETTE = sv.ColorPalette.VIVID
except AttributeError:
    VIVID_PALETTE = sv.ColorPalette(
        colors=[
            sv.Color.from_hex("#FF6B6B"),
            sv.Color.from_hex("#FFD166"),
            sv.Color.from_hex("#06D6A0"),
            sv.Color.from_hex("#118AB2"),
            sv.Color.from_hex("#9B5DE5"),
            sv.Color.from_hex("#EF476F"),
            sv.Color.from_hex("#F78C6B"),
            sv.Color.from_hex("#073B4C"),
            sv.Color.from_hex("#4ECDC4"),
            sv.Color.from_hex("#FFC300"),
        ]
    )

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r', encoding='utf-8') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)



class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        self.config_main = "./config.yml"
        self.main_cfg = get_config()
        self.main_cfg.merge_from_file(self.config_main)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names

        # Video save options
        self.save_video = bool(getattr(self.main_cfg.demo_test, "save_video", False))
        self.save_video_path = getattr(self.main_cfg.demo_test, "save_video_path", "./demo_out.avi")
        self.save_video_fps = float(getattr(self.main_cfg.demo_test, "save_video_fps", 0))  # 0 means auto from source
        self.save_video_codec = str(getattr(self.main_cfg.demo_test, "save_video_codec", "XVID"))

        # Optional downscale or ROI crop to handle oversized frames
        self.resize_ratio = float(getattr(self.main_cfg.demo_test, "resize_ratio", 1.0))
        self.display_ratio = float(getattr(self.main_cfg.demo_test, "display_ratio", 1.0))
        roi_cfg = getattr(self.main_cfg.demo_test, "roi", None)
        if roi_cfg is not None and all(hasattr(roi_cfg, k) for k in ("x", "y", "w", "h")):
            self.roi = (int(roi_cfg.x), int(roi_cfg.y), int(roi_cfg.w), int(roi_cfg.h))
        else:
            self.roi = None

        # Single-person tracking: set target track id in config.yml (demo_test.target_track_id)
        target_id_cfg = getattr(self.main_cfg.demo_test, "target_track_id", None)
        self.target_track_id = int(target_id_cfg) if target_id_cfg not in (None, "", "null") else None

        # Mouse-based selection state
        self.last_outputs = []  # store latest tracker outputs for hit-test
        self.click_point = None

        self.box_annotator = sv.BoxAnnotator(color=VIVID_PALETTE, thickness=3)

        label_annotator_cls = getattr(sv, "LabelAnnotator", None)
        self.text_annotator = label_annotator_cls() if label_annotator_cls else None
        reid_weights   = Path(self.main_cfg.demo_test.reid_model_path)
          
        if self.main_cfg.demo_test.tracker == "bytetrack":
            tracker_config = "bytetrack/configs/bytetrack.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
     
            self.tracker = BYTETracker(
                track_thresh=cfg.bytetrack.track_thresh,
                match_thresh=cfg.bytetrack.match_thresh,
                track_buffer=cfg.bytetrack.track_buffer,
                frame_rate=cfg.bytetrack.frame_rate
            )
        else :
            tracker_config = "strongsort/configs/strongsort.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
    
            self.tracker = StrongSORT (
                reid_weights,
                torch.device("cpu"),
                False,
                max_dist=cfg.strongsort.max_dist,
                max_iou_dist=cfg.strongsort.max_iou_dist,
                max_age=cfg.strongsort.max_age,
                max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
                n_init=cfg.strongsort.n_init,
                nn_budget=cfg.strongsort.nn_budget,
                mc_lambda=cfg.strongsort.mc_lambda,
                ema_alpha=cfg.strongsort.ema_alpha,
            )


    def load_model(self):
       
        model = YOLO(self.main_cfg.demo_test.detection_model_path)  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def draw_results(self, frame, tracked_outputs):
        # tracked_outputs: list of arrays [x1, y1, x2, y2, track_id, cls, conf]
        if tracked_outputs is None or len(tracked_outputs) == 0:
            return frame, []

        if self.target_track_id is not None:
            tracked_outputs = [o for o in tracked_outputs if int(o[4]) == self.target_track_id]
            if len(tracked_outputs) == 0:
                return frame, []

        xyxy = np.array([o[0:4] for o in tracked_outputs])
        confs = np.array([o[6] if len(o) > 6 else 1.0 for o in tracked_outputs])
        cls_ids = np.array([int(o[5]) if len(o) > 5 else 0 for o in tracked_outputs])

        detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls_ids)

        labels = [
            f"{self.CLASS_NAMES_DICT[cid]} {conf:0.2f}"
            for conf, cid in zip(detections.confidence, detections.class_id)
        ]

        if self.text_annotator:
            annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = self.text_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        else:
            annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return annotated_frame, detections
       
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        # Allow lowering capture resolution if specified in config
        target_w = getattr(self.main_cfg.demo_test, "input_width", None)
        target_h = getattr(self.main_cfg.demo_test, "input_height", None)
        if target_w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(target_w))
        if target_h:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target_h))

        outputvid = None
        # setup tracker
        tracker = self.tracker

        # Mouse callback for manual selection: left-click to select目标，右键清除
        cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('YOLOv8 Detection', self.on_mouse)

        # if tracker is using model then warmup
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = []
        curr_frames, prev_frames = None, None

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            # Apply ROI crop first if configured
            if self.roi:
                x, y, w, h = self.roi
                frame = frame[y:y + h, x:x + w]

            # Downscale for display/speed if requested
            if self.resize_ratio != 1.0:
                frame = cv2.resize(frame, dsize=None, fx=self.resize_ratio, fy=self.resize_ratio, interpolation=cv2.INTER_AREA)
            results = self.predict(frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            # Update tracker and collect tracked outputs
            outputs = []
            for result in results:
                tracked = tracker.update(result, frame)
                if tracked is not None:
                    outputs.extend(tracked)

            # Save for mouse hit-test
            self.last_outputs = outputs.copy()

            # Annotate only the target track (if configured)
            frame, _ = self.draw_results(frame, outputs)

            # Lazy-init video writer with annotated frame size
            if self.save_video and outputvid is None:
                h, w = frame.shape[:2]
                codec = cv2.VideoWriter_fourcc(*self.save_video_codec)
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                video_fps = self.save_video_fps if self.save_video_fps > 0 else (src_fps if src_fps > 0 else 20)
                outputvid = cv2.VideoWriter(self.save_video_path, codec, video_fps, (w, h))
            
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:  # camera motion compensation
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            # Overlay ID text for kept tracks
            for output in outputs:
                bbox = output[0:4]
                tracked_id = output[4]
                if self.target_track_id is not None and int(tracked_id) != self.target_track_id:
                    continue
                top_left = (
                    int(bbox[-2]-100),
                    int(bbox[1])
                )
                cv2.putText(
                    frame,
                    f"ID : {tracked_id}",
                    top_left,
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255,0), 
                    3
                )

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            # Shrink display window if requested
            if self.display_ratio != 1.0:
                disp = cv2.resize(frame, dsize=None, fx=self.display_ratio, fy=self.display_ratio, interpolation=cv2.INTER_AREA)
            else:
                disp = frame
            cv2.imshow('YOLOv8 Detection', disp)
            if outputvid is not None:
                outputvid.write(frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        if outputvid is not None:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, param):
        # Left click: select the track whose bbox contains the point
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.last_outputs:
                return
            # Map click coords back if display has been downscaled
            if self.display_ratio != 1.0:
                x = int(x / self.display_ratio)
                y = int(y / self.display_ratio)
            for o in self.last_outputs:
                x1, y1, x2, y2 = o[0:4]
                tid = int(o[4]) if len(o) > 4 else None
                if tid is None:
                    continue
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target_track_id = tid
                    print(f"Selected track id: {tid}")
                    break
        # Right click: clear selection
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.target_track_id = None
            print("Cleared target selection")
        
        


detector = ObjectDetection(capture_index="E:\yolov8-reid-track\\1.mp4")
detector()
