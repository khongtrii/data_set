import argparse
import cv2
from inference.models.utils import get_roboflow_model
from ultralytics import YOLO
import numpy as np
import supervision as sv

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolov8x.pt")
    # model = get_roboflow_model("yolov8x-640")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    test_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=test_scale, text_thickness=thickness)
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, video_info.resolution_wh)

    for frame in frame_generator:
        # result = model.infer(frame)[0]
        # detections = sv.Detections.from_inference(result)
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger]
        detections = byte_track.update_with_detections(detections=detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]


        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.red())
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        annotated_frame = cv2.resize(annotated_frame, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()