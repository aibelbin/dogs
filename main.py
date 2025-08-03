#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime


class DogDetector:
    def __init__(self, confidence_threshold=0.3, device='cpu', skip_frames=2):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.class_names = []
        self.dog_class_id = None
        self.show_confidence = True
        self.frame_count = 0
        self.fps_counter = 0
        self.last_time = time.time()
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_detections = []
        self.detection_history = []
        self.alert_cooldown = 0
        self.last_alert_time = 0
        
        self.colors = {
            'dog': (0, 255, 0),
            'text': (255, 255, 255),
            'bg': (0, 0, 0),
            'alert': (0, 0, 255)
        }
        
        self.load_model()
    
    def load_model(self):
        try:
            print("Loading optimized model for Raspberry Pi...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.to(self.device)
            
            if self.device == 'cpu':
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.class_names = self.model.names
            
            for class_id, class_name in self.class_names.items():
                if class_name.lower() == 'dog':
                    self.dog_class_id = class_id
                    break
            
            if self.dog_class_id is None:
                raise ValueError("Dog class not found in model classes")
                
            print(f"Model loaded successfully. Dog class ID: {self.dog_class_id}")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_dogs(self, frame):
        self.frame_counter += 1
        
        if self.frame_counter % self.skip_frames != 0 and len(self.last_detections) > 0:
            return self.last_detections
        
        small_frame = cv2.resize(frame, (416, 416))
        results = self.model(small_frame)
        
        detections = results.pandas().xyxy[0]
        
        if len(detections) > 0:
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            detections['xmin'] *= scale_x
            detections['xmax'] *= scale_x
            detections['ymin'] *= scale_y
            detections['ymax'] *= scale_y
        
        all_detections = detections[detections['confidence'] >= self.confidence_threshold]
        
        dog_detections = all_detections[all_detections['class'] == self.dog_class_id]
        
        self.track_detections(dog_detections)
        
        self.last_detections = dog_detections
        return dog_detections
    
    def track_detections(self, detections):
        current_time = time.time()
        
        if len(detections) > 0:
            self.detection_history.append({
                'time': current_time,
                'count': len(detections),
                'max_confidence': detections['confidence'].max() if len(detections) > 0 else 0
            })
            
            if (current_time - self.last_alert_time) > 5.0:
                self.save_detection_alert()
                self.last_alert_time = current_time
        
        self.detection_history = [d for d in self.detection_history if current_time - d['time'] < 30.0]
    
    def save_detection_alert(self):
        if not os.path.exists('alerts'):
            os.makedirs('alerts')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = f"alerts/dog_alert_{timestamp}.txt"
        
        with open(alert_file, 'w') as f:
            f.write(f"DOG DETECTION ALERT\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Detections in last 30s: {len(self.detection_history)}\n")
            if self.detection_history:
                max_conf = max(d['max_confidence'] for d in self.detection_history)
                f.write(f"Max confidence: {max_conf:.3f}\n")
        
        print(f"🚨 DOG DETECTED! Alert saved: {alert_file}")
    
    def draw_detections(self, frame, detections):
        if len(detections) == 0:
            return frame
        
        current_time = time.time()
        is_recent_detection = (current_time - self.last_alert_time) < 10.0
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                            int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            
            color = self.colors['alert'] if is_recent_detection else self.colors['dog']
            thickness = 3 if is_recent_detection else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            if self.show_confidence:
                label = f"Dog {confidence:.2f}"
            else:
                label = "Dog"
            
            if is_recent_detection:
                label = f"🚨 {label}"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
            
            bg_color = self.colors['alert'] if is_recent_detection else self.colors['bg']
            cv2.rectangle(frame, 
                         (text_x, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), 
                         bg_color, -1)
            
            cv2.putText(frame, label, (text_x + 2, text_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return frame
    
    def draw_fps(self, frame):
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        else:
            self.frame_count += 1
        
        fps_text = f"FPS: {self.fps_counter}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        return frame
    
    def draw_instructions(self, frame, dog_count):
        height, width = frame.shape[:2]
        
        status_text = f"Dogs detected: {dog_count}"
        cv2.putText(frame, status_text, (10, height - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        history_text = f"Detections (30s): {len(self.detection_history)}"
        cv2.putText(frame, history_text, (10, height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        if self.detection_history:
            last_detection = time.time() - max(d['time'] for d in self.detection_history)
            time_text = f"Last seen: {last_detection:.1f}s ago"
            cv2.putText(frame, time_text, (10, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        instructions = [
            "Controls:",
            "Press 'q' to quit",
            "Press 's' to save screenshot",
            "Press 'c' to toggle confidence"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 50 + (i * 15)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return frame
    
    def save_screenshot(self, frame):
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/dog_detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def run(self, camera_index=0):
        print(f"Initializing camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)
        
        print("Dog detection started. Press 'q' to quit.")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                detections = self.detect_dogs(frame)
                dog_count = len(detections)
                
                frame = self.draw_detections(frame, detections)
                
                frame = self.draw_fps(frame)
                
                frame = self.draw_instructions(frame, dog_count)
                
                cv2.imshow('Dog Detection - Live Stream', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame)
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Resources released successfully")


def main():
    print("Hot Dog Detection System")
    print("=" * 30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        detector = DogDetector(
            confidence_threshold=0.25,
            device=device,
            skip_frames=1
        )
        
        detector.run(camera_index=0)
        
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Make sure you have a webcam connected and the required packages installed.")


if __name__ == "__main__":
    main()
