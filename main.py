import datetime
import math
import torch
import numpy as np
import mediapipe as mp
import cv2
from time import time
import concurrent.futures
import collections
from concurrent.futures import wait


class MediaPipeDemo:

    def __init__(self, desired_width: int, desired_height: int,
                 image_files: list, bg_color: tuple = (192, 192, 192)):
        self.bg_color = bg_color
        self.image_files = image_files
        self.desired_width = desired_width
        self.desired_height = desired_height

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.landmarks_by_id = {}

    def moving_images(self, input_video_file_name: str, output_video_file_name: str):
        """Обработка видео."""
        cap = cv2.VideoCapture(input_video_file_name)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        writer = cv2.VideoWriter(output_video_file_name, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
        # time measurement
        time1 = 0
        frames = 0
        frames_per_second_sum = 0
        start_time = time()
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while True:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                time2 = time()
                # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
                if time2 - time1 > 0:
                    # Calculate the number of frames per second.
                    frames_per_second = 1.0 / (time2 - time1)
                    frames_per_second_sum += frames_per_second
                    frames += 1
                # Update the previous frame time to this frame time.
                # As this frame will become previous frame in next iteration.
                time1 = time2

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print(results.pose_landmarks)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                if success:
                    writer.write(image)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

        print('\nAverage fps: {}'.format(np.round(frames_per_second_sum / frames), 3))
        print('Ms per frame: {}'.format(np.round(1000 * (time() - start_time) / frames, 3)))

    @staticmethod
    def get_cropped_landmarks(original_image, tl_crop: tuple, br_crop: tuple, pose):
        crop_image = original_image[tl_crop[1]:br_crop[1], tl_crop[0]:br_crop[0], :]
        crop_image.flags.writeable = True
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
        results = pose.process(crop_image)
        return results.pose_landmarks

    @staticmethod
    def landmark_translation(landmarks, tl_crop: tuple, br_crop: tuple,
                             width: int, height: int):
        """Приведение к одной системе координат."""
        for index, landmark in enumerate(landmarks.landmark):
            landmarks.landmark[index].x = float(landmark.x * (br_crop[0] - tl_crop[0] + 1) + tl_crop[0]) / width
            landmarks.landmark[index].y = float(landmark.y * (br_crop[1] - tl_crop[1] + 1) + tl_crop[1]) / height
        return landmarks

    @staticmethod
    def calculate_angle(landmark1: tuple, landmark2: tuple, landmark3: tuple):
        """Вычисляет угол между тремя landmark'ами."""
        x1, y1 = landmark1
        x2, y2 = landmark2
        x3, y3 = landmark3
        # Вычисление угла между тремя точками
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # Проверка на то, что угол меньше нуля
        if angle < 0:
            angle += 360
        return angle

    @staticmethod
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def pose_classifier(self, landmarks, mp_pose, person_id):
        """Классификация поз человека (в частности, - что он наклонился)."""
        # Получаем углы между плечом, бедром и коленом
        left_hip_angle = self.calculate_angle(landmark1=(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                                              landmark2=(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                                              landmark3=(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                                              )
        right_hip_angle = self.calculate_angle(landmark1=(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                                               landmark2=(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                                               landmark3=(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y))

        # if len(self.landmarks_by_id[person_id]) == 10:
        #     previous_right_norm = cv2.norm(
        #         (self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        #          self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        #          self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z),
        #         (self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        #          self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        #          self.landmarks_by_id[person_id][0][mp_pose.PoseLandmark.RIGHT_HIP.value].z)
        #     )
        #     next_right_norm = cv2.norm(
        #         (self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        #          self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
        #          self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z),
        #         (self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        #          self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        #          self.landmarks_by_id[person_id][1][mp_pose.PoseLandmark.RIGHT_HIP.value].z)
        #     )
        #     dif_right_norm = math.fabs(previous_right_norm - next_right_norm)
        #     print(f'previous_right_norm: {previous_right_norm}\tdif_right_norm: {dif_right_norm}')
        #     # print(next_right_norm)
        #     # Проверка на то, что человек наклонился
        #     if (left_hip_angle < 165 or right_hip_angle < 165) and (dif_right_norm < 0.01 * previous_right_norm):
        #         return True
        #     else:
        #         return False
        # else:
        #     return False

    def parallel_run(self, result, original_image, person_id: int):
        """Основная логика."""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        bbox = [int(result[0]), int(result[1]), int(result[2]), int(result[3])]
        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            landmarks = self.get_cropped_landmarks(original_image=original_image, tl_crop=(bbox[0], bbox[1]),
                                                   br_crop=(bbox[2], bbox[3]), pose=pose)
            if landmarks:
                translated_landmarks = self.landmark_translation(landmarks=landmarks,
                                                                 tl_crop=(bbox[0], bbox[1]),
                                                                 br_crop=(bbox[2], bbox[3]),
                                                                 width=original_image.shape[1],
                                                                 height=original_image.shape[0])
                mp_drawing.draw_landmarks(
                    original_image, translated_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                self.landmarks_by_id[person_id].append(translated_landmarks.landmark)
            if self.pose_classifier(landmarks=translated_landmarks.landmark, mp_pose=mp_pose, person_id=person_id):
                cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(original_image, str(person_id), (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(original_image, str(person_id), (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    def webcam_detection(self):
        """Детекция с веб-камеры."""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        cap = cv2.VideoCapture(0)
        time_1, frames, frames_per_second_sum, frames_per_second, start_time = 0, 0, 0, 0, time()  # fps
        self.landmarks_by_id[0] = collections.deque(maxlen=2)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            # fps
            time_2 = time()
            if time_2 - time_1 > 0:
                frames_per_second = 1.0 / (time_2 - time_1)
                frames_per_second_sum += frames_per_second
                frames += 1
            time_1 = time_2

            yolo_results = model(image)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for result in yolo_results.xyxy[0]:
                    if int(result[5]) == 0 and result[4] > 0.5:
                        futures.append(executor.submit(self.parallel_run, result=result,
                                                       original_image=image, person_id=0))
                    else:
                        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                for future in concurrent.futures.as_completed(futures):
                    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

                # print('fps: {}'.format(np.round(frames_per_second, 2)))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

        print('\nAverage fps: {}'.format(np.round(frames_per_second_sum / frames), 3))
        print('Ms per frame: {}'.format(np.round(1000 * (time() - start_time) / frames, 3)))


if __name__ == '__main__':
    mediapipe_demo = MediaPipeDemo(desired_width=480, desired_height=480,
                                   image_files=['video1.avi'], bg_color=(192, 192, 192))
    # mediapipe_demo.moving_images(input_video_file_name='video2.mp4', output_video_file_name='demo2.avi')
    mediapipe_demo.webcam_detection()
