import csv
import cv2
import mediapipe as mp

# Compatibilidad con la versión de mediapipe instalada en el entorno
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('IMG_5796.MOV')

if not cap.isOpened():
    raise RuntimeError('No se pudo abrir el archivo de video IMG_5796.MOV')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
output_path = 'analisis_caleta_output.mp4'
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_count = 0
frames_con_pose = 0
landmarks_seen = 0
results_rows = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frames_con_pose += 1
        landmarks_seen += len(results.pose_landmarks.landmark)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        row = {
            'frame': frame_count,
            'landmarks': len(results.pose_landmarks.landmark),
            'nose_x': results.pose_landmarks.landmark[0].x if len(results.pose_landmarks.landmark) > 0 else None,
            'nose_y': results.pose_landmarks.landmark[0].y if len(results.pose_landmarks.landmark) > 0 else None,
            'left_shoulder_x': results.pose_landmarks.landmark[11].x if len(results.pose_landmarks.landmark) > 11 else None,
            'left_shoulder_y': results.pose_landmarks.landmark[11].y if len(results.pose_landmarks.landmark) > 11 else None,
            'right_shoulder_x': results.pose_landmarks.landmark[12].x if len(results.pose_landmarks.landmark) > 12 else None,
            'right_shoulder_y': results.pose_landmarks.landmark[12].y if len(results.pose_landmarks.landmark) > 12 else None,
        }
        results_rows.append(row)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'Procesadas {frame_count} frames')
print(f'Frames con pose detectada: {frames_con_pose}')
print(f'Landmarks detectados en total: {landmarks_seen}')
if frames_con_pose > 0:
    print(f'Promedio de landmarks por frame con pose: {landmarks_seen / frames_con_pose:.2f}')
print(f'Video guardado en {output_path}')

csv_path = 'analisis_caleta_datos.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['frame', 'landmarks', 'nose_x', 'nose_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y'])
    writer.writeheader()
    writer.writerows(results_rows)

print(f'Datos guardados en {csv_path}')