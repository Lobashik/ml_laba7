import cv2
import os
import urllib.request


def download_if_not_exists(filename, url):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)


yunet_path = "face_detection_yunet_2023mar.onnx"
download_if_not_exists(
    yunet_path,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

sface_path = "face_recognition_sface_2021dec.onnx"
download_if_not_exists(
    sface_path,
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

image_path = "./image.png"

reference_img = cv2.imread(image_path)
if reference_img is None:
    print(f"Ошибка: {image_path} не найден!")
    exit()

detector = cv2.FaceDetectorYN.create(
    yunet_path,
    "",
    (320, 320),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

h, w, _ = reference_img.shape
detector.setInputSize((w, h))
_, ref_faces = detector.detect(reference_img)

if ref_faces is None:
    print("Лицо не найдено!")
    exit()

ref_face = ref_faces[0]
ref_align = recognizer.alignCrop(reference_img, ref_face)
ref_feature = recognizer.feature(ref_align)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            align = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(align)

            score = recognizer.match(
                ref_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            x, y, bw, bh = map(int, face[:4])

            if score > 0.5:
                color = (0, 255, 0)
                label = "ME"
            else:
                color = (0, 0, 255)
                label = "OTHER"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    max_width = 800

    scale = max_width / w

    new_w = int(w * scale)
    new_h = int(h * scale)

    frame_resized = cv2.resize(frame, (new_w, new_h))
    cv2.imshow("Face Verification", frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()