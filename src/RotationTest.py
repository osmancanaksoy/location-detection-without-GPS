import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, \
    QGridLayout, QGroupBox, QSizePolicy
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from loguru import logger
from superpoint_superglue_deployment import Matcher


class VideoMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ref_image = cv2.imread("rize_drone/rize1.jpg")
        self.ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        self.scale_percent = 30
        self.frame_counter = 0
        self.centroid_list = []

        width = int(self.ref_image.shape[1] * self.scale_percent / 100)
        height = int(self.ref_image.shape[0] * self.scale_percent / 100)
        self.dim = (width, height)

        self.superglue_matcher = Matcher(
            {
                "superpoint": {
                    "input_shape": (-1, -1),
                    "keypoint_threshold": 0.003,
                },
                "superglue": {
                    "match_threshold": 0.5,
                },
                "use_gpu": True,
            }
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        self.setWindowTitle('Video Matcher')
        self.setGeometry(100, 100, 1600, 900)

        # Control panel
        self.control_panel = QWidget(self)
        self.control_layout = QVBoxLayout(self.control_panel)

        self.open_button = QPushButton('Select Video', self)
        self.open_button.clicked.connect(self.select_video)
        self.control_layout.addWidget(self.open_button)

        self.save_button = QPushButton('Save Route Image', self)
        self.save_button.clicked.connect(self.save_route_image)
        self.control_layout.addWidget(self.save_button)

        self.control_group = QGroupBox('Control Panel')
        self.control_group.setLayout(self.control_layout)
        self.control_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Original video display
        self.original_video_widget = QVideoWidget(self)
        self.original_video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.audio_output = QAudioOutput(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.original_video_widget)

        self.original_video_group = QGroupBox('Original Video')
        original_video_layout = QVBoxLayout()
        original_video_layout.addWidget(self.original_video_widget)
        self.original_video_group.setLayout(original_video_layout)
        self.original_video_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Processed video display
        self.processed_video_label = QLabel(self)
        self.processed_video_label.setAlignment(Qt.AlignCenter)
        self.processed_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.processed_video_group = QGroupBox('Matching Result')
        processed_video_layout = QVBoxLayout()
        processed_video_layout.addWidget(self.processed_video_label)
        self.processed_video_group.setLayout(processed_video_layout)
        self.processed_video_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Route image display
        self.route_label = QLabel(self)
        self.route_label.setAlignment(Qt.AlignCenter)
        self.route_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.route_group = QGroupBox('Route')
        route_layout = QVBoxLayout()
        route_layout.addWidget(self.route_label)
        self.route_group.setLayout(route_layout)
        self.route_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Grid layout
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.control_group, 0, 0)
        self.grid_layout.addWidget(self.original_video_group, 0, 1)
        self.grid_layout.addWidget(self.processed_video_group, 1, 0)
        self.grid_layout.addWidget(self.route_group, 1, 1)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.central_widget)

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)",
                                                   options=options)
        if file_name:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            # self.media_player.play()
            self.video_capture = cv2.VideoCapture(file_name)
            self.timer.start(1)  # Start the timer for video processing

    def update_frame(self):
        if self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self.video_capture.release()
            self.display_route_image()  # Display route image at the end
            return

        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            query_image = frame
            query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            query_kpts, ref_kpts, _, _, matches = self.superglue_matcher.match(query_gray, self.ref_gray)

            if not matches:
                logger.warning("No matches found.")
                return

            M, mask = cv2.findHomography(
                np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
                np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=5.0,
                maxIters=10000,
                confidence=0.95,
            )
            logger.info(f"number of inliers: {mask.sum()}")
            matches = np.array(matches)[np.all(mask > 0, axis=1)]
            matches = sorted(matches, key=lambda match: match.distance)

            result_image, centroid = self.draw_matches_with_lines(query_image, self.ref_image, query_kpts, ref_kpts,
                                                                  matches, M)
            self.centroid_list.append(centroid)
            self.display_image(result_image, self.processed_video_label)

    def draw_matches_with_lines(self, query_image, ref_image, query_kpts, ref_kpts, matches, M):
        query_corners = np.float32([[0, 0], [0, query_image.shape[0]], [query_image.shape[1], query_image.shape[0]],
                                    [query_image.shape[1], 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(query_corners, M)
        transformed_corners_list = [tuple(point[0]) for point in transformed_corners]

        ref_image = cv2.polylines(ref_image, [np.array(transformed_corners_list, dtype=np.int32)], True, (255, 0, 0), 4,
                                  cv2.LINE_AA)
        centroid = np.mean(transformed_corners, axis=0, dtype=np.int32)

        matched_image = cv2.drawMatches(query_image, query_kpts, ref_image, ref_kpts, matches[:50], None, flags=2)
        return matched_image, centroid

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        target_size = label.size()  # Hedef boyutu al
        scale_factor = min(target_size.width() / w, target_size.height() / h)  # Ölçek faktörünü belirle
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)  # Yeni boyutları hesapla
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Yeni boyutlara göre yeniden boyutlandır

        bytes_per_line = ch * new_w
        q_img = QImage(img.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
    def save_route_image(self):
        logger.info('Route image saved as route_image.jpg')
        self.display_route_image()


    def project_point_to_route(self, point, route):
        """
        Verilen bir noktayı rotaya yansıtır.

        :param point: Yansıtılacak nokta (x, y)
        :param route: Rota noktalarının listesi [(x1, y1), (x2, y2), ...]
        :return: Yansıtılmış nokta (x', y')
        """
        x, y = point
        min_dist = float('inf')
        closest_point = None
        for route_point in route:
            dist = np.linalg.norm(np.array(point) - np.array(route_point))
            if dist < min_dist:
                min_dist = dist
                closest_point = route_point
        return closest_point

    def display_route_image(self):
        route_image = cv2.imread("rize_drone/rize1.jpg")
        smoothed_centroids = self.centroid_list
        print(smoothed_centroids)
        for centroid in smoothed_centroids:
            centroid = centroid[0]  # NumPy dizisini dizeden çıkar
            if len(centroid) == 2:  # Koordinatlar (x, y) şeklinde mi kontrol et
                cv2.circle(route_image, tuple(centroid), 15, (0, 255, 0), -1)
        for i in range(len(smoothed_centroids) - 1):
            if len(smoothed_centroids[i]) == 2 and len(smoothed_centroids[i + 1]) == 2:  # Koordinatlar (x, y) şeklinde mi kontrol et
                cv2.line(route_image, tuple(smoothed_centroids[i]), tuple(smoothed_centroids[i + 1]),
                         color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        self.display_image(route_image, self.route_label)
        self.media_player.play()
        self.media_player.mediaStatusChanged.connect(self.handleMediaStatus)

    def handleMediaStatus(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.play()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoMatcherApp()
    main_window.show()
    sys.exit(app.exec())
