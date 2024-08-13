import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, \
    QGridLayout, QGroupBox, QSizePolicy, QLineEdit
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from loguru import logger
from superpoint_superglue_deployment import Matcher
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter



class VideoMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ref_image = cv2.imread("uydu/rize_uydu.jpg")
        self.ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        self.frame_interval = 30

        self.scale_percent = 50
        width = int(self.ref_image.shape[1] * self.scale_percent / 100)
        height = int(self.ref_image.shape[0] * self.scale_percent / 100)
        self.dim = (width, height)
        self.frame_counter = 0
        self.centroid_list = []



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

        self.select_satellite_button = QPushButton('Select Satellite Image', self)
        self.select_satellite_button.clicked.connect(self.select_satellite_image)
        self.control_layout.addWidget(self.select_satellite_button)

        self.frame_interval_edit = QLineEdit(self)
        self.frame_interval_edit.setPlaceholderText('Enter frame interval')
        self.control_layout.addWidget(self.frame_interval_edit)

        self.open_button = QPushButton('Select Video', self)
        self.open_button.clicked.connect(self.select_video)
        self.control_layout.addWidget(self.open_button)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_app)
        self.control_layout.addWidget(self.reset_button)

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

    def select_satellite_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Satellite Image File", "uydu", "Image Files (*.jpg *.png)",
                                                   options=options)
        if file_name:
            self.ref_image = cv2.imread(file_name)
            self.route_image = cv2.imread(file_name)
            self.ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.ref_image, self.processed_video_label)  # Display the selected image


            # Update the label to show the selected image's name
            self.select_satellite_button.setText(f'Selected Satellite Image: {file_name}')

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "drone", "Video Files (*.mp4 *.avi *.mov)",
                                                   options=options)
        if file_name:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            self.video_capture = cv2.VideoCapture(file_name)

            # Frame interval'i kullanıcı tarafından girilen değere göre güncelle
            frame_interval_text = self.frame_interval_edit.text()
            try:
                self.frame_interval = int(frame_interval_text)
            except ValueError:
                logger.warning(f"Invalid frame interval: {frame_interval_text}. Using default value.")

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
        if self.frame_counter % self.frame_interval == 0:
            frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
            query_image = frame
            query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            query_kpts, ref_kpts, _, _, matches = self.superglue_matcher.match(query_gray, self.ref_gray)
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

    def correct_centroids(self,centroid_list):
        # Extract X and Y coordinates from the centroid list
        X = np.array([centroid[0][0] for centroid in centroid_list]).reshape(-1, 1)
        Y = np.array([centroid[0][1] for centroid in centroid_list])

        # Perform linear regression to fit a line to the points
        reg = LinearRegression().fit(X, Y)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        # Project the original points onto the fitted line
        corrected_centroid_list = []
        for x in X:
            y_on_line = slope * x + intercept
            corrected_centroid_list.append(np.array([[int(x), int(y_on_line)]]))

        return corrected_centroid_list
    def save_route_image(self):
        route_image = cv2.imread("uydu/rize_uydu.jpg")
        smoothed_centroids = self.centroid_list
        for centroid in smoothed_centroids:
            cv2.circle(route_image, centroid, 15, (0, 255, 0), -1)
        for i in range(len(smoothed_centroids) - 1):
            cv2.line(route_image, smoothed_centroids[i], smoothed_centroids[i + 1],
                     color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.imwrite('route_image.jpg', route_image)
        logger.info('Route image saved as route_image.jpg')
        self.display_route_image()

    def display_route_image(self):
        route_image = self.route_image
        print(self.centroid_list)
        corrected_centroid_list = self.correct_centroids(self.centroid_list)
        #corrected_centroid_list=self.centroid_list
        for centroid in corrected_centroid_list:
            cv2.circle(route_image, tuple(centroid[0]), 15, (0, 255, 0), -1)
        for i in range(len(corrected_centroid_list) - 1):
            cv2.line(route_image, tuple(corrected_centroid_list[i][0]), tuple(corrected_centroid_list[i + 1][0]),
                     color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        self.display_image(route_image, self.route_label)

        self.media_player.play()
        self.media_player.mediaStatusChanged.connect(self.handleMediaStatus)
    def handleMediaStatus(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.media_player.play()

    def reset_app(self):
        # Reset all variables to initial state
        self.ref_image = cv2.imread("default.jpg")
        self.ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        self.frame_interval = 30
        self.scale_percent = 50
        width = int(self.ref_image.shape[1] * self.scale_percent / 100)
        height = int(self.ref_image.shape[0] * self.scale_percent / 100)
        self.dim = (width, height)
        self.frame_counter = 0
        self.centroid_list = []
        self.route_image = None

        # Reset UI elements
        self.select_satellite_button.setText('Select Satellite Image')
        self.frame_interval_edit.clear()

        # Stop video if playing
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            self.timer.stop()
            self.video_capture.release()

        # Reset displayed images
        self.display_image(self.ref_image, self.processed_video_label)
        self.display_image(np.zeros_like(self.ref_image), self.route_label)  # Display empty image

        # Reset media player
        self.media_player.stop()

        logger.info('Application reset completed.')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoMatcherApp()
    main_window.show()
    sys.exit(app.exec())
