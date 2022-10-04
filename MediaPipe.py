import mediapipe as mp
import streamlit as st
import cv2

#-------------------------------------------------------------------------
# グローバル変数の宣言と定義
#-------------------------------------------------------------------------

# オブジェクトの生成
mp_pose        = mp.solutions.pose
mp_hands       = mp.solutions.hands
mp_face_mesh   = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection

# 推定結果の可視化に関するパラメータ
mp_drawing        = mp.solutions.drawing_utils

# 読み込む動画のパス
#cap = cv2.VideoCapture( 0 )

option_select = st.sidebar.radio(
        'MediaPipe Demo',
        ( 'Face Detection', 'Face Mesh', 'Hands', 'Pose' ) )

# 表示用の空のオブジェクト
image_container = st.empty()


#-------------------------------------------------------------------------
# Pose
#-------------------------------------------------------------------------
def detect_Pose():
    # 読み込む動画のパス
    cap = cv2.VideoCapture( 0 )

    # 推定結果の可視化に関するパラメータ
    mesh_drawing_spec = mp_drawing.DrawingSpec( thickness = 2, color=(0, 255, 0) )
    mark_drawing_spec = mp_drawing.DrawingSpec( thickness = 2, circle_radius = 2, color=(0, 0, 255) )

    with mp_pose.Pose(
            min_detection_confidence = 0.5,                         # 検出の閾値
            static_image_mode        = False ) as pose_detection:   # 静止画の否定（＝動画を指定）

        while cap.isOpened():
            success, image = cap.read()             # 動画から１枚の画像を読み込む

            if not success:                         # 読み込みに失敗したときの処理
                print("empty camera frame")
                break

            image      = cv2.flip( image, 1 )                               # 胸像反転
            image_ORG  = cv2.resize( image, dsize=None, fx=1.0, fy=1.0 )    # リサイズ
            image_RGB  = cv2.cvtColor( image_ORG, cv2.COLOR_BGR2RGB )       # カラースペース変換

            results = pose_detection.process( image_RGB )       # MediaPipeのキモ

            if results.pose_landmarks:
                # 座標をターミナルに出力する
                # print( 'x:', results.pose_landmarks.landmark[ 11 ].x )    #left shoulder
                # print( 'y:', results.pose_landmarks.landmark[ 11 ].y )    #left shoulder

                # 検出結果を画像に描画する
                mp_drawing.draw_landmarks(
                    image                   = image_RGB,
                    landmark_list           = results.pose_landmarks,
                    connections             = mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec   = mark_drawing_spec,
                    connection_drawing_spec = mesh_drawing_spec
                    )

            # cv2.imshow('Pose', image_RGB )      # 処理結果を画面出力する
            image_container.image( image_RGB )

            if option_select != 'Pose':
                break

    cap.release()       # オブジェクトの解放


#-------------------------------------------------------------------------
# hands
#-------------------------------------------------------------------------
def detect_HandTracking():
    # 読み込む動画のパス
    cap = cv2.VideoCapture( 0 )

    # 推定結果の可視化に関するパラメータ
    mesh_drawing_spec = mp_drawing.DrawingSpec( thickness = 2, color=(0, 255, 0) )
    mark_drawing_spec = mp_drawing.DrawingSpec( thickness = 2, circle_radius = 2, color=(0, 0, 255) )

    with mp_hands.Hands(
            max_num_hands            = 2, 
            min_detection_confidence = 0.5,
            static_image_mode        = False) as hands_detection:

        while cap.isOpened():
            success, image = cap.read()             # 動画から１枚の画像を読み込む

            if not success:                         # 読み込みに失敗したときの処理
                print("empty camera frame")
                break

            image      = cv2.flip( image, 1 )                               # 胸像反転
            image_ORG  = cv2.resize( image, dsize=None, fx=1.0, fy=1.0 )    # リサイズ
            image_RGB  = cv2.cvtColor( image_ORG, cv2.COLOR_BGR2RGB )       # カラースペース変換

            results = hands_detection.process( image_RGB )

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # for id, lm in enumerate( hand_landmarks.landmark ):
                    #     print( id, lm.x )
                    
                    mp_drawing.draw_landmarks(
                                image                   = image_RGB,
                                landmark_list           = hand_landmarks,
                                connections             = mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec   = mark_drawing_spec,
                                connection_drawing_spec = mesh_drawing_spec
                                )

            image_container.image( image_RGB )

            if option_select != 'Hands':
                break

    cap.release()       # オブジェクトの解放


#-------------------------------------------------------------------------
# Face Mesh
#-------------------------------------------------------------------------
def detect_FaceMesh():
    # 読み込む動画のパス
    cap = cv2.VideoCapture( 0 )

    # 推定結果の可視化に関するパラメータ
    mesh_drawing_spec = mp_drawing.DrawingSpec( thickness = 1, color=(0, 255, 0) )
    mark_drawing_spec = mp_drawing.DrawingSpec( thickness = 1, circle_radius = 1, color=(0, 0, 255) )

    with mp_face_mesh.FaceMesh(
            max_num_faces            = 5,                       # 検出する顔の上限値
            min_detection_confidence = 0.5,                     # 検出の閾値
            static_image_mode        = False ) as face_mesh:    # 静止画の否定（動画を指定）

        while cap.isOpened():
            success, image = cap.read()             # 動画から１枚の画像を読み込む

            if not success:                         # 読み込みに失敗したときの処理
                print("empty camera frame")
                break

            image      = cv2.flip( image, 1 )                               # 胸像反転
            image_ORG  = cv2.resize( image, dsize=None, fx=1.0, fy=1.0 )    # リサイズ
            image_RGB  = cv2.cvtColor( image_ORG, cv2.COLOR_BGR2RGB )       # カラースペース変換

            results = face_mesh.process( image_RGB )

            # 検出した顔を順番に処理する
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # ランドマークの座標を数値出力する
                    # for id, lm in enumerate( face_landmarks.landmark ):
                    #     print( id, lm.x )

                    # 検出結果を画像に描画する
                    mp_drawing.draw_landmarks(
                      image                   = image_RGB,
                      landmark_list           = face_landmarks,
                      connections             = mp_face_mesh.FACEMESH_TESSELATION, #mp_face_mesh.FACEMESH_CONTOURS
                    landmark_drawing_spec   = mark_drawing_spec,
                    connection_drawing_spec = mesh_drawing_spec
                    )

            image_container.image( image_RGB )

            if option_select != 'Face Mesh':
                break

    cap.release()       # オブジェクトの解放



#-------------------------------------------------------------------------
# 顔の領域
#-------------------------------------------------------------------------
def detect_FaceDetect():
    # 読み込む動画のパス
    cap = cv2.VideoCapture( 0 )

    # 推定結果の可視化に関するパラメータ
    kp_drawing_spec   = mp_drawing.DrawingSpec( thickness=3, circle_radius=3, color=(0, 255, 0) )
    bbox_drawing_spec = mp_drawing.DrawingSpec( thickness=3, color=(0, 0, 255) )

    with mp_face_detect.FaceDetection(
            min_detection_confidence = 0.5 ) as face_detection:

        while cap.isOpened():
            success, image = cap.read()             # 動画から１枚の画像を読み込む

            if not success:                         # 読み込みに失敗したときの処理
                print("empty camera frame")
                break

            image      = cv2.flip( image, 1 )                               # 胸像反転
            image_ORG  = cv2.resize( image, dsize=None, fx=1.0, fy=1.0 )    # リサイズ
            image_RGB  = cv2.cvtColor( image_ORG, cv2.COLOR_BGR2RGB )       # カラースペース変換

            results = face_detection.process( image_RGB )

            # 検出した顔を順番に処理する
            if results.detections:
                for detection in results.detections:                # 検出した顔を順番に処理する
                    mp_drawing.draw_detection(                      # 演算結果を画像に描画する
                        image_RGB, 
                        detection,
                        keypoint_drawing_spec = kp_drawing_spec,
                        bbox_drawing_spec     = bbox_drawing_spec
                        )

            image_container.image( image_RGB )

            if option_select != 'Face Detection':
                break

    cap.release()       # オブジェクトの解放


#-------------------------------------------------------------------------
# メイン
#-------------------------------------------------------------------------
if __name__ == "__main__":
    if option_select == 'Face Detection':
        detect_FaceDetect()
    elif option_select == 'Face Mesh':
        detect_FaceMesh()
    elif option_select == 'Hands':
        detect_HandTracking()
    else:
        detect_Pose()
