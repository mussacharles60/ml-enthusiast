
import numpy as np
import cv2
import os
import json
import Detector
import Distance


# An input file to get stored embeddings
input_file = 'embeddings/embeddings.json'

stored_embeddings = {}

# Load the embeddings from the JSON file
with open(input_file, 'r') as json_file:
    stored_data = json.load(json_file)

    # Iterate through the JSON data and convert it to the desired format
    for user_data in stored_data:
        user_name = user_data["name"]
        user_embeddings = user_data["embeddings"]

        # Convert user's embeddings to a flat list
        embeddings_list = []
        for key, embeddings in user_embeddings.items():
            for embedding in embeddings:
                embeddings_list.append(embedding)

        # Store the converted data in the dictionary
        stored_embeddings[user_name] = {
            "embeddings": embeddings_list
        }

print(stored_embeddings)


def draw_text(
        img, 
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0)
    ):

    x, y = pos
    text_size, _ = cv2.getTextSize(
        text=text, 
        fontFace=font, 
        fontScale=font_scale, 
        thickness=font_thickness
    )
    text_w, text_h = text_size
    cv2.rectangle(
        img=img, 
        pt1=pos, 
        pt2=(x + text_w, y + text_h), 
        color=text_color_bg, 
        thickness=-1
    )
    cv2.putText(
        img=img, 
        text=text, 
        org=(x, int(y + text_h + font_scale - 1)), 
        fontFace=font, 
        fontScale=font_scale, 
        color=text_color, 
        thickness=font_thickness
    )

    return text_size

def process_detected_faces(detected_faces, frame):
    # print("detected_faces:", detected_faces)

    if len(detected_faces) > 0:

        for face_obj in detected_faces:

            if face_obj["confidence"] >= 2.0:

                x = face_obj["facial_area"]["x"]
                y = face_obj["facial_area"]["y"]
                w = face_obj["facial_area"]["w"]
                h = face_obj["facial_area"]["h"]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract embeddings from the face using FaceNet
                # face_embeddings = face_net_model.predict(face_img)
                face_embeddings = []
                _face_embeddings = Detector.represent(
                    img_path=face_obj["face"],
                    enforce_detection=False,
                    normalization="Facenet",
                )
                if len(_face_embeddings) > 0:
                    face_embeddings.append(_face_embeddings[0]["embedding"])

                    # Compare the embeddings with stored embeddings for each user using cosine similarity
                    found_similarity = -1.0
                    recognized_face = None

                    for name, stored_user_data in stored_embeddings.items():

                        distances = []

                        for stored_embedding in stored_user_data["embeddings"]:

                            source_representation = stored_embedding
                            test_representation = face_embeddings[0]

                            print("source_representation:  ", source_representation)
                            print("test_representation:    ", test_representation)

                            # print(f"source_representation: {len(source_representation)}")
                            # print(f"test_representation:   {len(test_representation)}")

                            # Check if there is a dimension mismatch
                            if len(test_representation) != len(source_representation):
                                # Calculate the number of zero values to append
                                num_zeros_to_append = len(
                                    source_representation) - len(test_representation)
                                # Append zero values to test_representation
                                test_representation = np.append(
                                    test_representation, np.zeros(num_zeros_to_append))

                            # Calculate cosine distance
                            cosine_distance = Distance.findCosineDistance(source_representation, test_representation)

                            print(f"Cosine Distance: {cosine_distance}")

                            distances.append(cosine_distance)
                        
                        threshold = Distance.findThreshold(model_name="Facenet", distance_metric="cosine")
                        distance = min(distances) # best distance
                        print(f"Min Distance: {distance}, threshold: {threshold}")

                        if distance <= threshold:
                            found_similarity = distance
                            recognized_face = name
                            break

                    print(f"similarity: {found_similarity:.3f}")
                    print("--------------------------------------")

                    if recognized_face is not None:  # Check against the threshold for recognition
                        
                        cv2.rectangle(
                            img=frame,
                            pt1=(x, y),
                            pt2=(x + w, y + h),
                            color=(0, 255, 0),
                            thickness=2,
                        )
                        draw_text(
                            img=frame, 
                            text=f"{recognized_face} ({found_similarity:.3f})", 
                            pos=(x, y+1), 
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=.7,
                            font_thickness=1,
                            text_color=(255, 255, 255),
                            text_color_bg=(0, 255, 0),
                        )

                    else:
                        cv2.rectangle(
                            img=frame,
                            pt1=(x, y),
                            pt2=(x + w, y + h),
                            color=(0, 0, 255),
                            thickness=2,
                            # lineType: int = ...,
                            # shift: int = ...
                        )
                        draw_text(
                            frame, 
                            f"Unknown", 
                            pos=(x, y+1), 
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=.7,
                            font_thickness=1,
                            text_color=(255, 255, 255),
                            text_color_bg=(0, 0, 255),
                        )



def recognize_image(img_path):

    image = cv2.imread(img_path)

    # Define the maximum width and height for display
    max_width = 800
    max_height = 600

    # Get the current dimensions of the image
    height, width, _ = image.shape

    # Check if resizing is necessary
    if width > max_width or height > max_height:
        # Calculate the scaling factor to fit within the maximum dimensions
        scale = min(max_width / width, max_height / height)

        # Calculate the new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        image = cv2.resize(image, (new_width, new_height))

    detected_faces = Detector.extract_faces(
        img_path=image,
        enforce_detection=False
    )
    for i, face_obj in enumerate(detected_faces):
        print(f'face: {i + 1}, confidence: {face_obj["confidence"]}')

    process_detected_faces(detected_faces=detected_faces, frame=image)

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("-------------------------------------")

# image_folder = 'gallery_images'
# # List all image files in the folder
# image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if (filename.lower().endswith('.jpg') or filename.lower().endswith('.png'))]

# image_files

# if len(image_files) > 0:
#     for image_path in image_files:
#         recognize_image(image_path)

# recognize_image("input_images\\Mussa\\2022-12-12 (1).jpg")
# recognize_image("D:\\Downloads\\ali_gallo_instagram_2.jpg")
# recognize_image("D:\\Downloads\\pearl-andersson-soundcloud.jpg")
# recognize_image("D:\\Projects\\ml-enthusiast\\face\\input_images\\Mussa\\20200130_172549.jpg")
# recognize_image("D:\\Projects\\ml-enthusiast\\face\\input_images\\Mussa\\2022-12-12 (16).jpg")
# recognize_image("D:\\Downloads\\Ai Khodijah - singer Islam.png")
# recognize_image("D:\\Downloads\\portrait-young-patient-lying-bed-during-clinical-examination-recovering-after-medical-surgery-hospital-ward-sick-woman-looking-into-camera-waiting-healthcare-treatment.jpg")
# recognize_image("D:\\Projects\\ml-enthusiast\\face\\input_images\\Mussa\\2022-12-12 (17).jpg")
# recognize_image("D:\\Downloads\\FB_IMG_1581885670173.jpg")
# recognize_image("D:\\Projects\\ml-enthusiast\\face\\input_images\Mussa\\2022-12-12 (6).jpg")
# recognize_image("D:\\Downloads\\FvCGMCrakAMFIMB.png")
# recognize_image("D:\\Downloads\\130863182_382785296508190_2107725103963850677_n.jpg")
# recognize_image("D:\\Downloads\\IMG_E0196.JPG")
# recognize_image("D:\\Downloads\\20230820_121603.jpg")
# recognize_image("D:\\Downloads\\07b7ab7c91ac4820b7a535b0ecd9b6c3.jpg")
# recognize_image("D:\\Downloads\\Screenshot_20230523_163206_WhatsAppBusiness.jpg")
# recognize_image("D:\\Downloads\\2022-12-12 (18).jpg")


def main():
    # Create a VideoCapture object to capture video from the default camera (usually 0)
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # width
    cap.set(4,480) # height

    while True:
        # Read a frame from the camera
        _, frame = cap.read()
        
        detected_faces = Detector.extract_faces(
            img_path=frame,
            enforce_detection=False
        )
        for i, face_obj in enumerate(detected_faces):
            print(f'face: {i + 1}, confidence: {face_obj["confidence"]}')

        process_detected_faces(detected_faces=detected_faces, frame=frame)
        
        cv2.imshow('Face Recognition', frame)

        # Listen for a key press and check if it's the "Esc" key (27 is the ASCII code for Esc)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

main()
