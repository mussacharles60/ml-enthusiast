import FaceFunctions
import Distance
import Facenet
import numpy as np
import cv2
import time

model = Facenet.loadModel()

def extract_faces(
    img_path,
    target_size=(224, 224),
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
    grayscale=False,
):

    """
    This function applies pre-processing stages of a face recognition pipeline
    including detection and alignment

    Parameters:
            img_path: exact image path, numpy array (BGR) or base64 encoded image.
            Source image can have many face. Then, result will be the size of number
            of faces appearing in that source image.

            target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image.

            detector_backend (string): face detection backends are retinaface, mtcnn,
            opencv, ssd or dlib

            enforce_detection (boolean): function throws exception if face cannot be
            detected in the fed image. Set this to False if you do not want to get
            an exception and run the function anyway.

            align (boolean): alignment according to the eye positions.

            grayscale (boolean): extracting faces in rgb or gray scale

    Returns:
            list of dictionaries. Each dictionary will have facial image itself,
            extracted area from the original image and confidence score.

    """

    resp_objs = []
    img_objs = FaceFunctions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img, region, confidence in img_objs:
        resp_obj = {}

        # discard expanded dimension
        if len(img.shape) == 4:
            img = img[0]

        resp_obj["face"] = img[:, :, ::-1]
        resp_obj["facial_area"] = region
        resp_obj["confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


def represent(
    img_path,
    model_name="Facenet",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
):

    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
            encoded images could be passed. Source image can have many faces. Then, result will
            be the size of number of faces appearing in the source image.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object with multidimensional vector (embedding).
            The number of dimensions is changing based on the reference model.
            E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """
    resp_objs = []

    # model = Facenet.loadModel()

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = FaceFunctions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = FaceFunctions.extract_faces(
            img=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:  # skip
        if isinstance(img_path, str):
            img = FaceFunctions.load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f"unexpected type for img_path - {type(img_path)}")
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        # --------------------------------
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]
    # ---------------------------------

    for img, region, _ in img_objs:
        # custom normalization
        img = FaceFunctions.normalize_input(img=img, normalization=normalization)

        # represent
        # if "keras" in str(type(model)):
        #     # new tf versions show progress bar and it is annoying
        #     # embedding = model.predict(img, verbose=0)[0].tolist()
        # else:
        #     # SFace and Dlib are not keras models and no verbose arguments
        #     embedding = model.predict(img)[0].tolist()
        embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_objs.append(resp_obj)

    return resp_objs

def verify(
    img1_path,
    img2_path,
    model_name="Facenet",
    detector_backend="opencv",
    distance_metric="cosine",
    enforce_detection=True,
    align=True,
    normalization="base",
):

    """
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    , "distance": 0.2563
                    , "max_threshold_to_verify": 0.40
                    , "model": "VGG-Face"
                    , "similarity_metric": "cosine"
                    , 'facial_areas': {
                            'img1': {'x': 345, 'y': 211, 'w': 769, 'h': 769},
                            'img2': {'x': 318, 'y': 534, 'w': 779, 'h': 779}
                    }
                    , "time": 2
            }

    """

    tic = time.time()

    # --------------------------------
    target_size = FaceFunctions.find_target_size(model_name=model_name)

    # img pairs might have many faces
    img1_objs = FaceFunctions.extract_faces(
        img=img1_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    img2_objs = FaceFunctions.extract_faces(
        img=img2_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    # --------------------------------
    distances = []
    regions = []
    # now we will find the face pair with minimum distance
    for img1_content, img1_region, _ in img1_objs:
        for img2_content, img2_region, _ in img2_objs:
            img1_embedding_obj = represent(
                img_path=img1_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img2_embedding_obj = represent(
                img_path=img2_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img1_representation = img1_embedding_obj[0]["embedding"]
            img2_representation = img2_embedding_obj[0]["embedding"]

            if distance_metric == "cosine":
                distance = Distance.findCosineDistance(img1_representation, img2_representation)
            elif distance_metric == "euclidean":
                distance = Distance.findEuclideanDistance(img1_representation, img2_representation)
            elif distance_metric == "euclidean_l2":
                distance = Distance.findEuclideanDistance(
                    Distance.l2_normalize(img1_representation), Distance.l2_normalize(img2_representation)
                )
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)

            distances.append(distance)
            regions.append((img1_region, img2_region))

    # -------------------------------
    threshold = Distance.findThreshold(model_name, distance_metric)
    distance = min(distances)  # best distance
    facial_areas = regions[np.argmin(distances)]

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(toc - tic, 2),
    }

    return resp_obj
