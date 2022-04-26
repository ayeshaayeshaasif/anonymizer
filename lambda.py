from anonymizer.detection import Detector, get_weights_path
from anonymizer.obfuscation import Obfuscator
from anonymizer.utils import Box
from PIL import Image
import numpy as np
from aws_lambda_typing import context as context_, events, responses
import json
import requests
from typing import Dict
import io
import boto3
import math



s3_bucket_name = "my-test-bucket-name"
s3 = boto3.client('s3', aws_access_key_id="test", aws_secret_access_key="test")
weights_path = './weights'
kernel_size, sigma, box_kernel_size = 21,2,9
obfuscator = Obfuscator(kernel_size=int(kernel_size), sigma=float(sigma), box_kernel_size=int(box_kernel_size))
detectors = {
    'face': Detector(kind='face', weights_path=get_weights_path(weights_path, kind='face')),
    'plate': Detector(kind='plate', weights_path=get_weights_path(weights_path, kind='plate'))
}
detection_thresholds = {
    'face': 0.3,
    'plate': 0.3
}

def handler(event: events.APIGatewayProxyEventV2, context: context_.Context) -> responses.APIGatewayProxyResponseV2:
    try:
        input: RequestBody = json.loads(event["body"])
    except:
        return respond_with_error("Invalid json", status=400)

    try:
        image = Image.open(requests.get(input["url"], stream=True).raw).convert('RGB')
    except:
        return respond_with_error("Error loading image", status=500)

    
    split_width = 2000
    split_height = 2500

    X_points = start_points(image.width, split_width, overlap=0.1)

    # for equirrectangular panoramas, we can ignore the upper and lower thirds
    if math.floor(image.width / image.height) == 2:
        roi_start = math.floor(image.height * 0.4)
        roi_stop = math.floor(image.height * 0.75)
        Y_points = start_points(image.height, split_height, overlap=0.1, region_of_interest=[roi_start, roi_stop])
    else:
        Y_points = start_points(image.height, split_height, overlap=0.1)


    coords = []
    for x_start, x_stop in X_points:
        for y_start, y_stop in Y_points:
            coords.append((x_start, x_stop, y_start, y_stop))

    detections = []
    for x_start, x_stop, y_start, y_stop in coords:
        tile = image.crop((x_start, y_start, x_stop, y_stop))
        np_tile = np.array(tile)

        for kind, detector in detectors.items():
            boxes = detector.detect(np_tile, detection_threshold=detection_thresholds[kind])

            for detection in boxes:
                detections.append(Box(
                    detection.x_min + x_start,
                    detection.y_min + y_start,
                    detection.x_max + x_start,
                    detection.y_max + y_start,
                    detection.score,
                    detection.kind
                ))
    
    result_url = None

    if (len(detections)):
        np_image = np.array(image)
        obfuscated_np_image = obfuscator.obfuscate(np_image, detections)
        result_url = upload_image_to_s3(obfuscated_np_image, image.format)

    return respond({
        "detections": list(map(lambda box: { 
            "x_min": box.x_min,
            "y_min": box.y_min,
            "x_max": box.x_max,
            "x_max": box.x_max,
            "score": box.score,
            "kind": box.kind,
         }, detections)),
        "result_url": result_url,
        "meta": input["meta"], # passthrough any meta fields for convenience
    }, status=200)

def respond_with_error(message: str, status: int = 400) -> responses.APIGatewayProxyResponseV2:
    return respond({
        "error": message,
    }, status=status)
    
def respond(body: Dict, status: int = 200) -> responses.APIGatewayProxyResponseV2:
    return responses.APIGatewayProxyResponseV2(
        statusCode=status,
        headers={
            "Content-Type": "application/json",
        },
        body=json.dumps(body),
        isBase64Encoded=False,
    )

def load_np_image(image_url: str) -> np.ndarray:
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    np_image = np.array(image)
    return np_image


def upload_image_to_s3(image, format) -> str:
    obfuscated_image = Image.fromarray((image).astype(np.uint8), mode='RGB')
    in_mem_file = io.BytesIO()
    obfuscated_image.save(in_mem_file, format=format)
    in_mem_file.seek(0)

    import uuid
    object_name = str(uuid.uuid4())

    s3.upload_fileobj(
        in_mem_file,
        s3_bucket_name,
        object_name,
        ExtraArgs={
            'ACL': 'public-read'
        }
    )

    return s3.generate_presigned_url('get_object', Params={
        "Bucket": s3_bucket_name,
        "Key": object_name
    }, ExpiresIn=3600)

def start_points(size, split_size, overlap=0, region_of_interest=None):
    offset = 0

    if region_of_interest:
        size = region_of_interest[1] - region_of_interest[0]
        offset = region_of_interest[0]
    
    stride = int(split_size * (1-overlap))
    
    points = []
    counter = 0
    
    while True:
        start_point = stride * counter
        
        if start_point + split_size >= size: # last box, will potentially exceed size
            points.append((start_point+offset, size+offset))
            break
        
        points.append((start_point+offset, start_point+split_size+offset))
        counter += 1
    return points

class RequestBody():
    url: str
    meta: Dict[str, str]
