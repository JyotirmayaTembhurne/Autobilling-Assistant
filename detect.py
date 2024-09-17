from inference_sdk import InferenceHTTPClient


def Detect(frame):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com", api_key="jo44BrPI86EBU8XCkWGM"
    )

    result = CLIENT.infer(
        frame,
        model_id="grocery-dataset-q9fj2/5",
    )
    return result


# result = Detect(r"E:\Proj2\360_F_251436027_P0Azvbjh2sLqDvAQ5DYUh3B0ptDBw71B.webp")
# print(result)
