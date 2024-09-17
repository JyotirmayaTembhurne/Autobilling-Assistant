import streamlit as st
import cv2
import detect
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLOWorld


allowed_classes = [
    # "person",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "chair",
    "couch",
    # "potted plant",
    # "dining table",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Initialize the model
model = YOLOWorld(r"yolov8x-worldv2.pt")
model.set_classes(allowed_classes)


def count_objects(file):
    results = model.predict(file)
    return results


def get_class_counts(results):
    class_counts = {class_name: 0 for class_name in allowed_classes}
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        class_counts[class_name] += 1
    return class_counts


bill = list()
product_prices = {
    "Bisconni Chocolate Chip Cookies 46.8gm": 20,
    "Coca Cola Can 250ml": 30,
    "Colgate Maximum Cavity Protection 75gm": 60,
    "Fanta 500ml": 40,
    "Fresher Guava Nectar 500ml": 50,
    "Fruita Vitals Red Grapes 200ml": 34,
    "Islamabad Tea 238gm": 120,
    "Kolson Slanty Jalapeno 18gm": 12,
    "Kurkure Chutney Chaska 62gm": 16,
    "LU Candi Biscuit 60gm": 8,
    "LU Oreo Biscuit 19gm": 4,
    "LU Prince Biscuit 55.2gm": 14,
    "Lays Masala 34gm": 24,
    "Lays Wavy Mexican Chili 34gm": 24,
    "Lifebuoy Total Protect Soap 96gm": 40,
    "Lipton Yellow Label Tea 95gm": 100,
    "Meezan Ultra Rich Tea 190gm": 140,
    "Peek Freans Sooper Biscuit 13.2gm": 6,
    "Safeguard Bar Soap Pure White 175gm": 48,
    "Shezan Apple 250ml": 36,
    "Sunsilk Shampoo Soft - Smooth 160ml": 120,
    "Super Crisp BBQ 30gm": 12,
    "Supreme Tea 95gm": 100,
    "Tapal Danedar 95gm": 100,
    "Vaseline Healthy White Lotion 100ml": 80,
}

counter = {
    "Bisconni Chocolate Chip Cookies 46.8gm": 0,
    "Coca Cola Can 250ml": 0,
    "Colgate Maximum Cavity Protection 75gm": 0,
    "Fanta 500ml": 0,
    "Fresher Guava Nectar 500ml": 0,
    "Fruita Vitals Red Grapes 200ml": 0,
    "Islamabad Tea 238gm": 0,
    "Kolson Slanty Jalapeno 18gm": 0,
    "Kurkure Chutney Chaska 62gm": 0,
    "LU Candi Biscuit 60gm": 0,
    "LU Oreo Biscuit 19gm": 0,
    "LU Prince Biscuit 55.2gm": 0,
    "Lays Masala 34gm": 0,
    "Lays Wavy Mexican Chili 34gm": 0,
    "Lifebuoy Total Protect Soap 96gm": 0,
    "Lipton Yellow Label Tea 95gm": 0,
    "Meezan Ultra Rich Tea 190gm": 0,
    "Peek Freans Sooper Biscuit 13.2gm": 0,
    "Safeguard Bar Soap Pure White 175gm": 0,
    "Shezan Apple 250ml": 0,
    "Sunsilk Shampoo Soft - Smooth 160ml": 0,
    "Super Crisp BBQ 30gm": 0,
    "Supreme Tea 95gm": 0,
    "Tapal Danedar 95gm": 0,
    "Vaseline Healthy White Lotion 100ml": 0,
}


def process_detection_result(detection_result, bill):
    try:
        if detection_result["predictions"]:
            for prediction in detection_result["predictions"]:
                item_name = prediction["class"]
                confidence = prediction["confidence"]
                if confidence >= 0.5:
                    price = product_prices.get(item_name, 0)
                    counter[item_name] += 1
                    bill.append((item_name, price, counter[item_name]))
    except Exception as e:
        print(f"Error processing detection result: {e}")


def real_time_detection(cap):
    start_time = time.time()
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        current_time = time.time()
        if current_time - start_time >= 2:
            detection_result = detect.Detect(last_frame)
            print("Detection result:", detection_result)
            process_detection_result(detection_result, bill)
            print(bill)
            start_time = time.time()


def draw_boxes(frame, detections):
    global high_confidence_objects
    for detection in detections["predictions"]:
        x1 = int(detection["x"] - detection["width"] / 2)
        y1 = int(detection["y"] - detection["height"] / 2)
        x2 = int(detection["x"] + detection["width"] / 2)
        y2 = int(detection["y"] + detection["height"] / 2)
        label = detection["class"]
        confidence = detection["confidence"]

        # Append object class to dictionary if confidence is above 0.60
        if confidence > 0.60:
            if label in st.session_state.high_confidence_objects:
                st.session_state.high_confidence_objects[label] += 1
            else:
                st.session_state.high_confidence_objects[label] = 1

        # Determine the color based on the confidence score
        if confidence <= 0.30:
            color = (0, 0, 255)  # Red
        elif 0.30 < confidence < 0.60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame


def main():
    st.set_page_config(
        layout="wide", page_title="AI Billing System", page_icon="receipt"
    )
    st.sidebar.title("DETECTABLE OBJECTS FOR BILLING")
    df = pd.DataFrame(list(product_prices.items()), columns=["Item", "Price"])
    st.sidebar.table(df)
    st.sidebar.title("DETECTABLE OBJECTS FOR INVENTORY COUNTING")
    df = pd.DataFrame(list(allowed_classes), columns=["Item"])
    st.sidebar.table(df)
    st.markdown(
        """
    <style>
    .body{
        padding: 4em;
    }
    .st-al {
        display: flex;
        justify-content: center;
    }
    .st-emotion-cache-vdokb0 p{
        text-align:center;
        background-color:#CCD7DA;
        font-size:large;
        border-radius: 10px; 
        padding: 1em;
    }

    .st-b2 {
        min-height: 2.7vh;
        min-width: 2.5vw;
    }
    .st-emotion-cache-j6qv4b p{
        font-size: 1.2rem;
    }
    .st-emotion-cache-gdzsw5{
        color: darkolivegreen;
    }
    .st-emotion-cache-qeahdt h1 {
        text-align: center;
    }
    .st-emotion-cache-1jmvea6 p{
        margin-top: 8vh;
        margin-bottom:2px;
        font-size: 20px;
        text-align: center;
    }
    .st-emotion-cache-g03d4b{
        margin-left: 0.7vw;
        border-radius: 20px;
    }
    .st-emotion-cache-1ec096l{
        
    }
}   
    </style>
    """,
        unsafe_allow_html=True,
    )

    page_title = """<p style= "font-family: Source Sans Pro; 
                    color:Darkred; font-size: 45px; text-align:center;
                    position: relative; bottom: 5vh; margin-bottom: 2vh;">AI AUTO-BILLING & INVENTORY COUNTING SYSTEM</p>"""
    st.markdown(page_title, unsafe_allow_html=True)
    on = st.toggle(
        "Start Billing",
        help="Toggle the button to start billing",
        label_visibility="visible",
        value=False,
    )
    reset = st.button(label="Reset Table")
    frame_placeholder = st.empty()
    object_list_placeholder = st.empty()
    total_bill_placeholder = st.empty()
    if "high_confidence_objects" not in st.session_state:
        st.session_state.high_confidence_objects = {}
    if reset:
        st.session_state.high_confidence_objects = {}

    if on:
        cap = cv2.VideoCapture(0)

        if cap is not None and cap.isOpened():
            last_time = time.time()
            while on:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture video frame.")
                    break
                current_time = time.time()
                if current_time - last_time >= 2:
                    detection_results = detect.Detect(frame)
                    frame_with_boxes = draw_boxes(frame, detection_results)
                    annotated_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB")
                    data = []
                    total_bill = 0
                    for item, count in st.session_state.high_confidence_objects.items():
                        price = product_prices.get(item, 0) * count
                        total_bill += price
                        data.append({"Object": item, "Count": count, "Price": price})
                    high_confidence_df = pd.DataFrame(data)
                    object_list_placeholder.table(high_confidence_df)
                    total_bill_placeholder.write(
                        f"Total Bill: {total_bill} currency units"
                    )

                    last_time = current_time
    else:
        cap = None
    data = []
    total_bill = 0
    for item, count in st.session_state.high_confidence_objects.items():
        price = product_prices.get(item, 0) * count
        total_bill += price
        data.append({"Object": item, "Count": count, "Price": price})
    high_confidence_df = pd.DataFrame(data)
    object_list_placeholder.table(high_confidence_df)
    total_bill_text = f"""<p style="font-family: Source Sans Pro; 
                    color:darkred; font-size: 24px;">
                    Total Bill: {total_bill} currency units</p>"""
    st.markdown(total_bill_text, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "INVENTORY COUNTING", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = count_objects(image)
        annotated_image = results[0].plot()
        annotated_image_pil = Image.fromarray(annotated_image)
        st.image(
            annotated_image_pil,
            caption="Uploaded Image with Detections",
            use_column_width=True,
        )
        st
        st.write("Image uploaded successfully!")
        class_counts = get_class_counts(results)
        class_counts_df = pd.DataFrame(
            list(class_counts.items()), columns=["Object", "Count"]
        )
        class_counts_df = class_counts_df[class_counts_df["Count"] > 0]
        class_counts_df.index.name = "Index"
        st.table(class_counts_df)
    else:
        upload_image = """<p style= "font-family: Source Sans Pro; 
                    color:darkolivegreen; font-size: 20px; text-align:left;
                    background: None; 
                    margin-top: -1.2em; margin-left: -0.9em;
                    ">Please upload an image</p>"""
        st.markdown(upload_image, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
