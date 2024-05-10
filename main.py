import streamlit as st
import tensorflow as tf
import numpy as np
import os

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Bảng điều khiển")
app_mode = st.sidebar.selectbox("Chọn trang",["Trang chủ","Giới thiệu","Nhận biết bệnh"])

#Main Page
if(app_mode=="Trang chủ"):
    st.header("HỆ THỐNG NHẬN BIẾT BỆNH THỰC VẬT")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Chào mừng bạn đến với Hệ thống nhận biết bệnh cây trồng! 🌿🔍
    
     Nhiệm vụ của chúng tôi là giúp xác định bệnh cây một cách hiệu quả. Tải hình ảnh cây lên và hệ thống của chúng tôi sẽ phân tích hình ảnh đó để phát hiện bất kỳ dấu hiệu bệnh nào. Cùng nhau, chúng ta hãy bảo vệ cây trồng của chúng ta và đảm bảo một vụ thu hoạch lành mạnh hơn!

     ### Làm thế nào nó hoạt động
     1. **Upload Image:** Vào trang **Disease Recognition** và tải lên hình ảnh cây nghi ngờ mắc bệnh.
     2. **Analysis:** Hệ thống của chúng tôi sẽ xử lý hình ảnh bằng các thuật toán tiên tiến để xác định các bệnh tiềm ẩn.
     3. **Results:** Xem kết quả và đề xuất hành động tiếp theo.

     ### Tại sao chọn chúng tôi?
     - **Độ chính xác:** Hệ thống của chúng tôi sử dụng các kỹ thuật máy học hiện đại để phát hiện bệnh chính xác.
     - **Thân thiện với người dùng:** Giao diện đơn giản và trực quan mang lại trải nghiệm người dùng liền mạch.
     - **Nhanh chóng và Hiệu quả:** Nhận kết quả sau vài giây, cho phép đưa ra quyết định nhanh chóng.

     ### Bắt đầu
     Nhấp vào trang **Disease Recognition** trong thanh bên để tải lên hình ảnh và trải nghiệm sức mạnh của Hệ thống nhận biết bệnh thực vật của chúng tôi!

     ### Về chúng tôi
     Tìm hiểu thêm về dự án, nhóm của chúng tôi và mục tiêu của chúng tôi trên trang **About**.
    """)

#About Project
elif(app_mode=="Giới thiệu"):
    st.header("Giới thiệu")
    st.markdown("""
                #### Giới thiệu về Tập dữ liệu
                 Tập dữ liệu này được tạo lại bằng cách sử dụng tính năng tăng cường ngoại tuyến từ tập dữ liệu gốc.
                 Tập dữ liệu này bao gồm khoảng 3k hình ảnh rgb của lá cây khỏe mạnh và bị bệnh, được phân loại thành 38 lớp khác nhau.
                 Một thư mục mới chứa 33 ảnh thử nghiệm được tạo ra nhằm mục đích dự đoán.
                #### Nội dung
                1. train (2281 images)
                2. test (33 images)
                3. validation (1392 images)
                """)

#Prediction Page
elif(app_mode=="Nhận biết bệnh"):
    st.header("Nhận biết bệnh")
    test_image = st.file_uploader("Chọn một ảnh:")
    if(st.button("Hiển thị ảnh")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Dự đoán")):
        st.success("Kết quả dự đoán sẽ được hiển thị ở đây.")
        if test_image is not None:
            # Save the uploaded file to a temporary location
            with open("temp_image.jpg", "wb") as f:
                f.write(test_image.read())
            # Use the saved file path to load and predict
            result_index = model_prediction("temp_image.jpg")
            # Remove the temporary file
            os.remove("temp_image.jpg")

        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Mô hình dự đoán đó là một {}".format(class_name[result_index]))
else:
        st.error("Vui lòng tải lên một hình ảnh trước khi dự đoán.")