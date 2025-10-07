import streamlit as st

def creds_entered():
    if st.session_state['user'].strip() == 'admin' and st.session_state['passwd'].strip() == 'admin':
        st.session_state['authenticated'] = True
    else:
        st.session_state['authenticated'] = False
        if not st.session_state['passwd']:
            st.warning('Please enter password')
        elif not st.session_state['user']:
            st.warning('Please enter username')
        else:
            st.error('Invalid username or password')

def authenticate_user():
    if 'authenticated' not in st.session_state:

        st.title("Please enter your credentials")

        st.text_input(label = 'Username :', value = '', key = 'user', on_change = creds_entered)
        st.text_input(label = 'Passowrd :', value = '', key = 'passwd', type = 'password', on_change = creds_entered)
        return False

    else:
        if st.session_state['authenticated']:
            return True
        else:
            st.text_input(label = 'Username :', value = '', key = 'user', on_change = creds_entered)
            st.text_input(label = 'Passowrd :', value = '', key = 'passwd', type = 'password', on_change = creds_entered)
            return False
        
if authenticate_user():        

        from PIL import Image
        from Model import main_1, main_2, main_3
        import base64
        import tempfile
        import os
        import numpy as np

        if 'binary_images' not in st.session_state:
            st.session_state.binary_images = []
        if 'cluster_labels' not in st.session_state:
            st.session_state.cluster_labels = 0

        # Streamlit configuration
        st.set_page_config(page_title="Digital Image Processing")
        st.title("Rock Image Segmentation")

        uploaded_file = st.file_uploader("Upload an image")
        refresh = st.button("Refresh")


        if refresh:
            st.session_state.binary_images = []
            st.session_state.cluster_labels = 0
            uploaded_file = None


        if uploaded_file is not None or refresh:
            
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)

            if image is not None:
                st.image(image, caption="Uploaded Image", width=350)

            if uploaded_file is not None:
                submit = st.button("Segment")
                new = st.button("Remove boundary")

                if new:
                    # Process the image using main_3 function on uploaded image

                    numpy_image = np.array(image)

                    binary_image = main_3(numpy_image)
                    pil_binary_image = Image.fromarray(binary_image)

                    st.image(pil_binary_image, width=350)

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        pil_binary_image.save(temp_file.name)
                    st.markdown(f"<a href='data:file/png;base64,{base64.b64encode(open(temp_file.name, 'rb').read()).decode()}' download='No_boundary_image.png'>Download Image</a>", unsafe_allow_html=True)

                if submit:
                    # Process the image using the main function in the backend
                    st.session_state.binary_images, st.session_state.cluster_labels = main_1(image)

            if len(st.session_state.binary_images) != 0:

                # Convert binary images to PIL format
                pil_image_1 = Image.fromarray(st.session_state.binary_images[0])
                pil_image_2 = Image.fromarray(st.session_state.binary_images[1])
                pil_image_3 = Image.fromarray(st.session_state.binary_images[2])

                # Display the binary images
                st.image(pil_image_1, caption="Image 1", width=350)
                st.image(pil_image_2, caption="Image 2", width=350)
                st.image(pil_image_3, caption="Image 3", width=350)

                st.markdown("### Download Binary Images")

                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path_1 = os.path.join(temp_dir, "binary_image_1.png")
                    file_path_2 = os.path.join(temp_dir, "binary_image_2.png")
                    file_path_3 = os.path.join(temp_dir, "binary_image_3.png")
                    file_path_labels = os.path.join(temp_dir, "cluster_labels.npy")


                    pil_image_1.save(file_path_1)
                    pil_image_2.save(file_path_2)
                    pil_image_3.save(file_path_3)
                    np.save(file_path_labels, st.session_state.cluster_labels)

                    st.markdown(f"<a href='data:file/png;base64,{base64.b64encode(open(file_path_1, 'rb').read()).decode()}' download='Image_1.png'>Download Image 1</a>", unsafe_allow_html=True)
                    st.markdown(f"<a href='data:file/png;base64,{base64.b64encode(open(file_path_2, 'rb').read()).decode()}' download='Image_2.png'>Download Image 2</a>", unsafe_allow_html=True)
                    st.markdown(f"<a href='data:file/png;base64,{base64.b64encode(open(file_path_3, 'rb').read()).decode()}' download='Image_3.png'>Download Image 3</a>", unsafe_allow_html=True)

                    st.markdown("### Combine Image:")

                    labels = ["Image 1", "Image 2", "Image 3"]
                    color_inputs = []
                    for i, label in enumerate(labels):
                        st.write(f"Select Color ({label}):")
                        color_input = st.radio(f"Color {i+1}", options=['R', 'G', 'B'], index=0, key=f"color_{i}")
                        color_inputs.append(color_input)

                    if st.button("Combine Labels"):
                        # Process the image using the main_2 function with the original image, cluster labels, and colors as inputs
                        colors = []
                        for color_input in color_inputs:
                            if color_input == 'R':
                                color = (0, 0, 255)  # R equivalent to (0, 0, 255)
                            elif color_input == 'G':
                                color = (0, 255, 0)  # G equivalent to (0, 255, 0)
                            elif color_input == 'B':
                                color = (255, 0, 0)  # B equivalent to (255, 0, 0)
                            colors.append(color)

                        bgr_image = main_2(image, st.session_state.cluster_labels, colors)

                        rgb_image = bgr_image[:, :, ::-1]

                        rgb_image = rgb_image.astype(np.uint8)

                        pil_rgb_image = Image.fromarray(rgb_image)

                        st.image(pil_rgb_image, caption="Combined RGB Image", width=350)

                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                            pil_rgb_image.save(temp_file.name)

                        st.markdown(f"<a href='data:file/png;base64,{base64.b64encode(open(temp_file.name, 'rb').read()).decode()}' download='combined_image.png'>Download Combined Image</a>", unsafe_allow_html=True)

