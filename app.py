import streamlit as st
import tempfile
import os
from matrix_functions import (
    image_to_matrix, 
    get_matrix_stats, 
    extract_channel, 
    get_matrix_dimensions,
    get_submatrix,
    matrix_to_image,
    print_matrix_info,
    add_pixel_value,
    subtract_pixel_value,
    merge_images
)

st.set_page_config(
    page_title="Image to Matrix Processor",
    page_icon="🖼️",
    layout="wide"
    
)

st.title("🖼️ Dual Image Matrix Processor")
st.markdown("Upload two images and compare their matrix representations using Pillow library")

def display_matrix_info(matrix, matrix_type, image_name):
    """Display information about the image matrix"""
    matrix_height, matrix_width = get_matrix_dimensions(matrix)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dimensions", f"{matrix_height} × {matrix_width}")
    
    with col2:
        st.metric("Type", "List of Lists")
    
    with col3:
        if matrix_type == 'rgb':
            st.metric("Channels", "3 (RGB)")
        else:
            st.metric("Channels", "1 (Grayscale)")
    
    stats = get_matrix_stats(matrix, matrix_type)
    
    with st.expander(f"📊 Detailed Statistics for {image_name}"):
        st.markdown(f"""
        **Dimensions:**
        - Height: {matrix_height} pixels
        - Width: {matrix_width} pixels
        - Total pixels: {stats['total_pixels']:,}
        """)
        
        if matrix_type == 'grayscale':
            st.markdown(f"""
            **Value Range:**
            - Minimum: {stats['min']}
            - Maximum: {stats['max']}
            - Mean: {stats['rmean']:.2f}
            """)
        else:
            st.markdown(f"""
            **Value Range (RGB):**
            - Red: min={stats['min']['r']}, max={stats['max']['r']}, mean={stats['mean']['r']:.2f}
            - Green: min={stats['min']['g']}, max={stats['max']['g']}, mean={stats['mean']['g']:.2f}
            - Blue: min={stats['min']['b']}, max={stats['max']['b']}, mean={stats['mean']['b']:.2f}
            """)

def process_uploaded_image(uploaded_file, image_number):
    """Process a single uploaded image and return its matrix data"""
    if uploaded_file is None:
        return None, None, None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        image_matrix, matrix_type = image_to_matrix(tmp_file_path)
        pil_image = matrix_to_image(image_matrix, matrix_type)
        return image_matrix, matrix_type, pil_image
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Create two columns for image uploaders
st.markdown("### Upload Images")
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown("#### 📸 Image 1")
    uploaded_file1 = st.file_uploader(
        "Choose first image", 
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "ppm", "pgm"],
        key="image1"
    )

with upload_col2:
    st.markdown("#### 📸 Image 2")
    uploaded_file2 = st.file_uploader(
        "Choose second image", 
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "ppm", "pgm"],
        key="image2"
    )

# Process images if they are uploaded
image1_data = process_uploaded_image(uploaded_file1, 1) if uploaded_file1 else (None, None, None)
image2_data = process_uploaded_image(uploaded_file2, 2) if uploaded_file2 else (None, None, None)

# Display images and their information
if uploaded_file1 or uploaded_file2:
    st.markdown("---")
    st.markdown("### 🖼️ Image Display & Analysis")
    
    display_col1, display_col2 = st.columns(2)
    
    # Display Image 1
    with display_col1:
        if uploaded_file1 and image1_data[0] is not None:
            st.subheader("Image 1")
            st.image(image1_data[2], caption=f"Image 1: {uploaded_file1.name}", width=400)
            display_matrix_info(image1_data[0], image1_data[1], "Image 1")
        else:
            st.subheader("Image 1")
            st.info("No image uploaded yet")
    
    # Display Image 2
    with display_col2:
        if uploaded_file2 and image2_data[0] is not None:
            st.subheader("Image 2")
            st.image(image2_data[2], caption=f"Image 2: {uploaded_file2.name}", width=400)
            display_matrix_info(image2_data[0], image2_data[1], "Image 2")
        else:
            st.subheader("Image 2")
            st.info("No image uploaded yet")

# Comparison section
if (uploaded_file1 and image1_data[0] is not None) and (uploaded_file2 and image2_data[0] is not None):
    st.markdown("---")
    st.markdown("### 📊 Image Comparison")
    
    # Get dimensions for comparison
    height1, width1 = get_matrix_dimensions(image1_data[0])
    height2, width2 = get_matrix_dimensions(image2_data[0])
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.metric(
            "Size Difference", 
            f"{abs(height1 * width1 - height2 * width2):,} pixels",
            delta=f"{height1 * width1 - height2 * width2:,}"
        )
    
    with comp_col2:
        type_match = "✅ Same" if image1_data[1] == image2_data[1] else "❌ Different"
        st.metric("Color Type", type_match)
    
    with comp_col3:
        dimension_match = "✅ Same" if (height1, width1) == (height2, width2) else "❌ Different"
        st.metric("Dimensions", dimension_match)
    
    # Detailed comparison
    with st.expander("🔍 Detailed Comparison"):
        comparison_data = []
        comparison_data.append(["Property", "Image 1", "Image 2"])
        comparison_data.append(["Filename", uploaded_file1.name, uploaded_file2.name])
        comparison_data.append(["Dimensions", f"{height1} × {width1}", f"{height2} × {width2}"])
        comparison_data.append(["Total Pixels", f"{height1 * width1:,}", f"{height2 * width2:,}"])
        comparison_data.append(["Color Type", image1_data[1].title(), image2_data[1].title()])
        
        # Create a simple table
        for row in comparison_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{row[0]}**")
            with col2:
                st.write(row[1])
            with col3:
                st.write(row[2])

# Image Processing Section
if uploaded_file1 and image1_data[0] is not None:
    st.markdown("---")
    st.markdown("### 🔧 Image Processing")
    
    # Create tabs for different processing options
    tab1, tab2, tab3 = st.tabs(["➕ Add/Subtract Pixels", "🔀 Merge Images", "📊 Results"])
    
    with tab1:
        st.markdown("#### Adjust Pixel Values")
        
        # Processing controls for each image
        proc_col1, proc_col2 = st.columns(2)
        
        with proc_col1:
            st.markdown("**Image 1 Processing**")
            operation1 = st.selectbox("Operation for Image 1:", ["Add", "Subtract"], key="op1")
            value1 = st.slider("Value for Image 1:", min_value=0, max_value=255, value=50, key="val1")
            
            if st.button("Process Image 1", key="process1"):
                if operation1 == "Add":
                    processed_matrix1 = add_pixel_value(image1_data[0], image1_data[1], value1)
                else:
                    processed_matrix1 = subtract_pixel_value(image1_data[0], image1_data[1], value1)
                
                processed_image1 = matrix_to_image(processed_matrix1, image1_data[1])
                st.session_state.processed1 = (processed_matrix1, image1_data[1], processed_image1)
                st.success(f"Applied {operation1.lower()} {value1} to Image 1")
        
        with proc_col2:
            if uploaded_file2 and image2_data[0] is not None:
                st.markdown("**Image 2 Processing**")
                operation2 = st.selectbox("Operation for Image 2:", ["Add", "Subtract"], key="op2")
                value2 = st.slider("Value for Image 2:", min_value=0, max_value=255, value=50, key="val2")
                
                if st.button("Process Image 2", key="process2"):
                    if operation2 == "Add":
                        processed_matrix2 = add_pixel_value(image2_data[0], image2_data[1], value2)
                    else:
                        processed_matrix2 = subtract_pixel_value(image2_data[0], image2_data[1], value2)
                    
                    processed_image2 = matrix_to_image(processed_matrix2, image2_data[1])
                    st.session_state.processed2 = (processed_matrix2, image2_data[1], processed_image2)
                    st.success(f"Applied {operation2.lower()} {value2} to Image 2")
            else:
                st.info("Upload Image 2 to enable processing")
    
    with tab2:
        if uploaded_file2 and image2_data[0] is not None:
            st.markdown("#### Merge Two Images")
            
            # Check if images have same dimensions
            height1, width1 = get_matrix_dimensions(image1_data[0])
            height2, width2 = get_matrix_dimensions(image2_data[0])
            
            if (height1, width1) == (height2, width2):
                merge_operation = st.selectbox("Merge Operation:", ["Add Images", "Subtract Images"])
                
                if st.button("Merge Images"):
                    operation = 'add' if merge_operation == "Add Images" else 'subtract'
                    merged_matrix, merged_type = merge_images(
                        image1_data[0], image1_data[1], 
                        image2_data[0], image2_data[1], 
                        operation
                    )
                    
                    if merged_matrix is not None:
                        merged_image = matrix_to_image(merged_matrix, merged_type)
                        st.session_state.merged = (merged_matrix, merged_type, merged_image)
                        st.success(f"Successfully merged images using {operation}")
                    else:
                        st.error("Failed to merge images")
            else:
                st.warning(f"⚠️ Images must have the same dimensions to merge!")
                st.info(f"Image 1: {height1}×{width1}, Image 2: {height2}×{width2}")
        else:
            st.info("Upload both images to enable merging")
    
    with tab3:
        st.markdown("#### Processing Results")
        
        # Display processed images
        result_cols = st.columns(3)
        
        with result_cols[0]:
            if hasattr(st.session_state, 'processed1'):
                st.markdown("**Processed Image 1**")
                st.image(st.session_state.processed1[2], caption="Processed Image 1", width=300)
            else:
                st.info("No processed Image 1 yet")
        
        with result_cols[1]:
            if hasattr(st.session_state, 'processed2'):
                st.markdown("**Processed Image 2**")
                st.image(st.session_state.processed2[2], caption="Processed Image 2", width=300)
            else:
                st.info("No processed Image 2 yet")
        
        with result_cols[2]:
            if hasattr(st.session_state, 'merged'):
                st.markdown("**Merged Image**")
                st.image(st.session_state.merged[2], caption="Merged Image", width=300)
            else:
                st.info("No merged image yet")
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            for key in ['processed1', 'processed2', 'merged']:
                if hasattr(st.session_state, key):
                    delattr(st.session_state, key)
            st.success("All processing results cleared!")
            st.rerun()
    