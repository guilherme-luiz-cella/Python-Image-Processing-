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
    print_matrix_info
)

st.set_page_config(
    page_title="Image to Matrix Processor",
    page_icon="🖼️",
    layout="wide"
    
)

st.title("🖼️ Image to Matrix Processor")
st.markdown("Upload an image and explore its matrix representation using Pillow library")

def display_matrix_info(matrix, matrix_type, width, height):
    """Display information about the image matrix"""
    st.subheader("Matrix Information")
    
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
        - Mean: {stats['mean']:.2f}
        """)
    else:
        st.markdown(f"""
        **Value Range (RGB):**
        - Red: min={stats['min']['r']}, max={stats['max']['r']}, mean={stats['mean']['r']:.2f}
        - Green: min={stats['min']['g']}, max={stats['max']['g']}, mean={stats['mean']['g']:.2f}
        - Blue: min={stats['min']['b']}, max={stats['max']['b']}, mean={stats['mean']['b']:.2f}
        """)

uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "ppm", "pgm"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        image_matrix, matrix_type = image_to_matrix(tmp_file_path)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            
            pil_image = matrix_to_image(image_matrix, matrix_type)
            st.image(pil_image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            
            height, width = get_matrix_dimensions(image_matrix)
            
            st.write(f"**Image Information:**")
            st.write(f"- Dimensions: {width} × {height}")
            st.write(f"- Mode: {matrix_type}")
            st.write(f"- Format: {uploaded_file.name.split('.')[-1].upper()}")
            
            preview_size = min(10, min(height, width))
            st.write(f"**Pixel Preview ({preview_size}×{preview_size}):**")
            preview_matrix = get_submatrix(image_matrix, 0, 0, preview_size)
            st.dataframe(preview_matrix)
        
        with col2:
            display_matrix_info(image_matrix, matrix_type, width, height)
    
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    