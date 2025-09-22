import streamlit as st
import tempfile
import os
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from matrix_functions import (
    image_to_matrix, 
    get_matrix_stats, 
    extract_channel, 
    get_matrix_dimensions,
    get_submatrix,
    matrix_to_image,
    save_matrix_as_image,
    print_matrix_info,
    add_pixel_value,
    subtract_pixel_value,
    multiply_pixel_value,
    divide_pixel_value,
    merge_images,
    image_difference,
    linear_combination,
    average_images,
    flip_horizontal,
    flip_vertical,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    threshold_image,
    negative_image,
    calculate_histogram,
    equalize_histogram,
    matrix_to_grayscale
)

st.set_page_config(
    page_title="Complete Image Processing Suite",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Complete Image Processing Suite")
st.markdown("Professional image processing with matrix operations using Pillow library")

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
            - Mean: {stats['mean']:.2f}
            """)
        else:
            st.markdown(f"""
            **Value Range (RGB):**
            - Red: min={stats['min']['r']}, max={stats['max']['r']}, mean={stats['mean']['r']:.2f}
            - Green: min={stats['min']['g']}, max={stats['max']['g']}, mean={stats['mean']['g']:.2f}
            - Blue: min={stats['min']['b']}, max={stats['max']['b']}, mean={stats['mean']['b']:.2f}
            """)

def plot_histogram(matrix, matrix_type, title="Image Histogram"):
    """Plot histogram using Plotly"""
    hist_data = calculate_histogram(matrix, matrix_type)
    
    fig = go.Figure()
    
    if matrix_type == 'grayscale':
        x_values = list(range(256))
        fig.add_trace(go.Scatter(x=x_values, y=hist_data['grayscale'], 
                                mode='lines', name='Grayscale', 
                                line=dict(color='black')))
    else:
        x_values = list(range(256))
        fig.add_trace(go.Scatter(x=x_values, y=hist_data['red'], 
                                mode='lines', name='Red', 
                                line=dict(color='red')))
        fig.add_trace(go.Scatter(x=x_values, y=hist_data['green'], 
                                mode='lines', name='Green', 
                                line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x_values, y=hist_data['blue'], 
                                mode='lines', name='Blue', 
                                line=dict(color='blue')))
    
    fig.update_layout(
        title=title,
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency",
        showlegend=True
    )
    
    return fig

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

def download_image(matrix, matrix_type, filename):
    """Create download button for processed image"""
    pil_image = matrix_to_image(matrix, matrix_type)
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=img_bytes,
        file_name=f"{filename}.png",
        mime="image/png"
    )

# Create two columns for image uploaders
st.markdown("### 📤 Upload Images")
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown("#### 📸 Image 1")
    uploaded_file1 = st.file_uploader(
        "Choose first image (supports BMP, JPG, PNG)", 
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif"],
        key="image1"
    )

with upload_col2:
    st.markdown("#### 📸 Image 2")
    uploaded_file2 = st.file_uploader(
        "Choose second image (supports BMP, JPG, PNG)", 
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif"],
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

# Main processing section
if uploaded_file1 and image1_data[0] is not None:
    st.markdown("---")
    st.markdown("### 🔧 Image Processing Operations")
    
    # Create tabs for different processing categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎨 Basic Operations", 
        "➕➖ Arithmetic", 
        "🔀 Two-Image Ops",
        "↔️ Geometric",
        "⚫⚪ Binary/Logic", 
        "📊 Histogram",
        "💾 Results"
    ])
    
    with tab1:
        st.markdown("#### Basic Image Processing")
        
        basic_col1, basic_col2 = st.columns(2)
        
        with basic_col1:
            st.markdown("**Single Value Operations**")
            
            # Brightness adjustment (add/subtract)
            brightness_op = st.selectbox("Brightness Operation:", ["Add Value", "Subtract Value"])
            brightness_val = st.slider("Brightness Value:", 0, 100, 25)
            
            if st.button("Apply Brightness"):
                if brightness_op == "Add Value":
                    result = add_pixel_value(image1_data[0], image1_data[1], brightness_val)
                else:
                    result = subtract_pixel_value(image1_data[0], image1_data[1], brightness_val)
                st.session_state.brightness_result = (result, image1_data[1])
                st.success(f"✅ Applied brightness operation")
            
            # Contrast adjustment (multiply/divide)
            st.markdown("**Contrast Operations**")
            contrast_op = st.selectbox("Contrast Operation:", ["Multiply", "Divide"])
            contrast_val = st.slider("Contrast Factor:", 0.1, 3.0, 1.0, 0.1)
            
            if st.button("Apply Contrast"):
                try:
                    if contrast_op == "Multiply":
                        result = multiply_pixel_value(image1_data[0], image1_data[1], contrast_val)
                    else:
                        result = divide_pixel_value(image1_data[0], image1_data[1], contrast_val)
                    st.session_state.contrast_result = (result, image1_data[1])
                    st.success(f"✅ Applied contrast operation")
                except ValueError as e:
                    st.error(f"❌ Error: {e}")
        
        with basic_col2:
            st.markdown("**Color Space Operations**")
            
            # RGB to Grayscale conversion
            if image1_data[1] == 'rgb':
                if st.button("Convert to Grayscale"):
                    grayscale_matrix = matrix_to_grayscale(image1_data[0])
                    st.session_state.grayscale_result = (grayscale_matrix, 'grayscale')
                    st.success("✅ Converted to grayscale")
            else:
                st.info("Image is already grayscale")
            
            # Negative operation
            if st.button("Create Negative"):
                negative_matrix = negative_image(image1_data[0], image1_data[1])
                st.session_state.negative_result = (negative_matrix, image1_data[1])
                st.success("✅ Created negative image")
            
            # Thresholding
            if st.button("Apply Basic Threshold (127)"):
                thresh_matrix, thresh_type = threshold_image(image1_data[0], image1_data[1], 127)
                st.session_state.threshold_result = (thresh_matrix, thresh_type)
                st.success("✅ Applied thresholding")
    
    with tab2:
        st.markdown("#### Arithmetic Operations")
        
        if uploaded_file2 and image2_data[0] is not None:
            # Check dimensions
            height1, width1 = get_matrix_dimensions(image1_data[0])
            height2, width2 = get_matrix_dimensions(image2_data[0])
            
            if (height1, width1) == (height2, width2):
                arith_col1, arith_col2 = st.columns(2)
                
                with arith_col1:
                    st.markdown("**Basic Arithmetic**")
                    
                    if st.button("Add Images"):
                        result, result_type = merge_images(image1_data[0], image1_data[1], 
                                                         image2_data[0], image2_data[1], 'add')
                        st.session_state.add_result = (result, result_type)
                        st.success("✅ Images added")
                    
                    if st.button("Subtract Images"):
                        result, result_type = merge_images(image1_data[0], image1_data[1], 
                                                         image2_data[0], image2_data[1], 'subtract')
                        st.session_state.subtract_result = (result, result_type)
                        st.success("✅ Images subtracted")
                    
                    if st.button("Image Difference"):
                        result, result_type = image_difference(image1_data[0], image1_data[1], 
                                                             image2_data[0], image2_data[1])
                        st.session_state.difference_result = (result, result_type)
                        st.success("✅ Difference calculated")
                
                with arith_col2:
                    st.markdown("**Linear Combinations**")
                    
                    alpha = st.slider("Weight for Image 1 (α):", 0.0, 1.0, 0.5, 0.1)
                    beta = st.slider("Weight for Image 2 (β):", 0.0, 1.0, 0.5, 0.1)
                    
                    if st.button("Linear Blend"):
                        result, result_type = linear_combination(image1_data[0], image1_data[1], 
                                                               image2_data[0], image2_data[1], alpha, beta)
                        st.session_state.blend_result = (result, result_type)
                        st.success("✅ Linear blend applied")
                    
                    if st.button("Average Images"):
                        result, result_type = average_images(image1_data[0], image1_data[1], 
                                                           image2_data[0], image2_data[1])
                        st.session_state.average_result = (result, result_type)
                        st.success("✅ Images averaged")
            else:
                st.warning(f"⚠️ Images must have same dimensions! Image1: {height1}×{width1}, Image2: {height2}×{width2}")
        else:
            st.info("Upload second image for arithmetic operations")
    
    with tab3:
        st.markdown("#### Two-Image Operations")
        
        if uploaded_file2 and image2_data[0] is not None:
            # More complex operations would go here
            st.info("Advanced two-image operations panel - extend as needed")
        else:
            st.info("Upload second image for two-image operations")
    
    with tab4:
        st.markdown("#### Geometric Transformations")
        
        geo_col1, geo_col2 = st.columns(2)
        
        with geo_col1:
            st.markdown("**Flipping Operations**")
            
            if st.button("Flip Horizontal (Left ↔ Right)"):
                flipped_matrix = flip_horizontal(image1_data[0], image1_data[1])
                st.session_state.flip_h_result = (flipped_matrix, image1_data[1])
                st.success("✅ Flipped horizontally")
            
            if st.button("Flip Vertical (Top ↔ Bottom)"):
                flipped_matrix = flip_vertical(image1_data[0], image1_data[1])
                st.session_state.flip_v_result = (flipped_matrix, image1_data[1])
                st.success("✅ Flipped vertically")
        
        with geo_col2:
            st.markdown("**Additional Transforms**")
            st.info("Rotation and scaling can be added here")
    
    with tab5:
        st.markdown("#### Binary and Logical Operations")
        
        logic_col1, logic_col2 = st.columns(2)
        
        with logic_col1:
            st.markdown("**Thresholding**")
            threshold_val = st.slider("Threshold Value:", 0, 255, 127)
            
            if st.button("Apply Custom Threshold"):
                thresh_matrix, thresh_type = threshold_image(image1_data[0], image1_data[1], threshold_val)
                st.session_state.custom_threshold = (thresh_matrix, thresh_type)
                st.success(f"✅ Threshold applied at {threshold_val}")
            
            # NOT operation
            if st.button("Logical NOT"):
                # First convert to grayscale if RGB
                if image1_data[1] == 'rgb':
                    gray_matrix = matrix_to_grayscale(image1_data[0])
                else:
                    gray_matrix = image1_data[0]
                
                not_result = logical_not(gray_matrix)
                st.session_state.not_result = (not_result, 'grayscale')
                st.success("✅ Logical NOT applied")
        
        with logic_col2:
            if uploaded_file2 and image2_data[0] is not None:
                st.markdown("**Two-Image Logic Operations**")
                
                # Convert both images to grayscale for logical operations
                if st.button("Logical AND"):
                    gray1 = matrix_to_grayscale(image1_data[0]) if image1_data[1] == 'rgb' else image1_data[0]
                    gray2 = matrix_to_grayscale(image2_data[0]) if image2_data[1] == 'rgb' else image2_data[0]
                    
                    and_result = logical_and(gray1, gray2)
                    if and_result:
                        st.session_state.and_result = (and_result, 'grayscale')
                        st.success("✅ Logical AND applied")
                    else:
                        st.error("❌ Images must have same dimensions")
                
                if st.button("Logical OR"):
                    gray1 = matrix_to_grayscale(image1_data[0]) if image1_data[1] == 'rgb' else image1_data[0]
                    gray2 = matrix_to_grayscale(image2_data[0]) if image2_data[1] == 'rgb' else image2_data[0]
                    
                    or_result = logical_or(gray1, gray2)
                    if or_result:
                        st.session_state.or_result = (or_result, 'grayscale')
                        st.success("✅ Logical OR applied")
                    else:
                        st.error("❌ Images must have same dimensions")
                
                if st.button("Logical XOR"):
                    gray1 = matrix_to_grayscale(image1_data[0]) if image1_data[1] == 'rgb' else image1_data[0]
                    gray2 = matrix_to_grayscale(image2_data[0]) if image2_data[1] == 'rgb' else image2_data[0]
                    
                    xor_result = logical_xor(gray1, gray2)
                    if xor_result:
                        st.session_state.xor_result = (xor_result, 'grayscale')
                        st.success("✅ Logical XOR applied")
                    else:
                        st.error("❌ Images must have same dimensions")
            else:
                st.info("Upload second image for two-image logical operations")
    
    with tab6:
        st.markdown("#### Histogram Operations")
        
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            st.markdown("**Original Image Histogram**")
            if st.button("Show Histogram"):
                fig = plot_histogram(image1_data[0], image1_data[1], "Original Image Histogram")
                st.plotly_chart(fig, use_container_width=True)
        
        with hist_col2:
            st.markdown("**Histogram Equalization**")
            if st.button("Equalize Histogram"):
                # Convert to grayscale if RGB
                if image1_data[1] == 'rgb':
                    equalized_matrix = equalize_histogram(image1_data[0], 'rgb')
                else:
                    equalized_matrix = equalize_histogram(image1_data[0], 'grayscale')
                
                st.session_state.equalized_result = (equalized_matrix, 'grayscale')
                st.success("✅ Histogram equalized")
                
                # Show equalized histogram
                fig = plot_histogram(equalized_matrix, 'grayscale', "Equalized Histogram")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.markdown("#### Processing Results")
        st.markdown("**Click on any image to view full size**")
        
        # Display all results in a grid
        results = [
            ('brightness_result', 'Brightness Adjusted'),
            ('contrast_result', 'Contrast Adjusted'),
            ('grayscale_result', 'Grayscale'),
            ('negative_result', 'Negative'),
            ('threshold_result', 'Thresholded'),
            ('add_result', 'Images Added'),
            ('subtract_result', 'Images Subtracted'),
            ('difference_result', 'Image Difference'),
            ('blend_result', 'Linear Blend'),
            ('average_result', 'Images Averaged'),
            ('flip_h_result', 'Flipped Horizontal'),
            ('flip_v_result', 'Flipped Vertical'),
            ('custom_threshold', 'Custom Threshold'),
            ('not_result', 'Logical NOT'),
            ('and_result', 'Logical AND'),
            ('or_result', 'Logical OR'),
            ('xor_result', 'Logical XOR'),
            ('equalized_result', 'Histogram Equalized')
        ]
        
        # Display results in rows of 3
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(results):
                    result_key, result_name = results[i + j]
                    with cols[j]:
                        if hasattr(st.session_state, result_key):
                            result_matrix, result_type = getattr(st.session_state, result_key)
                            result_image = matrix_to_image(result_matrix, result_type)
                            st.image(result_image, caption=result_name, width=200)
                            download_image(result_matrix, result_type, result_name.replace(' ', '_').lower())
                        else:
                            st.info(f"No {result_name.lower()} result")
        
        # Clear all results
        st.markdown("---")
        if st.button("🗑️ Clear All Results", type="primary"):
            for result_key, _ in results:
                if hasattr(st.session_state, result_key):
                    delattr(st.session_state, result_key)
            st.success("All results cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
### 📋 Implemented Features:
1. ✅ Read BMP, JPG, PNG images and store as matrices
2. ✅ Add/Subtract constant values (brightness with overflow/underflow handling)
3. ✅ Multiply/Divide by constants (contrast with overflow/underflow handling)  
4. ✅ Add/Subtract two images
5. ✅ Image difference calculation
6. ✅ Linear combination (blending)
7. ✅ Average of two images
8. ✅ RGB to Grayscale conversion
9. ✅ Horizontal and vertical flipping
10. ✅ Logical operations (AND, OR, NOT, XOR) for binary images
11. ✅ Image thresholding
12. ✅ Image negative
13. ✅ Histogram calculation and equalization
14. ✅ Save processed images
15. ✅ Display results in application interface

**All operations handle overflow/underflow by clamping values between 0-255**
""")
