# Image Processing App with Pure Python

A web-based image processing application built with Streamlit, focusing on pure Python implementations without numpy or other image processing libraries.

## Setup Instructions

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## Features

- Upload images in various formats (JPG, PNG, BMP, TIFF)
- Convert images to pure Python list matrices
- Explore matrix properties and statistics
- Extract individual color channels (R, G, B)
- View pixel values in any region
- Export matrices in multiple formats
- **No dependencies** on numpy or image processing libraries

## Project Structure

```
image-processing/
├── app.py                 # Main Streamlit web interface
├── matrix_functions.py    # Pure Python matrix operations
├── example_usage.py       # Example of using matrix functions independently
├── requirements.txt       # Python dependencies (minimal)
└── README.md             # This file
```

## Files Description

### `app.py`

- Streamlit web interface
- Image upload and visualization
- Interactive matrix exploration
- Export functionality

### `matrix_functions.py`

- **Core matrix operations** (pure Python, no numpy)
- Image to matrix conversion
- Statistics calculation
- Channel extraction
- File I/O operations
- Matrix manipulation utilities

### `example_usage.py`

- Standalone example showing how to use matrix functions
- Can be used independently of the Streamlit app
- Perfect for testing and learning

## Available Functions in `matrix_functions.py`

- `image_to_matrix(image)` - Convert PIL image to Python lists
- `get_matrix_stats(matrix, type)` - Calculate min, max, mean
- `extract_channel(matrix, channel)` - Extract R, G, or B channel
- `get_matrix_dimensions(matrix)` - Get height and width
- `get_submatrix(matrix, row, col, size)` - Extract region
- `matrix_to_grayscale(rgb_matrix)` - Convert RGB to grayscale
- `save_matrix_as_python(matrix, filename)` - Export as .py file
- `save_matrix_as_csv(matrix, filename)` - Export as CSV
- `print_matrix_info(matrix, type)` - Print detailed info

## Usage Examples

### Using the Streamlit App

```bash
streamlit run app.py
```

### Using Matrix Functions Independently

```python
from PIL import Image
from matrix_functions import image_to_matrix, print_matrix_info

# Load image and convert to matrix
image = Image.open("your_image.jpg")
matrix, matrix_type = image_to_matrix(image)

# Print information
print_matrix_info(matrix, matrix_type)
```

### Testing with Example Script

```bash
python example_usage.py
```

## Perfect for:

- **Learning image processing** from scratch
- **Implementing custom algorithms** without dependencies
- **Understanding pixel-level** image structure
- **Pure Python implementations** for educational purposes
- **Algorithm development** with full control over data structures

## Next Steps

- Implement image processing algorithms using matrix_functions.py
- Add convolution operations
- Create filtering algorithms (blur, sharpen, edge detection)
- Implement geometric transformations
- Add morphological operations
