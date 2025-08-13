"""
Image Matrix Functions using Pillow
Provides image loading and matrix operations using PIL/Pillow
"""

from PIL import Image

def load_image(file_path):
    """
    Load an image using Pillow and convert to matrix format
    
    Args:
        file_path: Path to the image file
        
    Returns:
        tuple: (matrix, matrix_type) where matrix_type is 'grayscale' or 'rgb'
    """
    image = Image.open(file_path)
    
    # Convert to RGB if needed (handles RGBA, palette, etc.)
    if image.mode == 'L':
        matrix_type = 'grayscale'
        # Convert to list of lists
        matrix = []
        width, height = image.size
        for y in range(height):
            row = []
            for x in range(width):
                row.append(image.getpixel((x, y)))
            matrix.append(row)
    else:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        matrix_type = 'rgb'
        # Convert to list of lists
        matrix = []
        width, height = image.size
        for y in range(height):
            row = []
            for x in range(width):
                pixel = image.getpixel((x, y))
                row.append(list(pixel))
            matrix.append(row)
    
    return matrix, matrix_type

def image_to_matrix(image_path):
    """
    Load image from file path and convert to matrix
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (matrix, matrix_type) where matrix_type is 'grayscale' or 'rgb'
    """
    return load_image(image_path)

def matrix_to_image(matrix, matrix_type):
    """
    Convert matrix back to PIL Image for display
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        PIL Image object
    """
    height = len(matrix)
    width = len(matrix[0]) if matrix else 0
    
    if matrix_type == 'grayscale':
        image = Image.new('L', (width, height))
        for y in range(height):
            for x in range(width):
                image.putpixel((x, y), int(matrix[y][x]))
    else:  # RGB
        image = Image.new('RGB', (width, height))
        for y in range(height):
            for x in range(width):
                r, g, b = matrix[y][x]
                image.putpixel((x, y), (int(r), int(g), int(b)))
    
    return image


def get_matrix_stats(matrix, matrix_type):
    """
    Calculate statistics for the matrix using pure Python
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        dict: Statistics including min, max, mean values
    """
    if matrix_type == 'grayscale':
        # Flatten the matrix to get all values
        all_values = []
        for row in matrix:
            all_values.extend(row)
        
        min_val = min(all_values)
        max_val = max(all_values)
        mean_val = sum(all_values) / len(all_values)
        
        return {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'total_pixels': len(all_values)
        }
    
    else:  # RGB
        all_r, all_g, all_b = [], [], []
        for row in matrix:
            for pixel in row:
                all_r.append(pixel[0])
                all_g.append(pixel[1])
                all_b.append(pixel[2])
        
        return {
            'min': {
                'r': min(all_r),
                'g': min(all_g),
                'b': min(all_b)
            },
            'max': {
                'r': max(all_r),
                'g': max(all_g),
                'b': max(all_b)
            },
            'mean': {
                'r': sum(all_r) / len(all_r),
                'g': sum(all_g) / len(all_g),
                'b': sum(all_b) / len(all_b)
            },
            'total_pixels': len(all_r)
        }


def extract_channel(matrix, channel):
    """
    Extract a single channel (0=R, 1=G, 2=B) from RGB matrix
    
    Args:
        matrix: RGB matrix (3D list)
        channel: int (0=Red, 1=Green, 2=Blue)
        
    Returns:
        list: 2D matrix with single channel values
    """
    result = []
    for row in matrix:
        new_row = []
        for pixel in row:
            new_row.append(pixel[channel])
        result.append(new_row)
    return result


def get_matrix_dimensions(matrix):
    """
    Get dimensions of the matrix
    
    Args:
        matrix: Python list matrix
        
    Returns:
        tuple: (height, width)
    """
    height = len(matrix)
    width = len(matrix[0]) if matrix else 0
    return height, width


def get_submatrix(matrix, start_row, start_col, size):
    """
    Extract a submatrix from the main matrix
    
    Args:
        matrix: Python list matrix
        start_row: Starting row index
        start_col: Starting column index
        size: Size of the square region to extract
        
    Returns:
        list: Submatrix as Python list
    """
    height, width = get_matrix_dimensions(matrix)
    
    submatrix = []
    for y in range(start_row, min(start_row + size, height)):
        row = []
        for x in range(start_col, min(start_col + size, width)):
            row.append(matrix[y][x])
        submatrix.append(row)
    
    return submatrix

def matrix_to_grayscale(rgb_matrix):
    """
    Convert RGB matrix to grayscale using luminance formula
    Uses: Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        rgb_matrix: 3D RGB matrix
        
    Returns:
        list: 2D grayscale matrix
    """
    grayscale_matrix = []
    for row in rgb_matrix:
        gray_row = []
        for pixel in row:
            r, g, b = pixel
            # Standard luminance formula
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(gray_value)
        grayscale_matrix.append(gray_row)
    
    return grayscale_matrix


def print_matrix_info(matrix, matrix_type):
    """
    Print detailed information about the matrix
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
    """
    height, width = get_matrix_dimensions(matrix)
    stats = get_matrix_stats(matrix, matrix_type)
    
    print(f"Matrix Information:")
    print(f"  Dimensions: {height} x {width}")
    print(f"  Type: {matrix_type}")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    
    if matrix_type == 'grayscale':
        print(f"  Value range: {stats['min']} - {stats['max']}")
        print(f"  Mean value: {stats['mean']:.2f}")
    else:
        print(f"  Red range: {stats['min']['r']} - {stats['max']['r']} (mean: {stats['mean']['r']:.2f})")
        print(f"  Green range: {stats['min']['g']} - {stats['max']['g']} (mean: {stats['mean']['g']:.2f})")
        print(f"  Blue range: {stats['min']['b']} - {stats['max']['b']} (mean: {stats['mean']['b']:.2f})")
