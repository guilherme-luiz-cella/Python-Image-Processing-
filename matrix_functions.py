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


def add_pixel_value(matrix, matrix_type, value):
    """
    Add a constant value to all pixels in the matrix
    Clamps values between 0 and 255
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        value: Integer value to add to each pixel
        
    Returns:
        list: New matrix with adjusted pixel values
    """
    result_matrix = []
    
    if matrix_type == 'grayscale':
        for row in matrix:
            new_row = []
            for pixel in row:
                new_value = pixel + value
                # Clamp between 0 and 255
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for row in matrix:
            new_row = []
            for pixel in row:
                r, g, b = pixel
                new_r = max(0, min(255, r + value))
                new_g = max(0, min(255, g + value))
                new_b = max(0, min(255, b + value))
                new_row.append([new_r, new_g, new_b])
            result_matrix.append(new_row)
    
    return result_matrix

def subtract_pixel_value(matrix, matrix_type, value):
    """
    Subtract a constant value from all pixels in the matrix
    Clamps values between 0 and 255
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        value: Integer value to subtract from each pixel
        
    Returns:
        list: New matrix with adjusted pixel values
    """
    result_matrix = []
    
    if matrix_type == 'grayscale':
        for row in matrix:
            new_row = []
            for pixel in row:
                new_value = pixel - value
                # Clamp between 0 and 255
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for row in matrix:
            new_row = []
            for pixel in row:
                r, g, b = pixel
                new_r = max(0, min(255, r - value))
                new_g = max(0, min(255, g - value))
                new_b = max(0, min(255, b - value))
                new_row.append([new_r, new_g, new_b])
            result_matrix.append(new_row)
    
    return result_matrix

def merge_images(matrix1, matrix_type1, matrix2, matrix_type2, operation='add'):
    """
    Merge two images using pixel arithmetic
    Images must have the same dimensions
    
    Args:
        matrix1: First image matrix
        matrix_type1: Type of first matrix ('grayscale' or 'rgb')
        matrix2: Second image matrix
        matrix_type2: Type of second matrix ('grayscale' or 'rgb')
        operation: 'add' or 'subtract'
        
    Returns:
        tuple: (result_matrix, result_type) or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    # Check if dimensions match
    if height1 != height2 or width1 != width2:
        return None, None
    
    # Convert both to same type (prefer RGB if one is RGB)
    if matrix_type1 != matrix_type2:
        if matrix_type1 == 'rgb' or matrix_type2 == 'rgb':
            # Convert grayscale to RGB
            if matrix_type1 == 'grayscale':
                matrix1 = grayscale_to_rgb(matrix1)
                matrix_type1 = 'rgb'
            if matrix_type2 == 'grayscale':
                matrix2 = grayscale_to_rgb(matrix2)
                matrix_type2 = 'rgb'
    
    result_matrix = []
    result_type = matrix_type1  # Both are same type now
    
    if result_type == 'grayscale':
        for y in range(height1):
            new_row = []
            for x in range(width1):
                pixel1 = matrix1[y][x]
                pixel2 = matrix2[y][x]
                
                if operation == 'add':
                    new_value = pixel1 + pixel2
                else:  # subtract
                    new_value = pixel1 - pixel2
                
                # Clamp between 0 and 255
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for y in range(height1):
            new_row = []
            for x in range(width1):
                r1, g1, b1 = matrix1[y][x]
                r2, g2, b2 = matrix2[y][x]
                
                if operation == 'add':
                    new_r = r1 + r2
                    new_g = g1 + g2
                    new_b = b1 + b2
                else:  # subtract
                    new_r = r1 - r2
                    new_g = g1 - g2
                    new_b = b1 - b2
                
                # Clamp between 0 and 255
                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                
                new_row.append([new_r, new_g, new_b])
            result_matrix.append(new_row)
    
    return result_matrix, result_type

def grayscale_to_rgb(grayscale_matrix):
    """
    Convert grayscale matrix to RGB by duplicating values
    
    Args:
        grayscale_matrix: 2D grayscale matrix
        
    Returns:
        list: 3D RGB matrix
    """
    rgb_matrix = []
    for row in grayscale_matrix:
        new_row = []
        for pixel in row:
            # Convert grayscale value to RGB by duplicating
            new_row.append([pixel, pixel, pixel])
        rgb_matrix.append(new_row)
    return rgb_matrix

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
