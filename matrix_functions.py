"""
Enhanced Image Matrix Functions using Pillow
Provides comprehensive image loading and matrix operations using PIL/Pillow
Includes all required image processing techniques
"""

from PIL import Image
import math

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

def save_matrix_as_image(matrix, matrix_type, file_path):
    """
    Save matrix as image file
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        file_path: Path to save the image
    """
    image = matrix_to_image(matrix, matrix_type)
    image.save(file_path)

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
    Clamps values between 0 and 255 to handle OVERFLOW
    
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
                # Clamp between 0 and 255 (handle overflow)
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
    Clamps values between 0 and 255 to handle UNDERFLOW
    
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
                # Clamp between 0 and 255 (handle underflow)
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

def multiply_pixel_value(matrix, matrix_type, value):
    """
    Multiply all pixels by a constant value (contrast adjustment)
    Handles OVERFLOW and UNDERFLOW by clamping between 0 and 255
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        value: Float value to multiply each pixel
        
    Returns:
        list: New matrix with adjusted pixel values
    """
    result_matrix = []
    
    if matrix_type == 'grayscale':
        for row in matrix:
            new_row = []
            for pixel in row:
                new_value = int(pixel * value)
                # Clamp between 0 and 255
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for row in matrix:
            new_row = []
            for pixel in row:
                r, g, b = pixel
                new_r = max(0, min(255, int(r * value)))
                new_g = max(0, min(255, int(g * value)))
                new_b = max(0, min(255, int(b * value)))
                new_row.append([new_r, new_g, new_b])
            result_matrix.append(new_row)
    
    return result_matrix

def divide_pixel_value(matrix, matrix_type, value):
    """
    Divide all pixels by a constant value (contrast adjustment)
    Handles division by zero and clamps between 0 and 255
    
    Args:
        matrix: Python list matrix
        matrix_type: 'grayscale' or 'rgb'
        value: Float value to divide each pixel (must not be 0)
        
    Returns:
        list: New matrix with adjusted pixel values
    """
    if value == 0:
        raise ValueError("Cannot divide by zero")
    
    result_matrix = []
    
    if matrix_type == 'grayscale':
        for row in matrix:
            new_row = []
            for pixel in row:
                new_value = int(pixel / value)
                # Clamp between 0 and 255
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for row in matrix:
            new_row = []
            for pixel in row:
                r, g, b = pixel
                new_r = max(0, min(255, int(r / value)))
                new_g = max(0, min(255, int(g / value)))
                new_b = max(0, min(255, int(b / value)))
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

def image_difference(matrix1, matrix_type1, matrix2, matrix_type2):
    """
    Calculate absolute difference between two images
    
    Args:
        matrix1: First image matrix
        matrix_type1: Type of first matrix ('grayscale' or 'rgb')
        matrix2: Second image matrix
        matrix_type2: Type of second matrix ('grayscale' or 'rgb')
        
    Returns:
        tuple: (result_matrix, result_type) or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    if height1 != height2 or width1 != width2:
        return None, None
    
    # Convert both to same type (prefer RGB if one is RGB)
    if matrix_type1 != matrix_type2:
        if matrix_type1 == 'rgb' or matrix_type2 == 'rgb':
            if matrix_type1 == 'grayscale':
                matrix1 = grayscale_to_rgb(matrix1)
                matrix_type1 = 'rgb'
            if matrix_type2 == 'grayscale':
                matrix2 = grayscale_to_rgb(matrix2)
                matrix_type2 = 'rgb'
    
    result_matrix = []
    result_type = matrix_type1
    
    if result_type == 'grayscale':
        for y in range(height1):
            new_row = []
            for x in range(width1):
                pixel1 = matrix1[y][x]
                pixel2 = matrix2[y][x]
                diff = abs(pixel1 - pixel2)
                new_row.append(diff)
            result_matrix.append(new_row)
    else:  # RGB
        for y in range(height1):
            new_row = []
            for x in range(width1):
                r1, g1, b1 = matrix1[y][x]
                r2, g2, b2 = matrix2[y][x]
                
                diff_r = abs(r1 - r2)
                diff_g = abs(g1 - g2)
                diff_b = abs(b1 - b2)
                
                new_row.append([diff_r, diff_g, diff_b])
            result_matrix.append(new_row)
    
    return result_matrix, result_type

def linear_combination(matrix1, matrix_type1, matrix2, matrix_type2, alpha=0.5, beta=0.5):
    """
    Linear combination (blending) of two images
    Result = alpha * image1 + beta * image2
    
    Args:
        matrix1: First image matrix
        matrix_type1: Type of first matrix
        matrix2: Second image matrix
        matrix_type2: Type of second matrix
        alpha: Weight for first image (default 0.5)
        beta: Weight for second image (default 0.5)
        
    Returns:
        tuple: (result_matrix, result_type) or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    if height1 != height2 or width1 != width2:
        return None, None
    
    # Convert both to same type
    if matrix_type1 != matrix_type2:
        if matrix_type1 == 'rgb' or matrix_type2 == 'rgb':
            if matrix_type1 == 'grayscale':
                matrix1 = grayscale_to_rgb(matrix1)
                matrix_type1 = 'rgb'
            if matrix_type2 == 'grayscale':
                matrix2 = grayscale_to_rgb(matrix2)
                matrix_type2 = 'rgb'
    
    result_matrix = []
    result_type = matrix_type1
    
    if result_type == 'grayscale':
        for y in range(height1):
            new_row = []
            for x in range(width1):
                pixel1 = matrix1[y][x]
                pixel2 = matrix2[y][x]
                new_value = int(alpha * pixel1 + beta * pixel2)
                new_value = max(0, min(255, new_value))
                new_row.append(new_value)
            result_matrix.append(new_row)
    else:  # RGB
        for y in range(height1):
            new_row = []
            for x in range(width1):
                r1, g1, b1 = matrix1[y][x]
                r2, g2, b2 = matrix2[y][x]
                
                new_r = int(alpha * r1 + beta * r2)
                new_g = int(alpha * g1 + beta * g2)
                new_b = int(alpha * b1 + beta * b2)
                
                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                
                new_row.append([new_r, new_g, new_b])
            result_matrix.append(new_row)
    
    return result_matrix, result_type

def average_images(matrix1, matrix_type1, matrix2, matrix_type2):
    """
    Calculate average of two images
    
    Args:
        matrix1: First image matrix
        matrix_type1: Type of first matrix
        matrix2: Second image matrix
        matrix_type2: Type of second matrix
        
    Returns:
        tuple: (result_matrix, result_type) or None if incompatible
    """
    return linear_combination(matrix1, matrix_type1, matrix2, matrix_type2, 0.5, 0.5)

def flip_horizontal(matrix, matrix_type):
    """
    Flip image horizontally (left to right)
    
    Args:
        matrix: Image matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        list: Horizontally flipped matrix
    """
    result_matrix = []
    for row in matrix:
        # Reverse each row
        result_matrix.append(row[::-1])
    return result_matrix

def flip_vertical(matrix, matrix_type):
    """
    Flip image vertically (top to bottom)
    
    Args:
        matrix: Image matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        list: Vertically flipped matrix
    """
    # Reverse the order of rows
    return matrix[::-1]

def logical_and(matrix1, matrix2):
    """
    Logical AND operation on two binary images
    
    Args:
        matrix1: First binary image matrix
        matrix2: Second binary image matrix
        
    Returns:
        list: Result matrix or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    if height1 != height2 or width1 != width2:
        return None
    
    result_matrix = []
    for y in range(height1):
        new_row = []
        for x in range(width1):
            # Convert to binary (0 or 255) then perform AND
            pixel1 = 255 if matrix1[y][x] > 127 else 0
            pixel2 = 255 if matrix2[y][x] > 127 else 0
            result = 255 if (pixel1 == 255 and pixel2 == 255) else 0
            new_row.append(result)
        result_matrix.append(new_row)
    
    return result_matrix

def logical_or(matrix1, matrix2):
    """
    Logical OR operation on two binary images
    
    Args:
        matrix1: First binary image matrix
        matrix2: Second binary image matrix
        
    Returns:
        list: Result matrix or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    if height1 != height2 or width1 != width2:
        return None
    
    result_matrix = []
    for y in range(height1):
        new_row = []
        for x in range(width1):
            pixel1 = 255 if matrix1[y][x] > 127 else 0
            pixel2 = 255 if matrix2[y][x] > 127 else 0
            result = 255 if (pixel1 == 255 or pixel2 == 255) else 0
            new_row.append(result)
        result_matrix.append(new_row)
    
    return result_matrix

def logical_not(matrix):
    """
    Logical NOT operation on a binary image
    
    Args:
        matrix: Binary image matrix
        
    Returns:
        list: Result matrix
    """
    result_matrix = []
    for row in matrix:
        new_row = []
        for pixel in row:
            binary_pixel = 255 if pixel > 127 else 0
            result = 0 if binary_pixel == 255 else 255
            new_row.append(result)
        result_matrix.append(new_row)
    
    return result_matrix

def logical_xor(matrix1, matrix2):
    """
    Logical XOR operation on two binary images
    
    Args:
        matrix1: First binary image matrix
        matrix2: Second binary image matrix
        
    Returns:
        list: Result matrix or None if incompatible
    """
    height1, width1 = get_matrix_dimensions(matrix1)
    height2, width2 = get_matrix_dimensions(matrix2)
    
    if height1 != height2 or width1 != width2:
        return None
    
    result_matrix = []
    for y in range(height1):
        new_row = []
        for x in range(width1):
            pixel1 = 255 if matrix1[y][x] > 127 else 0
            pixel2 = 255 if matrix2[y][x] > 127 else 0
            result = 255 if (pixel1 == 255) != (pixel2 == 255) else 0
            new_row.append(result)
        result_matrix.append(new_row)
    
    return result_matrix

def threshold_image(matrix, matrix_type, threshold=127):
    """
    Apply thresholding to create binary image
    
    Args:
        matrix: Image matrix
        matrix_type: 'grayscale' or 'rgb'
        threshold: Threshold value (0-255)
        
    Returns:
        tuple: (binary_matrix, 'grayscale')
    """
    if matrix_type == 'rgb':
        # Convert to grayscale first
        matrix = matrix_to_grayscale(matrix)
    
    result_matrix = []
    for row in matrix:
        new_row = []
        for pixel in row:
            binary_value = 255 if pixel > threshold else 0
            new_row.append(binary_value)
        result_matrix.append(new_row)
    
    return result_matrix, 'grayscale'

def negative_image(matrix, matrix_type):
    """
    Create negative of the image
    
    Args:
        matrix: Image matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        list: Negative image matrix
    """
    result_matrix = []
    
    if matrix_type == 'grayscale':
        for row in matrix:
            new_row = []
            for pixel in row:
                new_row.append(255 - pixel)
            result_matrix.append(new_row)
    else:  # RGB
        for row in matrix:
            new_row = []
            for pixel in row:
                r, g, b = pixel
                new_row.append([255 - r, 255 - g, 255 - b])
            result_matrix.append(new_row)
    
    return result_matrix

def calculate_histogram(matrix, matrix_type):
    """
    Calculate histogram of image
    
    Args:
        matrix: Image matrix
        matrix_type: 'grayscale' or 'rgb'
        
    Returns:
        dict: Histogram data
    """
    if matrix_type == 'grayscale':
        histogram = [0] * 256
        for row in matrix:
            for pixel in row:
                histogram[pixel] += 1
        return {'grayscale': histogram}
    else:  # RGB
        hist_r = [0] * 256
        hist_g = [0] * 256
        hist_b = [0] * 256
        
        for row in matrix:
            for pixel in row:
                r, g, b = pixel
                hist_r[r] += 1
                hist_g[g] += 1
                hist_b[b] += 1
        
        return {'red': hist_r, 'green': hist_g, 'blue': hist_b}

def equalize_histogram(matrix, matrix_type):
    """
    Perform histogram equalization
    
    Args:
        matrix: Image matrix (grayscale only)
        matrix_type: Must be 'grayscale'
        
    Returns:
        list: Equalized image matrix
    """
    if matrix_type != 'grayscale':
        # Convert RGB to grayscale first
        matrix = matrix_to_grayscale(matrix)
    
    # Calculate histogram
    histogram = [0] * 256
    height, width = get_matrix_dimensions(matrix)
    total_pixels = height * width
    
    for row in matrix:
        for pixel in row:
            histogram[pixel] += 1
    
    # Calculate cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
    
    # Normalize CDF to 0-255 range
    cdf_normalized = []
    for i in range(256):
        normalized_value = int((cdf[i] * 255.0) / total_pixels)
        cdf_normalized.append(normalized_value)
    
    # Apply equalization
    result_matrix = []
    for row in matrix:
        new_row = []
        for pixel in row:
            new_row.append(cdf_normalized[pixel])
        result_matrix.append(new_row)
    
    return result_matrix

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
