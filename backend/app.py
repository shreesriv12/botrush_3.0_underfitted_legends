from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import faiss
import heapq
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import joblib 
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
from flask_cors import CORS
    
app = Flask(__name__)
CORS(app)

# Global constants
IMG_SIZE = 96

# Initialize models and paths
model_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush hackactivate\UnderfittedLegends\models\knn_model.pkl"
faiss_index_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush hackactivate\UnderfittedLegends\backend\model\faiss_index.index"
labels_path = r"C:\Users\shree\OneDrive\Desktop\projects\botrush hackactivate\UnderfittedLegends\backend\model\labels.npy"

# Load models
knn = joblib.load(model_path)
faiss_index = faiss.read_index(faiss_index_path)
labels = np.load(labels_path)

# Initialize the feature extractor
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, rgb_image

def detect_grid_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    horizontal_lines, vertical_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                horizontal_lines.append((y1 + y2) // 2)
            elif abs(x2 - x1) < 10:
                vertical_lines.append((x1 + x2) // 2)

    # Remove duplicates
    def count_unique_lines(lines, tolerance=10):
        lines.sort()
        unique = []
        for l in lines:
            if all(abs(l - u) > tolerance for u in unique):
                unique.append(l)
        return unique

    unique_h = count_unique_lines(horizontal_lines)
    unique_v = count_unique_lines(vertical_lines)

    return unique_h, unique_v

def segment_by_detected_lines(image, h_lines, v_lines):
    grid_cells = []
    grid_coordinates = []

    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]
            cell = image[y1:y2, x1:x2]
            grid_cells.append(cell)
            grid_coordinates.append((i, j, x1, y1, x2, y2))

    return grid_cells, grid_coordinates

def detect_colors(cell):
    # Resize for consistent sampling
    resized = cv2.resize(cell, (20, 20))
    pixels = resized.reshape(-1, 3)

    # Target colors in BGR (OpenCV uses BGR)
    gray_bgr = np.array([119, 100, 73])    # from HEX #496477
    green_bgr = np.array([100, 190, 0])    # from HEX #00BE64

    # Distance threshold for close-enough color match
    threshold = 50  # Looser threshold helps capture variations

    # Calculate Euclidean distances
    dist_to_gray = np.linalg.norm(pixels - gray_bgr, axis=1)
    dist_to_green = np.linalg.norm(pixels - green_bgr, axis=1)

    # Percentages of pixels matching each target color
    gray_match_pct = np.sum(dist_to_gray < threshold) / len(pixels) * 100
    green_match_pct = np.sum(dist_to_green < threshold) / len(pixels) * 100

    if gray_match_pct > 90:
        return "road"
    elif green_match_pct > 80:
        return "ground"
    else:
        return "unknown"

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img, verbose=0).flatten()
    prediction = knn.predict([features])[0]
    return "Safe" if prediction == 0 else "Unsafe"

def extract_features_from_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

def faiss_classify_cell(cell_img, index, labels):
    feat = extract_features_from_image(cell_img).astype('float32').reshape(1, -1)
    D, I = index.search(feat, 1)
    predicted_label = labels[I[0][0]]
    return "Safe" if predicted_label == 0 else "Unsafe"

def visualize_with_safety(image, grid_coords, grid_cells, classifications, index, labels):
    output = image.copy()
    safety_labels = []

    for idx, (row, col, x1, y1, x2, y2) in enumerate(grid_coords):
        label = classifications[idx]
        safety = "Unknown"

        if label == 'road':
            safety = "Safe"
            color = (0, 255, 0)  # Green
        elif label == 'ground':
            safety = "Unsafe"
            color = (0, 0, 255)  # Red
        else:
            # Use FAISS to classify unknown cells
            faiss_result = faiss_classify_cell(grid_cells[idx], index, labels)
            safety = faiss_result
            color = (0, 255, 0) if safety == "Safe" else (0, 0, 255)

        safety_labels.append(safety)

        # Overlay safety color with transparency
        overlay = np.ones_like(output[y1:y2, x1:x2]) * color
        cv2.addWeighted(overlay.astype(np.uint8), 0.4, output[y1:y2, x1:x2], 0.6, 0, output[y1:y2, x1:x2])

        # Add text label
        text = 'S' if safety == "Safe" else 'U'
        cv2.putText(output, text, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_out)
    plt.title("Safety Classification (S = Safe, U = Unsafe)")
    plt.axis('off')
    
    # Save figure to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close()
    
    return rgb_out, img_bytes, safety_labels

def convert_to_binary_matrix(city_matrix):
    return [[1 if cell == 'S' else 0 for cell in row] for row in city_matrix]

def prepare_grid_for_astar(city_matrix):
    return [[1 if cell == 'S' else 0 for cell in row] for row in city_matrix]

def heuristic(a, b):
    # Manhattan distance heuristic (optimal for grids)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, grid):
    neighbors = []
    x, y = node
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    # Check the 4 possible neighbors
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Make sure it's within grid bounds and traversable (1 = safe, 0 = unsafe)
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
            neighbors.append((nx, ny))

    return neighbors

def a_star(start, goal, grid):
    open_list = []  # Priority queue
    closed_list = set()  # Set of nodes already evaluated
    came_from = {}  # Tracks the best path to each node

    g_score = {start: 0}  # Cost of the path from start to current node
    f_score = {start: heuristic(start, goal)}  # Estimated total cost (g_score + heuristic)

    # Push the starting point to the open list
    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        # Pop the node with the lowest f_score
        current_f, current = heapq.heappop(open_list)

        # If we've reached the goal, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # Reverse to get path from start to goal
            return path

        closed_list.add(current)

        # Explore the neighbors of the current node
        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_list:
                continue

            tentative_g_score = g_score[current] + 1  # Every move costs 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                # Push the neighbor to the open list for further exploration
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # Return None if no path is found

def draw_path_on_image(image, path, grid_coords):
    output = image.copy()

    for (row, col, x1, y1, x2, y2) in grid_coords:
        # Check if the current grid cell is part of the path
        if (row, col) in path:
            # Draw a red square for the path
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_out)
    plt.title("Shortest Path on Grid")
    plt.axis('off')
    
    # Save figure to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close()
    
    return rgb_out, img_bytes

def process_image_with_path(image_path):
    image, _ = load_and_preprocess_image(image_path)
    h_lines, v_lines = detect_grid_lines(image)

    if len(h_lines) < 2 or len(v_lines) < 2:
        return {"error": "Insufficient grid lines detected"}, None, None
    
    grid_cells, grid_coords = segment_by_detected_lines(image, h_lines, v_lines)
    classifications = [detect_colors(cell) for cell in grid_cells]
    
    # Process grid for safety
    safety_matrix = []
    prev_row = grid_coords[0][0]
    row_temp = []
    
    for idx, (r, c, *_) in enumerate(grid_coords):
        if r != prev_row:
            safety_matrix.append(row_temp)
            row_temp = []
            prev_row = r
            
        label = classifications[idx]
        if label == 'road':
            cell_label = 'S'
        elif label == 'ground':
            cell_label = 'U'
        else:
            result = faiss_classify_cell(grid_cells[idx], faiss_index, labels)
            cell_label = 'S' if result == "Safe" else 'U'
            
        row_temp.append(cell_label)
    
    safety_matrix.append(row_temp)  # Add the last row
    
    # Generate safety visualization
    safety_img, safety_bytes, _ = visualize_with_safety(image, grid_coords, grid_cells, classifications, faiss_index, labels)
    
    # Run A* algorithm to find path
    grid = prepare_grid_for_astar(safety_matrix)
    start = (len(grid) - 1, 0)  # Bottom-left corner
    goal = (0, len(grid[0]) - 1)  # Top-right corner
    
    path = a_star(start, goal, grid)
    
    if path:
        # Generate path visualization
        path_img, path_bytes = draw_path_on_image(image, path, grid_coords)
        return {
            "safety_matrix": safety_matrix,
            "binary_matrix": convert_to_binary_matrix(safety_matrix),
            "path": path,
            "path_exists": True
        }, safety_bytes, path_bytes
    else:
        return {
            "safety_matrix": safety_matrix, 
            "binary_matrix": convert_to_binary_matrix(safety_matrix),
            "path_exists": False
        }, safety_bytes, None

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file temporarily
    temp_path = "temp_uploaded_image.jpg"
    file.save(temp_path)
    
    try:
        # Process the image
        result, safety_img_bytes, path_img_bytes = process_image_with_path(temp_path)
        
        response = {
            "result": result,
            "safety_image_url": "/get_safety_image" if safety_img_bytes else None,
            "path_image_url": "/get_path_image" if path_img_bytes else None
        }
        
        # Store the image bytes in memory (in a real app, you'd use a caching solution)
        if safety_img_bytes:
            app.safety_img_bytes = safety_img_bytes
        if path_img_bytes:
            app.path_img_bytes = path_img_bytes
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/get_safety_image')
def get_safety_image():
    if hasattr(app, 'safety_img_bytes'):
        return send_file(app.safety_img_bytes, mimetype='image/png')
    return jsonify({"error": "No safety image available"}), 404

@app.route('/get_path_image')
def get_path_image():
    if hasattr(app, 'path_img_bytes'):
        return send_file(app.path_img_bytes, mimetype='image/png')
    return jsonify({"error": "No path image available"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)