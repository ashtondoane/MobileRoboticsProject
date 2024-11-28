import math
import numpy as np
import heapq
from PIL import Image, ImageDraw

# Node class to represent a node in the search space
class NODE:
    def __init__(self, position, g=float('inf'), h=0, parent=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    # Nodes will be compared based on their f value, we use this because of the heapq
    def __lt__(self, other):
        return self.f < other.f

    # Function to represent the node
    def __repr__(self):
        return f"Node(position={self.position}, g={self.g}, h={self.h}, f={self.f})"

    @staticmethod
    def reconstruct_path(node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

# Function to convert an image to a binary array 
def convert_image_to_binary_array(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    image = image.convert("L")

    # Set the binary threshold
    binary_threshold = 128

    # Convert grayscale to binary image
    bw_image = image.point(lambda p: p < binary_threshold and 1)

    # Convert image to numpy array
    binary_matrix = np.array(bw_image)
    return binary_matrix

# Heuristic function using Octile distance (allowing 8 directions of movement)
def Heuristic_function(node, goal):
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# A* search algorithm
def A_star(environment_map, start_position, goal_position):

    # Create the start and goal nodes
    start_node = NODE(start_position, g=0)
    goal_node = NODE(goal_position)

    # Set the heuristic value for the start node
    start_node.h = Heuristic_function(start_node, goal_node)
    start_node.f = start_node.g + start_node.h

    # Initialize the open and closed lists
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()

    # Dictionary to keep track of nodes
    nodes = {}
    nodes[tuple(start_node.position)] = start_node

    while open_list:
        # Pop the node with the lowest f value
        current_node = heapq.heappop(open_list)

        # If the current node is in the closed set, skip it
        if tuple(current_node.position) in closed_set:
            continue

        # Add the current node's position to the closed set
        closed_set.add(tuple(current_node.position))

        # If the current node is the goal, reconstruct the path
        if current_node.position == goal_node.position:
            path = NODE.reconstruct_path(current_node)
            print("Path found:", path)
            return path

        # Possible moves: 8 directions (including diagonals)
        moves = [[1, 0], [0, 1], [-1, 0], [0, -1],
                 [1, 1], [-1, -1], [1, -1], [-1, 1]]

        # Explore neighbors
        for move in moves:
            neighbor_position = [current_node.position[0] + move[0],
                                 current_node.position[1] + move[1]]
            neighbor_pos = tuple(neighbor_position)

            # Skip if out of bounds or obstacle
            if (neighbor_position[0] < 0 or neighbor_position[0] >= environment_map.shape[0] or
                neighbor_position[1] < 0 or neighbor_position[1] >= environment_map.shape[1] or
                environment_map[neighbor_position[0], neighbor_position[1]] == 1):
                continue

            # Skip if in closed set
            if neighbor_pos in closed_set:
                continue

            # Calculate movement cost (diagonal movement cost is sqrt(2))
            dx = abs(move[0])
            dy = abs(move[1])
            movement_cost = math.sqrt(2) if dx == 1 and dy == 1 else 1

            # Calculate tentative g value
            g_tentative = current_node.g + movement_cost

            # Create or get the neighbor node
            if neighbor_pos not in nodes:
                neighbor_node = NODE(neighbor_position)
                nodes[neighbor_pos] = neighbor_node
            else:
                neighbor_node = nodes[neighbor_pos]

            # If this path to neighbor is better, record it
            if g_tentative < neighbor_node.g:
                neighbor_node.g = g_tentative
                neighbor_node.h = Heuristic_function(neighbor_node, goal_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                neighbor_node.parent = current_node

                # Add the neighbor to the open list
                heapq.heappush(open_list, neighbor_node)

    print("No path found.")
    return None

# Function to swap the x and y coordinates of the path
def swap_path_coordinates(path_coordinates):
    swapped_path = [[y, x] for x, y in path_coordinates]
    return swapped_path

# Function to display the path on the maze image
def display_path(maze_name, path):
    # Load the maze image
    maze_path = maze_name
    maze_img = Image.open(maze_path)

    # Convert the maze image to RGB (to draw in color)
    maze_img = maze_img.convert("RGB")

    path = swap_path_coordinates(path)

    # Create a drawing object
    draw = ImageDraw.Draw(maze_img)

    # Draw the path (red line or points)
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        draw.line((x1, y1, x2, y2), fill=(255, 0, 255), width=4)  # Red line with width 2

    # Save and show the image
    maze_img.save("maze_with_path.png")
    maze_img.show()
    return
