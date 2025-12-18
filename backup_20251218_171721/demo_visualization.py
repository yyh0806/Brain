#!/usr/bin/env python3
"""
Visualization System Demo
Demonstrates 2D/3D visualization capabilities with pygame and opencv
"""

import pygame
import cv2
import numpy as np
import math
import sys

def pygame_3d_rotation_demo():
    """Demo of 3D rotation using pygame"""
    print("Starting Pygame 3D Rotation Demo...")
    print("Press ESC to exit")

    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Rotation Demo - Pygame")
    clock = pygame.time.Clock()

    # Define 3D cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
    ]) * 100

    # Define cube edges (which vertices to connect)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]

    angle_x, angle_y, angle_z = 0, 0, 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear screen
        screen.fill((0, 0, 0))

        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(angle_x), -math.sin(angle_x)],
            [0, math.sin(angle_x), math.cos(angle_x)]
        ])

        rot_y = np.array([
            [math.cos(angle_y), 0, math.sin(angle_y)],
            [0, 1, 0],
            [-math.sin(angle_y), 0, math.cos(angle_y)]
        ])

        rot_z = np.array([
            [math.cos(angle_z), -math.sin(angle_z), 0],
            [math.sin(angle_z), math.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Apply rotations
        rotated = vertices @ rot_x.T @ rot_y.T @ rot_z.T

        # Project to 2D (simple orthographic projection)
        projected = rotated[:, :2] + [width // 2, height // 2]

        # Draw edges
        for edge in edges:
            start_point = projected[edge[0]].astype(int)
            end_point = projected[edge[1]].astype(int)
            pygame.draw.line(screen, (0, 255, 0), start_point, end_point, 2)

        # Draw vertices
        for point in projected:
            pygame.draw.circle(screen, (255, 0, 0), point.astype(int), 5)

        # Update display
        pygame.display.flip()

        # Update rotation angles
        angle_x += 0.01
        angle_y += 0.015
        angle_z += 0.005

        clock.tick(60)

    pygame.quit()

def opencv_image_processing_demo():
    """Demo of image processing with OpenCV"""
    print("\nStarting OpenCV Image Processing Demo...")

    # Create a synthetic image with various shapes
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw various shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)          # Green circle
    cv2.ellipse(img, (450, 100), (50, 30), 45, 0, 360, (0, 0, 255), -1)  # Red ellipse

    # Draw some lines
    for i in range(5):
        cv2.line(img, (50 + i * 30, 200), (150 + i * 30, 300), (255, 255, 0), 2)

    # Add text
    cv2.putText(img, "OpenCV Demo", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Apply various filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Create a combined display
    combined = np.hstack([
        np.vstack([img, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]),
        np.vstack([blurred, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
    ])

    print("Displaying combined image with original, grayscale, blurred, and edge detection")
    print("Press any key to continue...")

    cv2.imshow("OpenCV Image Processing Demo", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def monitoring_panel_simulation():
    """Simulate a monitoring panel with real-time data visualization"""
    print("\nStarting Monitoring Panel Simulation...")
    print("Press ESC to exit")

    pygame.init()
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Monitoring Panel Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Data storage for visualization
    data_points = 100
    cpu_data = np.random.rand(data_points) * 50 + 25
    memory_data = np.random.rand(data_points) * 30 + 40
    network_data = np.random.rand(data_points) * 80 + 10

    running = True
    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear screen
        screen.fill((20, 20, 30))

        # Update data (simulate real-time monitoring)
        if frame_count % 5 == 0:
            cpu_data = np.roll(cpu_data, -1)
            cpu_data[-1] = np.random.rand() * 50 + 25

            memory_data = np.roll(memory_data, -1)
            memory_data[-1] = np.random.rand() * 30 + 40

            network_data = np.roll(network_data, -1)
            network_data[-1] = np.random.rand() * 80 + 10

        # Draw title
        title_text = font.render("Real-time Monitoring Panel", True, (255, 255, 255))
        screen.blit(title_text, (width // 2 - title_text.get_width() // 2, 20))

        # Draw CPU usage graph
        draw_graph(screen, cpu_data, 50, 100, 280, 200, "CPU Usage", (255, 100, 100))

        # Draw Memory usage graph
        draw_graph(screen, memory_data, 350, 100, 280, 200, "Memory Usage", (100, 255, 100))

        # Draw Network traffic graph
        draw_graph(screen, network_data, 650, 100, 280, 200, "Network Traffic", (100, 100, 255))

        # Draw status indicators
        draw_status_panel(screen, font, width, height)

        # Update display
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1

    pygame.quit()

def draw_graph(screen, data, x, y, width, height, title, color):
    """Draw a line graph on the screen"""
    # Draw border
    pygame.draw.rect(screen, (100, 100, 100), (x, y, width, height), 2)

    # Draw title
    font = pygame.font.Font(None, 20)
    title_text = font.render(title, True, color)
    screen.blit(title_text, (x + 5, y - 25))

    # Scale data to fit the graph area
    max_val = 100
    scaled_data = data / max_val * (height - 10)

    # Draw the graph line
    points = []
    for i, value in enumerate(scaled_data):
        px = x + (i * width // len(data))
        py = y + height - value - 5
        points.append((px, py))

    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 2)

    # Draw current value
    current_val = data[-1]
    val_text = font.render(f"{current_val:.1f}%", True, color)
    screen.blit(val_text, (x + width - 60, y - 25))

def draw_status_panel(screen, font, width, height):
    """Draw status indicators panel"""
    panel_y = height - 120

    # Draw panel background
    pygame.draw.rect(screen, (40, 40, 50), (50, panel_y, width - 100, 100))
    pygame.draw.rect(screen, (100, 100, 100), (50, panel_y, width - 100, 100), 2)

    # Status items
    status_items = [
        ("System Status", "ONLINE", (0, 255, 0)),
        ("CPU Temperature", "52°C", (255, 200, 0)),
        ("Active Processes", "147", (0, 200, 255)),
        ("Uptime", "2h 34m", (200, 200, 200))
    ]

    x_offset = 80
    for label, value, color in status_items:
        label_text = font.render(label + ":", True, (200, 200, 200))
        value_text = font.render(value, True, color)
        screen.blit(label_text, (x_offset, panel_y + 20))
        screen.blit(value_text, (x_offset, panel_y + 50))
        x_offset += 200

def main():
    """Run visualization demos"""
    print("=" * 60)
    print("VISUALIZATION SYSTEM DEMONSTRATION")
    print("=" * 60)

    demos = [
        ("1", "Pygame 3D Rotation Demo", pygame_3d_rotation_demo),
        ("2", "OpenCV Image Processing Demo", opencv_image_processing_demo),
        ("3", "Monitoring Panel Simulation", monitoring_panel_simulation),
        ("4", "Run All Demos", None)
    ]

    print("\nAvailable Demos:")
    for key, name, _ in demos:
        print(f"  {key}. {name}")
    print("  Q. Quit")

    while True:
        choice = input("\nSelect demo (1-4, Q): ").strip().upper()

        if choice == 'Q':
            print("Exiting...")
            break
        elif choice == '1':
            pygame_3d_rotation_demo()
        elif choice == '2':
            opencv_image_processing_demo()
        elif choice == '3':
            monitoring_panel_simulation()
        elif choice == '4':
            print("Running all demos...")
            pygame_3d_rotation_demo()
            opencv_image_processing_demo()
            monitoring_panel_simulation()
        else:
            print("Invalid choice. Please try again.")

    print("\nVisualization system demo completed!")

if __name__ == "__main__":
    main()