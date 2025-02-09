import pygame
import numpy as np

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def run_box_drawer(image):
    pygame.init()

    image_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(image)))

    image_rect = image_surface.get_rect()

    screen = pygame.display.set_mode((image_rect.width, image_rect.height))
    pygame.display.set_caption("Interactive Bounding Box Selector")

    bounding_boxes = []
    drawing = False
    start_pos = None
    current_label = "Foreground"

    main_loop(screen, bounding_boxes, drawing, start_pos, current_label, image_surface)

    return bounding_boxes

def main_loop(screen, bounding_boxes, drawing, start_pos, current_label, image_surface):
    running = True
    while running:
        screen.fill(WHITE)
        screen.blit(image_surface, (0, 0))

        for bbox, label in bounding_boxes:
            color = GREEN if label == "Foreground" else RED
            pygame.draw.rect(screen, color, bbox, 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    start_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    end_pos = event.pos

                    x1, y1 = start_pos
                    x2, y2 = end_pos
                    bbox = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                    bounding_boxes.append((bbox, current_label))
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    current_label = "Foreground"
                elif event.key == pygame.K_b:
                    current_label = "Background"
                elif event.key == pygame.K_u:
                    if bounding_boxes:
                        bounding_boxes.pop()
                elif event.key == pygame.K_c:
                    bounding_boxes.clear()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if drawing and start_pos:
            mouse_pos = pygame.mouse.get_pos()
            preview_bbox = pygame.Rect(
                min(start_pos[0], mouse_pos[0]),
                min(start_pos[1], mouse_pos[1]),
                abs(mouse_pos[0] - start_pos[0]),
                abs(mouse_pos[1] - start_pos[1])
            )
            pygame.draw.rect(screen, BLUE, preview_bbox, 2)

        font = pygame.font.Font(None, 36)
        text_current_mode = font.render(f"Current Mode: {current_label} | f: foreground | b: background | u: undo | c: clear", True, (0, 0, 0))
        screen.blit(text_current_mode, (10, 10))

        pygame.display.flip()

    pygame.quit()

