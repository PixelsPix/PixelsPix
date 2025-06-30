# import pygame
# from sys import exit

# pygame.init()
# pygame.display.set_caption("Simulador")

# screen_width  = 1200
# screen_height = 800
# screen = pygame.display.set_mode((screen_width, screen_height))
# screen.fill("White")
# clock = pygame.time.Clock()

# # coordenadas: primeiras duas var sao as coordenadas do ponto superior esquerdo
# lado_caixa = 500

# # superficie para desenho
# test_surface = pygame.Surface((lado_caixa, lado_caixa))
# test_surface.fill("Cyan")

# # imagem_surface = pygame.image.load("[path relativa pra imagem]").convert_alpha()
# # test_font = pygame.font.Font(Fonte, tamanho)
# # texto_surface = test_font.render(texto, AntiAliasing, cor)

# test_font = pygame.font.Font(None, 30)
# text_surface = test_font.render("Pressão: Pa", True, "Black")

# funcionando = True
# while funcionando:
#     for evento_usuario in pygame.event.get():
#         # fecha a janela quando sai do loop
#         if evento_usuario.type == pygame.QUIT:
#             pygame.quit()
#             exit()
    
#     # funcionamento
#     screen.blit(test_surface,(0.06 * screen_width, (screen_height - lado_caixa)//2))
#     screen.blit(text_surface,(0,0))

#     pygame.display.update()
#     clock.tick(60)

import pygame
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import threading
import queue
import time

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1200, 600))
clock = pygame.time.Clock()

# Generate sample data
data = np.random.normal(100, 15, 5000)

# 1. THREAD-SAFE RENDERING SYSTEM
class HistogramRenderer:
    def __init__(self):
        self.render_queue = queue.Queue(maxsize=1)
        self.current_surface = None
        self.thread = threading.Thread(target=self._render_worker, daemon=True)
        self.thread.start()
        
    def _render_worker(self):
        """Background rendering thread"""
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        fig.patch.set_facecolor('#f0f0f0')
        ax.set_facecolor('#f0f0f0')
        plt.tight_layout()
        
        while True:
            data = self.render_queue.get()
            if data is None:  # Exit signal
                break
                
            ax.clear()
            ax.hist(data, bins='auto', color='skyblue', edgecolor='none')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Optimized rendering
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            buf.seek(0)
            self.current_surface = pygame.image.load(buf)
            
    def update_data(self, new_data):
        """Thread-safe data update"""
        try:
            self.render_queue.put_nowait(new_data)
        except queue.Full:
            pass
            
    def get_surface(self):
        """Get the latest rendered surface"""
        return self.current_surface

# Initialize renderer
hist_renderer = HistogramRenderer()
hist_renderer.update_data(data)  # Initial render

# 2. MAIN GAME LOOP
running = True
last_update = time.time()
update_interval = 0.3  # Update every 300ms

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Throttled updates
    current_time = time.time()
    if current_time - last_update > update_interval:
        # Simulate new data (replace with your actual data)
        new_data = np.random.normal(100, 15, np.random.randint(1000, 5000))
        hist_renderer.update_data(new_data)
        last_update = current_time
    
    # Draw everything
    screen.fill((240, 240, 240))
    
    # Blit histogram if available
    hist_surface = hist_renderer.get_surface()
    if hist_surface:
        screen.blit(hist_surface, (50, 50))
    
    # FPS counter
    fps_text = pygame.font.SysFont('Arial', 24).render(
        f"FPS: {int(clock.get_fps())}", True, (0, 0, 0))
    screen.blit(fps_text, (1100, 20))
    
    pygame.display.flip()
    clock.tick(60)

# Cleanup
hist_renderer.render_queue.put(None)  # Stop thread
pygame.quit()