import pygame
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# Inicializar
pygame.init()

# Dimensoes da tela
WIDTH, HEIGHT = 1200, 600
tela = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulação de gás")

# Cores referencia pra usar
branco = (255, 255, 255)
preto = (0, 0, 0)

# Relogio pra framerate
clock = pygame.time.Clock()
FPS = 30

# figuras matplotlib
fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=100)  # Scatter plot
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)  # Histogram

# Get figure dimensions in display units
fig1_comprimento_polegadas = fig1.get_figwidth()

# Convert figure size to points (1 inch = 72 points)
fig_width_pontos = fig1_comprimento_polegadas * 72
fator_de_escala = fig_width_pontos / tamanho_parede
areas = np.pi * (raios * fator_de_escala)**2

x_data = np.random.normal(size=100)
y_data = np.random.normal(size=100)
values = np.random.randn(1000)  # For histogram

def transformar_em_superficie(figura):
    """Converte de figura do matplotlib pra uma superficie do pygame"""
    canvas = FigureCanvasAgg(figura)
    buffer = io.BytesIO()
    canvas.print_raw(buffer)
    buffer.seek(0)
    size = canvas.get_width_height()
    return pygame.image.fromstring(buffer.getvalue(), size, "RGBA")

def atualiza_plot():
    """Update the plot data"""
    # Clear and configure axes
    ax1.clear()
    ax2.clear()
    
    # --- Scatter Plot ---
    ax1.set_xlim(0, tamanho_parede)
    ax1.set_ylim(0, tamanho_parede)
    ax1.set_aspect('equal')
    ax1.set_title("Visualização das partículas")
    
    # Create scatter plot (size proportional to radius squared)
    ax1.scatter(
        posicoes[:, 0],
        posicoes[:, 1],
        s=(raios * fator_de_escala)**2,
        c='blue',
        alpha=0.5,
        edgecolors='red',
        linewidths=0.5
    )
    
    # --- Histogram Plot ---
    ax2.set_title("Distribuição de energia cinética")
    ax2.set_xlabel("Energia cinética")
    ax2.set_ylabel("Contagem")
    
    # Create histogram
    ax2.hist(
        energias_cineticas, 
        bins=30,
        color='skyblue',
        edgecolor='black',
        alpha=0.8
    )

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update plot (add your particle position updates here if needed)
    atualiza_plot()
    
    # Render and display
    surf = transformar_em_superficie(fig)
    tela.fill(branco)
    tela.blit(surf, (WIDTH//2 - surf.get_width()//2, HEIGHT//2 - surf.get_height()//2))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
plt.close('all')