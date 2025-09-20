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


# transforma de figura do matplotlib pra superficie do pygame
def transformar_em_superficie(figura):
    """Converte de figura do matplotlib pra uma superficie do pygame"""
    canvas = FigureCanvasAgg(figura)
    canvas.draw()  # Renderiza a figura
    buffer = canvas.buffer_rgba()  # Pega o buffer RGBA
    size = canvas.get_width_height()
    return pygame.image.frombuffer(buffer, size, "RGBA")

def atualiza_plot():
    """Atualiza o plot das particulas"""
    # Clear and configure axes
    ax1.clear()
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
    
def atualiza_histograma():
    """Atualiza o histograma dos valores de energia cinética"""
    ax2.clear()
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
    atualiza_histograma()
    
    # Render and display
    surperficie_particulas = transformar_em_superficie(fig1)
    superficie_histograma  = transformar_em_superficie(fig2)

    # Limpa a tela
    tela.fill(branco)

    # Blit both surfaces side by side
    tela.blit(surperficie_particulas, (50, 50))
    tela.blit(superficie_histograma, (50, HEIGHT//2 + 50))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
plt.close('all')