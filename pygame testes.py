import pygame
import sys
import numpy as np
import numpy.typing as npt
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict

# pygame.init()

# parametros da simulacao
QUANTIDADE_INICIAL_PARTICULAS = 200
MIN_PARTICULAS = 10
MAX_PARTICULAS = 500

RAIO_PARTICULA_NM = 0.5  # nm
MASSA_PARTICULA = 1.66e-27  # kg (massa do hidrogenio)
TEMPO_ENTRE_FRAMES = 0.2 # ns
TEMPERATURA_INICIAL = 300 # K

CONSTANTE_BOLTZMANN = 1.380649e-23  # J/K
COR_PARTICULA = (100, 150, 255) # azul
LOTE_PARTICULAS = 10 # particulas adicionadas/removidas por clique

LARGURA_TELA, ALTURA_TELA = 1200, 900  # tamanho da tela em pixels
PIXELS_POR_NM = 10  # fator de escala (10 pixels = 1 nm)
LARGURA_CAIXA_NM = 60  # nm de largura
ALTURA_CAIXA_NM  = 60  # nm de altura

RAIO_PARTICULA_PX = RAIO_PARTICULA_NM * PIXELS_POR_NM
LARGURA_CAIXA_PX  = LARGURA_CAIXA_NM * PIXELS_POR_NM
ALTURA_CAIXA_PX   = ALTURA_CAIXA_NM * PIXELS_POR_NM

POS_X_CAIXA_PX, POS_Y_CAIXA_PX = 50, 150  # posicao da caixa na tela em pixels

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (200, 200, 200)
VERDE = (0, 255, 0)
CINZA_ESCURO = (100, 100, 100)
LARANJA = (255, 165, 0) 
AZUL_CLARO = (100, 150, 255)

# configura a tela
# screen = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
# pygame.display.set_caption("Simulador de Partículas de Gás Ideal")
# clock = pygame.time.Clock()
# fonte = pygame.font.SysFont('Arial', 16)
# fonte_titulo = pygame.font.SysFont('Arial', 24, bold=True)

# grid para otimizar colisões
TAMANHO_CELULA_NM = LARGURA_CAIXA_NM / 10
TAMANHO_CELULA_PX = TAMANHO_CELULA_NM * PIXELS_POR_NM

LARGURA_GRID = int(LARGURA_CAIXA_NM / TAMANHO_CELULA_NM) + 1
ALTURA_GRID  = int(ALTURA_CAIXA_NM  / TAMANHO_CELULA_NM) + 1

# classe particula
class Particula:    
    def __init__(self, posicao_x, posicao_y, velocidade_x, velocidade_y, raio, massa) -> None:
        self.posicao_x = posicao_x
        self.posicao_y = posicao_y
        self.velocidade_x = velocidade_x
        self.velocidade_y = velocidade_y
        self.raio = raio
        self.massa = massa
    
    @property
    def momento(self) -> npt.NDArray[np.float64]:
        """Retorna um ndarray dado por: massa * velocidade"""
        return self.massa * np.array[self.velocidade_x, self.velocidade_y]
    