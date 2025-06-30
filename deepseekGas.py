import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict

pygame.init()

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

LARGURA_PX, ALTURA_PX = 1200, 900  # tamanho da tela em pixels
PIXELS_POR_NM = 10  # fator de escala (10 pixels = 1 nm)
LARGURA_CAIXA_NM = 60  # nm de largura
ALTURA_CAIXA_NM  = 60  # nm de altura

RAIO_PARTICULA_PX = RAIO_PARTICULA_NM * PIXELS_POR_NM
LARGURA_CAIXA_PX = LARGURA_CAIXA_NM * PIXELS_POR_NM
ALTURA_CAIXA_PX = ALTURA_CAIXA_NM * PIXELS_POR_NM
POS_X_CAIXA_PX, POS_Y_CAIXA_PX = 50, 150  # Posição na tela em pixels

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (200, 200, 200)
VERDE = (0, 255, 0)
CINZA_ESCURO = (100, 100, 100)
LARANJA = (255, 165, 0)

# configura a tela
screen = pygame.display.set_mode((LARGURA_PX, ALTURA_PX))
pygame.display.set_caption("Simulador de Partículas de Gás Ideal")
clock = pygame.time.Clock()
fonte = pygame.font.SysFont('Arial', 16)
fonte_titulo = pygame.font.SysFont('Arial', 24, bold=True)

# grid espacial para otimizar colisões
TAMANHO_CELULA_NM = LARGURA_CAIXA_NM / 10
TAMANHO_CELULA_PX = TAMANHO_CELULA_NM * PIXELS_POR_NM

LARGURA_GRID = int(LARGURA_CAIXA_NM / TAMANHO_CELULA_NM) + 1
ALTURA_GRID  = int(ALTURA_CAIXA_NM  / TAMANHO_CELULA_NM) + 1

class SistemaParticulas:
    def __init__(self):
        self.particulas = []
        self.grid = defaultdict(list)
        self.colisoes_parede = 0
        self.ultimo_tempo_colisao = pygame.time.get_ticks()
        self.pressao = 0
        self.temperatura_atual = TEMPERATURA_INICIAL
        self.temperatura_alvo = TEMPERATURA_INICIAL
        self.energia_max = 8e-20  # valor maximo fixo pro eixo x do histograma
        self.inicializar_particulas(QUANTIDADE_INICIAL_PARTICULAS, TEMPERATURA_INICIAL)

    def velocidade_boltzmann(self, temperatura):
        """Gera velocidade em m/s a partir da distribuição de Boltzmann"""
        sigma = np.sqrt(temperatura * CONSTANTE_BOLTZMANN / MASSA_PARTICULA) # m/s
        return random.gauss(0, sigma)

    def inicializar_particulas(self, num_particulas, temperatura):
        """Cria (e salva em self.particulas) particulas com uma distribuição uniforme de posições e velocidades"""
        self.particulas = []
        for _ in range(num_particulas):
            # posições aleatorias dentro da caixa (em coordenadas de pixel)
            x_px = random.uniform(POS_X_CAIXA_PX + RAIO_PARTICULA_PX, POS_X_CAIXA_PX + LARGURA_CAIXA_PX - RAIO_PARTICULA_PX)
            y_px = random.uniform(POS_Y_CAIXA_PX + RAIO_PARTICULA_PX, POS_Y_CAIXA_PX + ALTURA_CAIXA_PX  - RAIO_PARTICULA_PX)
            
            # gera velocidades aleatórias em m/s
            velocidade = self.velocidade_boltzmann(temperatura)
            angulo = random.uniform(0, 2*np.pi)
            vx = velocidade * np.cos(angulo)
            vy = velocidade * np.sin(angulo)
            
            self.particulas.append({
                'x': x_px, 'y': y_px,  # posicao em pixels
                'vx': vx,  'vy': vy,    # velocidade em m/s
                'raio': RAIO_PARTICULA_PX,
                'raio_nm': RAIO_PARTICULA_NM,
                'massa': MASSA_PARTICULA,
            })

    def atualizar_grade(self):
        """Salva o índice de cada particula em cada celula do self.grade"""
        self.grid.clear()
        for idx, p in enumerate(self.particulas):
            # converte posicao para coordenadas da grade
            grid_x = int((p['x'] - POS_X_CAIXA_PX) / PIXELS_POR_NM) // TAMANHO_CELULA_NM
            grid_y = int((p['y'] - POS_Y_CAIXA_PX) / PIXELS_POR_NM) // TAMANHO_CELULA_NM
            self.grid[(grid_x, grid_y)].append(idx)

    def tratar_colisoes(self):
        for celula, indices in self.grid.items():
            # Verifica partículas nesta célula
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    self.colisao_particulas(indices[i], indices[j])
            
            # Verifica células vizinhas
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                celula_vizinha = (celula[0] + dx, celula[1] + dy)
                if celula_vizinha in self.grid:
                    for i in indices:
                        for j in self.grid[celula_vizinha]:
                            if i < j:
                                self.colisao_particulas(i, j)

    def colisao_particulas(self, indice_particula_1, indice_particula_2):
        p1, p2 = self.particulas[indice_particula_1], self.particulas[indice_particula_2]
        # Converte posições para nm para detecção de colisão
        x1_nm = (p1['x'] - POS_X_CAIXA_PX) / PIXELS_POR_NM
        y1_nm = (p1['y'] - POS_Y_CAIXA_PX) / PIXELS_POR_NM
        x2_nm = (p2['x'] - POS_X_CAIXA_PX) / PIXELS_POR_NM
        y2_nm = (p2['y'] - POS_Y_CAIXA_PX) / PIXELS_POR_NM
        
        dx_nm = x2_nm - x1_nm
        dy_nm = y2_nm - y1_nm
        distancia_nm = np.sqrt(dx_nm**2 + dy_nm**2)
        
        if distancia_nm < p1['raio_nm'] + p2['raio_nm']:
            # Colisão elástica em coordenadas nm
            nx, ny = dx_nm/distancia_nm, dy_nm/distancia_nm
            dvx = p2['vx'] - p1['vx']
            dvy = p2['vy'] - p1['vy']
            vn = dvx * nx + dvy * ny
            
            if vn < 0:
                impulso = 2 * vn / (1/p1['massa'] + 1/p2['massa'])
                p1['vx'] += impulso * nx / p1['massa']
                p1['vy'] += impulso * ny / p1['massa']
                p2['vx'] -= impulso * nx / p2['massa']
                p2['vy'] -= impulso * ny / p2['massa']

    def atualizar(self, dt):
        """Atualiza as posições das partículas, atualiza o grid, faz colisões e atualiza pressão e temperatura."""
        energia_cinetica_total = 0
        transferencia_momento_parede = 0
        
        # Atualiza posições (converte velocidade de m/s para pixels/frame)
        for p in self.particulas:
            p['x'] += p['vx'] * dt * PIXELS_POR_NM
            p['y'] += p['vy'] * dt * PIXELS_POR_NM
            
            # Colisões com as paredes (em coordenadas de pixel)
            if p['x'] - p['raio'] < POS_X_CAIXA_PX:
                p['x'] = POS_X_CAIXA_PX + p['raio']
                p['vx'] = -p['vx']
                self.colisoes_parede += 1
                transferencia_momento_parede += 2 * abs(p['vx'] * p['massa'])
            
            if p['x'] + p['raio'] > POS_X_CAIXA_PX + LARGURA_CAIXA_PX:
                p['x'] = POS_X_CAIXA_PX + LARGURA_CAIXA_PX - p['raio']
                p['vx'] = -p['vx']
                self.colisoes_parede += 1
                transferencia_momento_parede += 2 * abs(p['vx'] * p['massa'])
            
            if p['y'] - p['raio'] < POS_Y_CAIXA_PX:
                p['y'] = POS_Y_CAIXA_PX + p['raio']
                p['vy'] = -p['vy']
                self.colisoes_parede += 1
                transferencia_momento_parede += 2 * abs(p['vy'] * p['massa'])
            
            if p['y'] + p['raio'] > POS_Y_CAIXA_PX + ALTURA_CAIXA_PX:
                p['y'] = POS_Y_CAIXA_PX + ALTURA_CAIXA_PX - p['raio']
                p['vy'] = -p['vy']
                self.colisoes_parede += 1
                transferencia_momento_parede += 2 * abs(p['vy'] * p['massa'])
            
            # calcula energia cinetica
            ke = 0.5 * p['massa'] * (p['vx']**2 + p['vy']**2)
            energia_cinetica_total += ke
        
        # atualiza grid e trata colisoes
        self.atualizar_grade()
        self.tratar_colisoes()
        
        # calcula pressao (area da parede em metros)
        tempo_atual = pygame.time.get_ticks()
        if tempo_atual - self.ultimo_tempo_colisao > 0:
            tempo_decorrido = (tempo_atual - self.ultimo_tempo_colisao) / 1000
            area_parede_m = 2 * (LARGURA_CAIXA_NM + ALTURA_CAIXA_NM) * 1e-9
            self.pressao = transferencia_momento_parede / (tempo_decorrido * area_parede_m)
            self.ultimo_tempo_colisao = tempo_atual
        
        # calcula temperatura
        if self.particulas:
            self.temperatura_atual = (energia_cinetica_total / len(self.particulas)) / CONSTANTE_BOLTZMANN
            
            # ajusta velocidades para corresponder a temperatura alvo
            if abs(self.temperatura_atual - self.temperatura_alvo) > 1:
                fator_escala = np.sqrt(self.temperatura_alvo / max(1, self.temperatura_atual))
                for p in self.particulas:
                    p['vx'] *= fator_escala
                    p['vy'] *= fator_escala
        
        return energia_cinetica_total
    
    def adicionar_particulas(self, quantidade, temperatura):
        """Adiciona partículas com velocidades térmicas adequadas"""
        for _ in range(quantidade):
            if len(self.particulas) >= MAX_PARTICULAS:
                break
                
            x_px = random.uniform(POS_X_CAIXA_PX + RAIO_PARTICULA_PX, POS_X_CAIXA_PX + LARGURA_CAIXA_PX - RAIO_PARTICULA_PX)
            y_px = random.uniform(POS_Y_CAIXA_PX + RAIO_PARTICULA_PX, POS_Y_CAIXA_PX + ALTURA_CAIXA_PX - RAIO_PARTICULA_PX)
            
            # Gera velocidades térmicas
            velocidade = self.velocidade_boltzmann(temperatura)
            angulo = random.uniform(0, 2*np.pi)
            vx = velocidade * np.cos(angulo)
            vy = velocidade * np.sin(angulo)
            
            self.particulas.append({
                'x': x_px, 'y': y_px,
                'vx': vx, 'vy': vy,
                'raio': RAIO_PARTICULA_PX,
                'raio_nm': RAIO_PARTICULA_NM,
                'massa': MASSA_PARTICULA,
            })

    def remover_particulas(self, quantidade):
        """Remove múltiplas partículas de uma vez"""
        for _ in range(quantidade):
            if len(self.particulas) <= MIN_PARTICULAS:
                break
            self.particulas.pop()

# inicializa o sistema de particulas
sistema_particulas = SistemaParticulas()

# plotagem
plt.ioff()
fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
fig.subplots_adjust(bottom=0.2)
canvas = FigureCanvasAgg(fig)

def desenhar_histograma(energias_cineticas):
    """Desenha o histograma e retorna uma superficie"""
    ax.clear()
    if energias_cineticas:
        bins = np.linspace(0, sistema_particulas.energia_max, 20)
        contagens, bins, patches = ax.hist(energias_cineticas, bins=bins, color='blue', alpha=0.7)
        
        # distribuicao de Maxwell-Boltzmann
        if len(energias_cineticas) > 1:
            kT = sistema_particulas.temperatura_atual * CONSTANTE_BOLTZMANN
            x = np.linspace(0, sistema_particulas.energia_max, 100)
            # calcula largura do bin para normalização adequada
            largura_bin = bins[1] - bins[0]
            # distribuicao de Maxwell-Boltzmann normalizada corretamente
            maxwell = (len(energias_cineticas) * largura_bin * (x/(kT**2)) * np.exp(-x/kT))
            ax.plot(x, maxwell, 'r-', linewidth=2)
    
    ax.set_title('Distribuição de Energia Cinética')
    ax.set_xlabel('Energia (J)')
    ax.set_ylabel('Contagem')
    ax.set_xlim(0, sistema_particulas.energia_max)
    
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    return pygame.image.fromstring(raw_data, size, "RGB")

def desenhar_botao(x, y, largura, altura, texto, ativo=True, cor=VERDE):
    """Desenha um botão verde que fica cinza se tiver inativo"""
    cor_botao = cor if ativo else CINZA_ESCURO

    pygame.draw.rect(screen, cor_botao, (x, y, largura, altura))
    pygame.draw.rect(screen, BRANCO, (x, y, largura, altura), 2)

    texto_surperficie = fonte.render(texto, True, BRANCO)
    texto_retangulo = texto_surperficie.get_rect(center=(x + largura/2, y + altura/2))
    screen.blit(texto_surperficie, texto_retangulo)
    return pygame.Rect(x, y, largura, altura)

def main():
    running = True
    pausado = False
    
    # Retângulos dos botões
    retangulo_botao_adicionar = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_remover = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_aquecer = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_resfriar = pygame.Rect(0, 0, 0, 0)
    
    while running:
        dt = 0.0005
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pausado = not pausado

                elif event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                posicao_mouse = pygame.mouse.get_pos()

                if retangulo_botao_adicionar.collidepoint(posicao_mouse):
                    sistema_particulas.adicionar_particulas(LOTE_PARTICULAS, sistema_particulas.temperatura_alvo)

                elif retangulo_botao_remover.collidepoint(posicao_mouse):
                    sistema_particulas.remover_particulas(LOTE_PARTICULAS)

                elif retangulo_botao_aquecer.collidepoint(posicao_mouse) and sistema_particulas.temperatura_alvo > 1300:
                    sistema_particulas.temperatura_alvo += 50 # K

                elif retangulo_botao_resfriar.collidepoint(posicao_mouse) and sistema_particulas.temperatura_alvo > 50:
                    sistema_particulas.temperatura_alvo -= 50 # K
        
        if not pausado:
            energia_total = sistema_particulas.atualizar(dt)
            energias_cineticas = [0.5 * p['massa'] * ((p['vx'])**2 + (p['vy'])**2) for p in sistema_particulas.particulas]
            superficie_hist = desenhar_histograma(energias_cineticas)
        
        # desenha tudo
        screen.fill(PRETO)
        
        # desenha titulos
        screen.blit(fonte_titulo.render("Simulação", True, BRANCO), (50, 50))
        screen.blit(fonte_titulo.render("Dados", True, BRANCO), (700, 50))
        screen.blit(fonte_titulo.render("Histograma", True, BRANCO), (700, 300))
        
        # desenha caixa de simulacao e particulas
        pygame.draw.rect(screen, BRANCO, (POS_X_CAIXA_PX, POS_Y_CAIXA_PX, LARGURA_CAIXA_PX, ALTURA_CAIXA_PX), 2)
        for p in sistema_particulas.particulas:
            pygame.draw.circle(screen, COR_PARTICULA, (int(p['x']), int(p['y'])), int(p['raio']))
        
        # Desenha botões de controle
        retangulo_botao_adicionar = desenhar_botao(700, 100, 150, 40, "Adicionar 10 Partículas", len(sistema_particulas.particulas) < MAX_PARTICULAS)
        retangulo_botao_remover   = desenhar_botao(700, 150, 150, 40, "Remover 10 Partículas", len(sistema_particulas.particulas) > MIN_PARTICULAS)
        retangulo_botao_aquecer   = desenhar_botao(700, 200, 150, 40, "Aquecer (+50K)",  sistema_particulas.temperatura_alvo < 1300)
        retangulo_botao_resfriar  = desenhar_botao(700, 250, 150, 40, "Resfriar (-50K)", sistema_particulas.temperatura_alvo > 50)
        
        # Mostra estatísticas (agora mostrando unidades nm)
        estatisticas = [
            f"Partículas: {len(sistema_particulas.particulas)}",
            f"Raio da Partícula: {RAIO_PARTICULA_NM} nm",
            f"Massa da Partícula: {MASSA_PARTICULA:.1e} kg",
            f"Tamanho da Caixa: {LARGURA_CAIXA_NM} nm x {ALTURA_CAIXA_NM} nm",
            f"Temp. Alvo: {sistema_particulas.temperatura_alvo:.1f} K",
            f"Temp. Atual: {sistema_particulas.temperatura_atual:.1f} K",
            f"Pressão: {sistema_particulas.pressao:.2e} Pa",
            f"Espaço: Pausar/Continuar | Esc: Sair"
        ]
        
        for i, estatistica in enumerate(estatisticas):
            screen.blit(fonte.render(estatistica, True, BRANCO), (700, 350 + i * 25))
        
        # Desenha histograma
        if not pausado:
            screen.blit(superficie_hist, (700, 550))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()