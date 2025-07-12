import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict

# parametros da simulacao
quantidade_inicial_particulas = 200
min_particulas = 100
max_particulas = 350

raio_particula:  np.float64 = 0.5  # nm
massa_particula: np.float64 = 1 * 1.66e-27  # kg (massa do hidrogenio)

tempo_entre_frames: np.float64 = 0.0002 # ns
tempo_coleta_dados: np.float64 = 30 * tempo_entre_frames # ns

temperatura_inicial: np.float64 = 300  # K
temperatura_maxima:  np.float64 = 1300 # K
temperatura_minima:  np.float64 = 100  # K

velocidade_maxima_histograma = 10000 # m/s
kb: np.float64 = 1.380649e-23  # J/K
quantas_particulas_adicionar: int = 10 # particulas adicionadas/removidas por clique
proporcao_considerada_na_temperatura: float = 0.99
temperatura_aquecimento: float = 5
intervalo_aquecimento: float = 0.01

LARGURA_TELA, ALTURA_TELA = 1200, 900  # tamanho da tela em pixels
PIXELS_POR_NM = 15  # fator de escala (10 pixels = 1 nm)
LARGURA_CAIXA: np.float64 = 40 # nm de largura
ALTURA_CAIXA:  np.float64 = 40 # nm de altura

RAIO_PARTICULA_PX = raio_particula * PIXELS_POR_NM
LARGURA_CAIXA_PX  = LARGURA_CAIXA * PIXELS_POR_NM
ALTURA_CAIXA_PX   = ALTURA_CAIXA * PIXELS_POR_NM

POS_X_CAIXA_PX, POS_Y_CAIXA_PX = 350 - int(LARGURA_CAIXA_PX/2), 450 - int(ALTURA_CAIXA_PX/2)  # posicao da caixa na tela em pixels

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (200, 200, 200)
VERDE = (0, 255, 0)
VERDE_ESCURO = (0, 200, 0)
CINZA_ESCURO = (100, 100, 100)
LARANJA = (255, 50, 0) 
AZUL = (0, 0, 255)
AZUL_CLARO = (100, 150, 255)

cor_particulas = AZUL_CLARO

# grid para otimizar colisões
tamanho_celulas: np.float64 = LARGURA_CAIXA / 10

#Particulas sao dicionarios da forma {x, y, vx, vy, massa, raio}
class SistemaParticulas:
    def __init__(self, quantidade_inicial_particulas: int, temperatura_inicial: int):
        self.particulas: list[dict] = []
        self.grid = defaultdict(list)

        self.momento_transferido_paredes = 0 # kg m/s

        self.inicializar_particulas(quantidade_inicial_particulas, temperatura_inicial)
        self.contagem_particulas = quantidade_inicial_particulas

    def velocidade_boltzmann(self, temperatura: np.float64):
        """Gera velocidade em m/s a partir da distribuição de Boltzmann"""
        sigma = np.sqrt(kb * temperatura / massa_particula)  # m/s
        # Amostra a magnitude usando a distribuicao de Rayleigh
        velocidade = np.random.rayleigh(sigma)
    
        return velocidade #random.gauss(media, dispersao)
    
    def inicializar_particulas(self, quantidade_particulas, temperatura):
        """Cria particulas iniciais nas condições iniciais dadas (quantidade e temperatura)"""
        for _ in range(quantidade_particulas):
            posicao_x = random.uniform(raio_particula, LARGURA_CAIXA - raio_particula)
            posicao_y = random.uniform(raio_particula, ALTURA_CAIXA  - raio_particula)

            velocidade = self.velocidade_boltzmann(temperatura)
            angulo = random.uniform(0, 2*np.pi)
            velocidade_x = velocidade * np.cos(angulo)
            velocidade_y = velocidade * np.sin(angulo)

            self.particulas.append({
                'posicao_x': posicao_x, # nm
                'posicao_y': posicao_y, # nm
                'vx': velocidade_x, # nm/ns = m/s
                'vy': velocidade_y, # nm/ns = m/s
                'raio': raio_particula, # nm
                'massa': massa_particula, # kg
            })
    
    def adicionar_particulas(self, quantidade_adicionar, temperatura):
        """Adiciona partículas com velocidades térmicas adequadas"""
        for _ in range(quantidade_adicionar):
            if len(self.particulas) >= max_particulas:
                break
                
            posicao_x = random.uniform(raio_particula, LARGURA_CAIXA - raio_particula)
            posicao_y = random.uniform(raio_particula, ALTURA_CAIXA  - raio_particula)
            
            # gera velocidades ajustadas pra temperatura
            velocidade = self.velocidade_boltzmann(temperatura)
            angulo = random.uniform(0, 2*np.pi)

            vx = velocidade * np.cos(angulo)
            vy = velocidade * np.sin(angulo)
            
            self.particulas.append({
                'posicao_x': posicao_x,
                'posicao_y': posicao_y,
                'vx': vx,
                'vy': vy,
                'raio': raio_particula,
                'massa': massa_particula,
            })

            self.contagem_particulas += 1
    
    def remover_particulas(self, quantidade):
        """Remove múltiplas partículas de uma vez"""
        for _ in range(quantidade):
            if len(self.particulas) <= min_particulas:
                break
            self.particulas.pop()
            self.contagem_particulas -= 1
    
    def atualizar_posicoes(self, tempo_decorrido: np.float64):
        """Atualiza a posição de todas as partículas do sistema, incluindo colisões com as paredes."""
        for p in self.particulas:
            x  = p['posicao_x']
            y  = p['posicao_y']
            m  = p['massa']
            r  = p['raio']
            vx = p['vx']
            vy = p['vy']

            # funcao horaria
            x_novo = x + vx * tempo_decorrido
            y_novo = y + vy * tempo_decorrido
            
            # Colisões com as paredes em x
            if x_novo - r < 0:
                x_novo = r
                p['vx'] = - vx
                self.momento_transferido_paredes += 2 * abs(vx * m)
            
            if x_novo + r > LARGURA_CAIXA:
                x_novo = LARGURA_CAIXA - r
                p['vx'] = - vx
                self.momento_transferido_paredes += 2 * abs(vx * m)
            
            # Colisões com as paredes em y
            if y_novo - r < 0:
                y_novo = r
                p['vy'] = - vy
                self.momento_transferido_paredes += 2 * abs(vy * m)
            
            if y_novo + r > ALTURA_CAIXA:
                y_novo = ALTURA_CAIXA - r
                p['vy'] = - vy
                self.momento_transferido_paredes += 2 * abs(vy * m)

            p['posicao_x'] = x_novo
            p['posicao_y'] = y_novo

    def colidir_particulas(self, indice_particula_1: int, indice_particula_2: int):
        """Checa se duas particulas estão em colisão e atualiza elas se necessário."""
        p1, p2 = self.particulas[indice_particula_1], self.particulas[indice_particula_2]

        x1, x2 = p1['posicao_x'], p2['posicao_x']
        y1, y2 = p1['posicao_y'], p2['posicao_y']
        m1, m2 = p1['massa'], p2['massa']
        r1, r2 = p1['raio'], p2['raio']
        vx1, vx2 = p1['vx'], p2['vx']
        vy1, vy2 = p1['vy'], p2['vy']

        dx = x1 - x2
        dy = y1 - y2
        distancia = np.sqrt(dx**2 + dy**2)

        if distancia < r1 + r2:
            # vetor normalizado de colisao
            nx, ny = dx / distancia, dy / distancia

            # velocidade relativa ao longo da normal
            v_rel = (vx1 - vx2) * nx + (vy1 - vy2) * ny

            if v_rel < 0:  # Aproximando uma da outra
                # Impulso
                j = -2 * v_rel / (1/m1 + 1/m2)

                # Atualiza velocidades
                p1['vx'] += j * nx / m1
                p1['vy'] += j * ny / m1
                p2['vx'] -= j * nx / m2
                p2['vy'] -= j * ny / m2

                # Corrige sobreposição (opcional)
                overlap = (r1 + r2) - distancia
                p1['posicao_x'] += overlap * nx * 0.5
                p1['posicao_y'] += overlap * ny * 0.5
                p2['posicao_x'] -= overlap * nx * 0.5
                p2['posicao_y'] -= overlap * ny * 0.5
    
    def rodar_colisoes(self):
        """Calcula todas as colisões entre as partículas do sistema"""
        for celula, indices_particulas in self.grid.items():
            # verifica particulas nesta celula
            for i in range(len(indices_particulas)):
                for j in range(i + 1, len(indices_particulas)):
                    self.colidir_particulas(indices_particulas[i], indices_particulas[j])
            
            # verifica celulas vizinhas
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                celula_vizinha = (celula[0] + dx, celula[1] + dy)

                if celula_vizinha in self.grid:
                    for i in indices_particulas:
                        for j in self.grid[celula_vizinha]:
                            if i < j:
                                self.colidir_particulas(i, j)

    def atualizar_grid(self):
        """Salva o índice de cada particula em cada celula do self.grade"""
        self.grid.clear()
        for indice_particula, particula in enumerate(self.particulas):
            # converte posicao para coordenadas da grade
            grid_x = int(particula['posicao_x'] // tamanho_celulas)
            grid_y = int(particula['posicao_y'] // tamanho_celulas)
            self.grid[(grid_x, grid_y)].append(indice_particula)
    
    def atualizar_sistema(self, tempo_decorrido):
        """Atualiza as posições, atualiza o grid e depois roda as colisões"""
        self.atualizar_posicoes(tempo_decorrido)
        self.atualizar_grid()
        self.rodar_colisoes()

    @property
    def energias_cineticas(self) -> list[np.float64]:
        """Retorna uma lista com as energias cinéticas de cada partícula"""
        lista_energias_cineticas = []

        for p in self.particulas:
            velocidade_quadrada = p['vx']**2 + p['vy']**2
            massa = p['massa']
            energia_cinetica = 0.5 * massa * velocidade_quadrada
            lista_energias_cineticas.append(energia_cinetica)

        return lista_energias_cineticas

    @property
    def velocidades(self) -> list[np.float64]:
        lista_velocidades = []

        for p in self.particulas:
            velocidade = np.sqrt(p['vx']**2 + p['vy']**2)
            lista_velocidades.append(velocidade)

        return lista_velocidades
    
    @property
    def temperatura_atual(self) -> np.float64:
        lista_energias_cineticas = self.energias_cineticas
        energia_media = np.mean(lista_energias_cineticas)
        temperatura = energia_media / kb
        return temperatura
    
    @property
    def temperatura_medida(self) -> np.float64:
        lista_energias_cineticas = self.energias_cineticas
        energia_media_medida = np.mean(lista_energias_cineticas[:int(proporcao_considerada_na_temperatura * self.contagem_particulas)])
        temperatura = energia_media_medida / kb
        return temperatura
    
    def pressao2d(self, tempo_decorrido) -> np.float64:
        area2d_paredes = 2 * (LARGURA_CAIXA + ALTURA_CAIXA) # nm
        forca_aplicada_paredes = self.momento_transferido_paredes / tempo_decorrido # GN
        pressao2d = forca_aplicada_paredes / area2d_paredes * 1e18 # N/m
        self.momento_transferido_paredes = 0
        return pressao2d
    
    def aquecer(self, delta_temperatura):
        """Ajusta a temperatura do sistema pelo fator especificado"""
        if delta_temperatura == 0:
            return
        
        temperatura_atual = self.temperatura_atual
        temperatura_nova = temperatura_atual + delta_temperatura
        
        # Evita temperaturas negativas
        if temperatura_nova < temperatura_minima:
            temperatura_nova = temperatura_minima
        
        elif temperatura_nova > temperatura_maxima:
            temperatura_nova = temperatura_maxima
        
        # Fator de escala de velocidade
        if temperatura_atual > 0:
            fator = np.sqrt(temperatura_nova / temperatura_atual)
        else:
            fator = 1
        
        # Ajusta todas as velocidades
        for p in self.particulas:
            p['vx'] *= fator
            p['vy'] *= fator

# plotagem do histograma
plt.ioff()
fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
fig.subplots_adjust(left=0.15,bottom=0.2)
canvas = FigureCanvasAgg(fig)

def desenhar_histograma(quantidade_particulas, velocidades, temperatura_atual) -> pygame.Surface:
    """Retorna uma superficie com o histograma desenhado"""
    ax.clear()
    if velocidades:
        bins = np.linspace(0, velocidade_maxima_histograma, 20)
        contagens, bins, patches = ax.hist(velocidades, bins=bins, color='blue', alpha=0.7, weights=np.ones(quantidade_particulas)/quantidade_particulas)

        # distribuicao de Maxwell-Boltzmann
        kT = temperatura_atual * kb
        m = massa_particula
        x = np.linspace(0, velocidade_maxima_histograma, 100)
        dx = bins[1] - bins[0]

        maxwell = m*x/kT * np.exp(-m*x**2/(2*kT)) * dx 
        ax.plot(x, maxwell, 'r-', linewidth=2)
    
    ax.set_title('Distribuição de Velocidades')
    ax.set_xlabel('Velocidade (m/s)')
    ax.set_ylabel('Frequência')
    ax.set_xlim(0, velocidade_maxima_histograma)
    ax.set_ylim(0,0.4)
    
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    return pygame.image.fromstring(raw_data, size, "RGB")

# loop simulacao
pygame.init()

# inicializa o sistema de particulas
sistema_particulas = SistemaParticulas(quantidade_inicial_particulas, temperatura_inicial)

# configura a tela
screen = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Simulador de Partículas de Gás Ideal")
clock = pygame.time.Clock()
fonte = pygame.font.SysFont('Arial', 16)
fonte_titulo = pygame.font.SysFont('Arial', 24, bold=True)

# funcao pra desenhar botoes
def desenhar_botao(x, y, largura, altura, texto, ativo=True, cor=VERDE_ESCURO):
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
    aquecendo  = False
    resfriando = False
    ultimo_tempo_aquecimento = 0

    # estatisticas iniciais
    tempo_ultima_coleta = 0
    ultima_pressao_medida = sistema_particulas.pressao2d(tempo_coleta_dados)

    estatisticas = [
        f"Partículas: {sistema_particulas.contagem_particulas}",
        f"Raio da Partícula: {raio_particula} nm",
        f"Massa da Partícula: {massa_particula:.2e} kg",
        f"Tamanho da Caixa: {LARGURA_CAIXA} nm x {ALTURA_CAIXA} nm",
        f"Temperatura Atual: {sistema_particulas.temperatura_atual - 273.15:.1f} °C",
        f"Pressão: {ultima_pressao_medida:.2e} N/m"
    ]
    
    # retangulos dos botoes
    retangulo_botao_adicionar = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_remover   = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_aquecer   = pygame.Rect(0, 0, 0, 0)
    retangulo_botao_resfriar  = pygame.Rect(0, 0, 0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # botao direito
                posicao_mouse = pygame.mouse.get_pos()

                if retangulo_botao_adicionar.collidepoint(posicao_mouse):
                    sistema_particulas.adicionar_particulas(quantas_particulas_adicionar, sistema_particulas.temperatura_atual)

                elif retangulo_botao_remover.collidepoint(posicao_mouse):
                    sistema_particulas.remover_particulas(quantas_particulas_adicionar)

                elif retangulo_botao_aquecer.collidepoint(posicao_mouse) and sistema_particulas.temperatura_atual < temperatura_maxima:
                    aquecendo  = True
                    resfriando = False

                elif retangulo_botao_resfriar.collidepoint(posicao_mouse) and sistema_particulas.temperatura_atual > temperatura_minima:
                    aquecendo  = False
                    resfriando = True
                
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Botão esquerdo do mouse liberado
                    aquecendo = False
                    resfriando = False
        
        tempo_atual = pygame.time.get_ticks() / 1000  # Converte para segundos
        if aquecendo and (tempo_atual - ultimo_tempo_aquecimento > intervalo_aquecimento):
            sistema_particulas.aquecer(temperatura_aquecimento)
            ultimo_tempo_aquecimento = tempo_atual
        
        if resfriando and (tempo_atual - ultimo_tempo_aquecimento > intervalo_aquecimento):
            sistema_particulas.aquecer(-temperatura_aquecimento)
            ultimo_tempo_aquecimento = tempo_atual

        tempo_ultima_coleta += tempo_entre_frames
        sistema_particulas.atualizar_sistema(tempo_entre_frames)
        superficie_histograma = desenhar_histograma(sistema_particulas.contagem_particulas, sistema_particulas.velocidades, sistema_particulas.temperatura_atual)
        
        # limpa a tela
        screen.fill(PRETO)
        
        # desenha titulos
        screen.blit(fonte_titulo.render("Simulação", True, BRANCO), (50, 50))
        screen.blit(fonte_titulo.render("Controles", True, BRANCO), (700, 50))
        screen.blit(fonte_titulo.render("Dados", True, BRANCO), (700, 300))

        # desenha caixa de simulacao e particulas
        for p in sistema_particulas.particulas:
            pixel_central_x = int(POS_X_CAIXA_PX + p['posicao_x'] * PIXELS_POR_NM)
            pixel_central_y = int(POS_Y_CAIXA_PX + p['posicao_y'] * PIXELS_POR_NM)
            raio_em_pixels  = int(p['raio'] * PIXELS_POR_NM)
            pygame.draw.circle(screen, cor_particulas, center=(pixel_central_x, pixel_central_y), radius=raio_em_pixels)

        pygame.draw.rect(screen, BRANCO, (POS_X_CAIXA_PX, POS_Y_CAIXA_PX, LARGURA_CAIXA_PX, ALTURA_CAIXA_PX), 2)

        # desenha botoes de controle
        retangulo_botao_adicionar = desenhar_botao(700, 100, 150, 40, f"Adicionar {quantas_particulas_adicionar} Partículas", len(sistema_particulas.particulas) < max_particulas)
        retangulo_botao_remover   = desenhar_botao(700, 150, 150, 40, f"Remover {quantas_particulas_adicionar} Partículas",   len(sistema_particulas.particulas) > min_particulas)
        retangulo_botao_aquecer   = desenhar_botao(700, 200, 150, 40, f"Aquecer (+{temperatura_aquecimento} °C)",  sistema_particulas.temperatura_atual < 0.99 * temperatura_maxima, VERDE_ESCURO if not aquecendo  else LARANJA)
        retangulo_botao_resfriar  = desenhar_botao(700, 250, 150, 40, f"Resfriar (-{temperatura_aquecimento} °C)", sistema_particulas.temperatura_atual > 1.01 * temperatura_minima, VERDE_ESCURO if not resfriando else AZUL)

        # estatisticas
        if tempo_ultima_coleta > tempo_coleta_dados:
            tempo_ultima_coleta = 0
            ultima_pressao_medida = sistema_particulas.pressao2d(tempo_coleta_dados)
        
        estatisticas = [
            f"Partículas: {sistema_particulas.contagem_particulas}",
            f"Raio da Partícula: {raio_particula} nm",
            f"Massa da Partícula: {massa_particula:.2e} kg",
            f"Tamanho da Caixa: {LARGURA_CAIXA} nm x {ALTURA_CAIXA} nm",
            f"Temperatura Atual: {sistema_particulas.temperatura_medida - 273.15:.1f} °C",
            f"Pressão: {ultima_pressao_medida:.2e} N/m"
        ]
        
        for i, estatistica in enumerate(estatisticas):
            screen.blit(fonte.render(estatistica, True, BRANCO), (700, 350 + i * 25))

        # Desenha histograma
        screen.blit(superficie_histograma, (700, 550))
        
        pygame.display.flip()
        clock.tick(120)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()