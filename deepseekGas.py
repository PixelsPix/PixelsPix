import pygame
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import defaultdict

# Inicializa o pygame
pygame.init()

# Parâmetros da simulação (todos em nanômetros)
LARGURA_PX, ALTURA_PX = 1200, 900  # Tamanho da tela em pixels
PIXELS_POR_NM = 10  # Fator de escala (10 pixels = 1 nm)
RAIO_PARTICULA_NM = 0.5  # Raio de 0.5 nm (molécula de gás típica)
RAIO_PARTICULA_PX = RAIO_PARTICULA_NM * PIXELS_POR_NM
MASSA_PARTICULA = 1.66e-27  # kg (massa do átomo de hidrogênio)
MIN_PARTICULAS = 10
MAX_PARTICULAS = 500
LARGURA_CAIXA_NM = 60  # 60 nm de largura
ALTURA_CAIXA_NM = 50  # 50 nm de altura
LARGURA_CAIXA_PX = LARGURA_CAIXA_NM * PIXELS_POR_NM
ALTURA_CAIXA_PX = ALTURA_CAIXA_NM * PIXELS_POR_NM
POS_X_CAIXA_PX, POS_Y_CAIXA_PX = 50, 150  # Posição na tela em pixels
TEMPERATURA_INICIAL = 300  # Kelvin
CONSTANTE_BOLTZMANN = 1.380649e-23  # J/K
COR_PARTICULA = (100, 150, 255)
LOTE_PARTICULAS = 10

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
CINZA = (200, 200, 200)
VERDE = (0, 255, 0)
CINZA_ESCURO = (100, 100, 100)
LARANJA = (255, 165, 0)

# Configura a tela
tela = pygame.display.set_mode((LARGURA_PX, ALTURA_PX))
pygame.display.set_caption("Simulador de Partículas de Gás Ideal (Nanoescala)")
relogio = pygame.time.Clock()
fonte = pygame.font.SysFont('Arial', 16)
fonte_titulo = pygame.font.SysFont('Arial', 24, bold=True)

# Grade espacial para otimizar colisões (em unidades nm)
TAMANHO_CELULA_NM = RAIO_PARTICULA_NM * 4
TAMANHO_CELULA_PX = TAMANHO_CELULA_NM * PIXELS_POR_NM
LARGURA_GRADE = int(LARGURA_CAIXA_NM / TAMANHO_CELULA_NM) + 1
ALTURA_GRADE = int(ALTURA_CAIXA_NM / TAMANHO_CELULA_NM) + 1

class SistemaParticulas:
    def __init__(self):
        self.particulas = []
        self.grade = defaultdict(list)
        self.colisoes_parede = 0
        self.ultimo_tempo_colisao = pygame.time.get_ticks()
        self.pressao = 0
        self.temperatura_atual = TEMPERATURA_INICIAL
        self.temperatura_alvo = TEMPERATURA_INICIAL
        self.energia_max = 8e-20  # Valor máximo fixo para o eixo x do histograma
        self.inicializar_particulas(200, TEMPERATURA_INICIAL)

    def velocidade_boltzmann(self, temperatura):
        """Gera velocidade em nm/ps a partir da distribuição de Maxwell-Boltzmann"""
        sigma = np.sqrt(temperatura * CONSTANTE_BOLTZMANN / MASSA_PARTICULA) * 1e-3  # Converte para nm/ps
        return random.gauss(0, sigma)

    def inicializar_particulas(self, num_particulas, temperatura):
        self.particulas = []
        for _ in range(num_particulas):
            # Posições aleatórias dentro da caixa (em coordenadas de pixel)
            x_px = random.uniform(POS_X_CAIXA_PX + RAIO_PARTICULA_PX, 
                                 POS_X_CAIXA_PX + LARGURA_CAIXA_PX - RAIO_PARTICULA_PX)
            y_px = random.uniform(POS_Y_CAIXA_PX + RAIO_PARTICULA_PX,
                                 POS_Y_CAIXA_PX + ALTURA_CAIXA_PX - RAIO_PARTICULA_PX)
            
            # Gera velocidades aleatórias em nm/ps
            velocidade = self.velocidade_boltzmann(temperatura)
            angulo = random.uniform(0, 2*np.pi)
            vx = velocidade * np.cos(angulo)
            vy = velocidade * np.sin(angulo)
            
            self.particulas.append({
                'x': x_px, 'y': y_px,  # Posição em pixels
                'vx': vx, 'vy': vy,    # Velocidade em nm/ps
                'raio': RAIO_PARTICULA_PX,
                'raio_nm': RAIO_PARTICULA_NM,
                'massa': MASSA_PARTICULA,
            })

    def atualizar_grade(self):
        self.grade.clear()
        for idx, p in enumerate(self.particulas):
            # Converte posição para coordenadas da grade
            grade_x = int((p['x'] - POS_X_CAIXA_PX) / PIXELS_POR_NM) // TAMANHO_CELULA_NM
            grade_y = int((p['y'] - POS_Y_CAIXA_PX) / PIXELS_POR_NM) // TAMANHO_CELULA_NM
            self.grade[(grade_x, grade_y)].append(idx)

    def tratar_colisoes(self):
        for celula, indices in self.grade.items():
            # Verifica partículas nesta célula
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    self.verificar_colisao_particulas(indices[i], indices[j])
            
            # Verifica células vizinhas
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                celula_vizinha = (celula[0] + dx, celula[1] + dy)
                if celula_vizinha in self.grade:
                    for i in indices:
                        for j in self.grade[celula_vizinha]:
                            if i < j:
                                self.verificar_colisao_particulas(i, j)

    def verificar_colisao_particulas(self, i, j):
        p1, p2 = self.particulas[i], self.particulas[j]
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
        energia_cinetica_total = 0
        transferencia_momento_total = 0
        
        # Atualiza posições (converte velocidade de nm/ps para pixels/quadro)
        for p in self.particulas:
            p['x'] += p['vx'] * dt * PIXELS_POR_NM
            p['y'] += p['vy'] * dt * PIXELS_POR_NM
            
            # Colisões com as paredes (em coordenadas de pixel)
            if p['x'] - p['raio'] < POS_X_CAIXA_PX:
                p['x'] = POS_X_CAIXA_PX + p['raio']
                p['vx'] = -p['vx']
                self.colisoes_parede += 1
                transferencia_momento_total += 2 * abs(p['vx'] * p['massa'])
            
            if p['x'] + p['raio'] > POS_X_CAIXA_PX + LARGURA_CAIXA_PX:
                p['x'] = POS_X_CAIXA_PX + LARGURA_CAIXA_PX - p['raio']
                p['vx'] = -p['vx']
                self.colisoes_parede += 1
                transferencia_momento_total += 2 * abs(p['vx'] * p['massa'])
            
            if p['y'] - p['raio'] < POS_Y_CAIXA_PX:
                p['y'] = POS_Y_CAIXA_PX + p['raio']
                p['vy'] = -p['vy']
                self.colisoes_parede += 1
                transferencia_momento_total += 2 * abs(p['vy'] * p['massa'])
            
            if p['y'] + p['raio'] > POS_Y_CAIXA_PX + ALTURA_CAIXA_PX:
                p['y'] = POS_Y_CAIXA_PX + ALTURA_CAIXA_PX - p['raio']
                p['vy'] = -p['vy']
                self.colisoes_parede += 1
                transferencia_momento_total += 2 * abs(p['vy'] * p['massa'])
            
            # Calcula energia cinética (converte velocidade para m/s)
            vx_mps = p['vx'] * 1e3  # nm/ps para m/s
            vy_mps = p['vy'] * 1e3
            ke = 0.5 * p['massa'] * (vx_mps**2 + vy_mps**2)
            energia_cinetica_total += ke
        
        # Atualiza grade e trata colisões
        self.atualizar_grade()
        self.tratar_colisoes()
        
        # Calcula pressão (área da parede em metros)
        tempo_atual = pygame.time.get_ticks()
        if tempo_atual - self.ultimo_tempo_colisao > 0:
            tempo_decorrido = (tempo_atual - self.ultimo_tempo_colisao) / 1000
            area_parede_m = 2 * (LARGURA_CAIXA_NM + ALTURA_CAIXA_NM) * 1e-9
            self.pressao = transferencia_momento_total / (tempo_decorrido * area_parede_m)
            self.ultimo_tempo_colisao = tempo_atual
        
        # Calcula temperatura
        if self.particulas:
            self.temperatura_atual = (energia_cinetica_total / len(self.particulas)) / CONSTANTE_BOLTZMANN
            
            # Ajusta velocidades para corresponder à temperatura alvo
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
                
            x_px = random.uniform(POS_X_CAIXA_PX + RAIO_PARTICULA_PX,
                                POS_X_CAIXA_PX + LARGURA_CAIXA_PX - RAIO_PARTICULA_PX)
            y_px = random.uniform(POS_Y_CAIXA_PX + RAIO_PARTICULA_PX,
                                POS_Y_CAIXA_PX + ALTURA_CAIXA_PX - RAIO_PARTICULA_PX)
            
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

# Inicializa os sistemas
sistema_particulas = SistemaParticulas()

# Para plotagem
plt.ioff()
fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
fig.subplots_adjust(bottom=0.2)
canvas = FigureCanvasAgg(fig)

def desenhar_histograma(energias_cineticas):
    ax.clear()
    if energias_cineticas:
        bins = np.linspace(0, sistema_particulas.energia_max, 20)
        contagens, bins, patches = ax.hist(energias_cineticas, bins=bins, color='blue', alpha=0.7)
        
        # Sobreposição da distribuição de Maxwell-Boltzmann
        if len(energias_cineticas) > 1:
            kT = sistema_particulas.temperatura_atual * CONSTANTE_BOLTZMANN
            x = np.linspace(0, sistema_particulas.energia_max, 100)
            # Calcula largura do bin para normalização adequada
            largura_bin = bins[1] - bins[0]
            # Distribuição de Maxwell-Boltzmann normalizada corretamente
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
    cor_btn = cor if ativo else CINZA_ESCURO
    pygame.draw.rect(tela, cor_btn, (x, y, largura, altura))
    pygame.draw.rect(tela, BRANCO, (x, y, largura, altura), 2)
    texto_surf = fonte.render(texto, True, BRANCO)
    texto_rect = texto_surf.get_rect(center=(x + largura/2, y + altura/2))
    tela.blit(texto_surf, texto_rect)
    return pygame.Rect(x, y, largura, altura)

def main():
    executando = True
    pausado = False
    
    # Retângulos dos botões
    rect_botao_adicionar = pygame.Rect(0, 0, 0, 0)
    rect_botao_remover = pygame.Rect(0, 0, 0, 0)
    rect_botao_aquecer = pygame.Rect(0, 0, 0, 0)
    rect_botao_resfriar = pygame.Rect(0, 0, 0, 0)
    
    while executando:
        dt = 0.2
        
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                executando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE:
                    pausado = not pausado
                elif evento.key == pygame.K_ESCAPE:
                    executando = False
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                pos_mouse = pygame.mouse.get_pos()
                if rect_botao_adicionar.collidepoint(pos_mouse):
                    sistema_particulas.adicionar_particulas(LOTE_PARTICULAS, sistema_particulas.temperatura_alvo)
                elif rect_botao_remover.collidepoint(pos_mouse):
                    sistema_particulas.remover_particulas(LOTE_PARTICULAS)
                elif rect_botao_aquecer.collidepoint(pos_mouse):
                    sistema_particulas.temperatura_alvo += 50
                elif rect_botao_resfriar.collidepoint(pos_mouse) and sistema_particulas.temperatura_alvo > 50:
                    sistema_particulas.temperatura_alvo -= 50
        
        if not pausado:
            energia_total = sistema_particulas.atualizar(dt)
            energias_cineticas = [0.5 * p['massa'] * ((p['vx']*1e3)**2 + (p['vy']*1e3)**2) for p in sistema_particulas.particulas]
            superficie_hist = desenhar_histograma(energias_cineticas)
        
        # Desenha tudo
        tela.fill(PRETO)
        
        # Desenha títulos
        tela.blit(fonte_titulo.render("Simulação", True, BRANCO), (50, 50))
        tela.blit(fonte_titulo.render("Dados", True, BRANCO), (700, 50))
        tela.blit(fonte_titulo.render("Histograma", True, BRANCO), (700, 300))
        
        # Desenha caixa de simulação e partículas
        pygame.draw.rect(tela, BRANCO, (POS_X_CAIXA_PX, POS_Y_CAIXA_PX, LARGURA_CAIXA_PX, ALTURA_CAIXA_PX), 2)
        for p in sistema_particulas.particulas:
            pygame.draw.circle(tela, COR_PARTICULA, (int(p['x']), int(p['y'])), int(p['raio']))
        
        # Desenha botões de controle
        rect_botao_adicionar = desenhar_botao(700, 100, 150, 40, "Adicionar 10 Partículas", len(sistema_particulas.particulas) < MAX_PARTICULAS)
        rect_botao_remover = desenhar_botao(700, 150, 150, 40, "Remover 10 Partículas", len(sistema_particulas.particulas) > MIN_PARTICULAS)
        rect_botao_aquecer = desenhar_botao(700, 200, 150, 40, "Aquecer (+50K)", cor=LARANJA)
        rect_botao_resfriar = desenhar_botao(700, 250, 150, 40, "Resfriar (-50K)", sistema_particulas.temperatura_alvo > 50)
        
        # Mostra estatísticas (agora mostrando unidades nm)
        estatisticas = [
            f"Partículas: {len(sistema_particulas.particulas)}",
            f"Raio da Partícula: {RAIO_PARTICULA_NM} nm",
            f"Massa da Partícula: {MASSA_PARTICULA:.1e} kg",
            f"Tamanho da Caixa: {LARGURA_CAIXA_NM}x{ALTURA_CAIXA_NM} nm",
            f"Temp. Alvo: {sistema_particulas.temperatura_alvo:.1f} K",
            f"Temp. Atual: {sistema_particulas.temperatura_atual:.1f} K",
            f"Pressão: {sistema_particulas.pressao:.2e} Pa",
            f"Espaço: Pausar/Continuar | Esc: Sair"
        ]
        
        for i, estatistica in enumerate(estatisticas):
            tela.blit(fonte.render(estatistica, True, BRANCO), (700, 350 + i * 25))
        
        # Desenha histograma
        if not pausado:
            tela.blit(superficie_hist, (700, 550))
        
        pygame.display.flip()
        relogio.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()