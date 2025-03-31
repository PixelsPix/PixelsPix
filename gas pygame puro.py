import sys
import pygame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# a parede é um cubo/quadrado determinado pelo tamanho do lado do cubo/quadrado e numero de dimensoes
tamanho_parede: float      = 700 #nm
numero_dimensoes: int      = 2
quantidade_particulas: int = 100

raio:  float = 5 #0.250 #nm
massa: float = 1 #* 1.66053966e-15 #pg (picogramas)

velocidade_maxima: float = 100 #nm/ns

tempo_global = 0 #ns
tempo_sample = 0.02 #ns

pygame.init()

# tela
WIDTH, HEIGHT = tamanho_parede, tamanho_parede
tela = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador de Partículas")

# Cores
preto = (0, 0, 0)
branco = (255, 255, 255)

# classe particula
class Particula:
    def __init__(self, raio: float, massa: float, vetor_posicao: list[float], vetor_velocidade: list[float]) -> None:
        self.massa = float(np.float64(massa))
        self.raio  = float(np.float64(raio))
        self.vetor_posicao    = np.array(vetor_posicao, dtype= np.float64)
        self.vetor_velocidade = np.array(vetor_velocidade, dtype= np.float64)

    @property
    def momento(self) -> npt.NDArray[np.float64]:
        """Retorna um ndarray dado por: massa * velocidade"""
        return self.massa * self.vetor_velocidade
    
    @property
    def energia_cinetica(self) -> np.float64:
        """Retorna um float64 dado por: massa * <velocidade, velocidade>"""
        return self.massa * np.vdot(self.vetor_velocidade, self.vetor_velocidade)
    
    def atualizar_posicao(self, tempo_decorrido: np.float64) -> None:
        """Usa a equação horária do movimento da partícula para atualizar sua posição um delta_t e colide elasticamente com as paredes"""
        self.vetor_posicao += self.vetor_velocidade * tempo_decorrido

        # Colisões com paredes (eixo X)
        if self.vetor_posicao[0] - self.raio <= 0:
            self.vetor_posicao[0] = self.raio
            self.vetor_velocidade[0] *= -1
        elif self.vetor_posicao[0] + self.raio >= WIDTH:
            self.vetor_posicao[0] = WIDTH - self.raio
            self.vetor_velocidade[0] *= -1

        # Colisões com paredes (eixo Y)
        if self.vetor_posicao[1] - self.raio <= 0:
            self.vetor_posicao[1] = self.raio
            self.vetor_velocidade[1] *= -1
        elif self.vetor_posicao[1] + self.raio >= HEIGHT:
            self.vetor_posicao[1] = HEIGHT - self.raio
            self.vetor_velocidade[1] *= -1

    def desenhar(self, superficie):
        pygame.draw.circle(superficie, preto, (int(self.vetor_posicao[0]), int(self.vetor_posicao[1])),  self.raio)

    def __str__(self) -> str:
        return "Raio {}, massa {}, posicao {} e velocidade {}".format(self.raio, self.massa, self.vetor_posicao, self.vetor_velocidade)
    
    def __repr__(self) -> str:
        return "Particula({}, {}, {}, {})".format(self.raio, self.massa, self.vetor_posicao.tolist(), self.vetor_velocidade.tolist())


# calcula nova velocidade das particulas depois de colidirem
def velocidades_apos_colisao(massas: list[float], velocidades_normais: list[np.float64]) -> list[np.float64]:
    """
    No momento da colisão, podemos decompor as velocidades das partículas em suas componentes normal ao impacto (v1 e v2) e tangencial. Somente a velocidade normal é alterada.\n
    v1_nova = ( (m1 - m2) v1 +   2 m2    v2 ) / (m1 + m2)\n
    v2_nova = (   2 m1    v1 + (m1 - m2) v2 ) / (m1 + m2)

    Returns
    -------
    lista_velocidades_novas: list[float64]
        É uma lista formada pelas velocidades normais novas calculadas, na forma  [v1_nova, v2_nova]
    """
    m1, m2 = massas
    v1, v2 = velocidades_normais
    
    diff_massa = m1 - m2
    soma_massa = m1 + m2

    velocidade_1_nova = (diff_massa * v1 +    2*m2    * v2) / soma_massa
    velocidade_2_nova = (   2*m1    * v1 + diff_massa * v2) / soma_massa

    return [velocidade_1_nova, velocidade_2_nova]

# colide e atualiza as particulas para a colisao
def colisao(particula_1: Particula, particula_2: Particula) -> None: # O(1)
    """
    Atualiza as velocidades das partículas durante uma colisão. Usar somente se a colisão já for confirmada.\n
    Para mudar a velocidade normal sem alterar a velocidade tangencial, o vetor normal é usado para alterar somente em relação à normal:\n\n
    velocidade_normal     = <velocidade_antes, vetor_normal>\n

    velocidade_antes  = velocidade_tangencial + velocidade_normal\n
    velocidade_depois = velocidade_tangencial + velocidade_normal_nova\n

    velocidade_nova = velocidade_antes + (v_normal_novo - v_normal_antes) * vetor_normal
    """
    vetor_distancia = particula_2.vetor_posicao - particula_1.vetor_posicao
    distancia = np.linalg.norm(vetor_distancia)

    if distancia < particula_1.raio + particula_2.raio:
        vetor_normal = vetor_distancia / distancia

        massa_1 = particula_1.massa
        massa_2 = particula_2.massa 

        v1_normal = np.vdot(particula_1.vetor_velocidade, vetor_normal)
        v2_normal = np.vdot(particula_2.vetor_velocidade, vetor_normal)

        v1_normal_novo, v2_normal_novo = velocidades_apos_colisao(massas = [massa_1, massa_2], velocidades_normais = [v1_normal, v2_normal])

        # Atualização das velocidades
        particula_1.vetor_velocidade += (v1_normal_novo - v1_normal) * vetor_normal
        particula_2.vetor_velocidade += (v2_normal_novo - v2_normal) * vetor_normal

        # Afasta as particulas pra evitar grudarem
        overlap = (particula_1.raio + particula_2.raio) - distancia
        particula_1.vetor_posicao -= (overlap/2) * vetor_normal
        particula_2.vetor_posicao += (overlap/2) * vetor_normal

# o RNG é um gerador aleatorio de float64 entre [0, 1)
rng = np.random.default_rng()
def vetor_aleatorio(valor_minimo: float, valor_maximo: float, dimensoes: int) -> npt.NDArray[np.float64]:
    """
    Gera um vetor (ndarray[float64]) com as dimensões dadas, com valores entre valor_minimo e valor_maximo, não inclusivo. Isso é para não ter problemas como iniciar na posição 0 (em colisão com parede)
    
    Returns
    -------
    vetor_aleatorio: ndarray[float64]
        Um vetor de dimensões dadas entre valor_minimo e valor_maximo, não inclusivo.
    """
    vetor_zero_a_um = rng.random(dimensoes)
    
    while min(vetor_zero_a_um) == 0:
        vetor_zero_a_um[vetor_zero_a_um == 0] = rng.random() # remapeia valores nulos, se tiver
    
    vetor_aleatorio = (valor_maximo - valor_minimo) * vetor_zero_a_um + valor_minimo

    return vetor_aleatorio

# criador de particula com posicao e velocidade aleatorias
def criar_particula_aleatoria(raio: float, massa: float) -> Particula:
    """
    Returns
    -------
    particula_criada: Particula
        Uma partícula com raio e massa dados pelo usuário e a posição e a velocidade aleatorizadas (distribuição uniforme)
    """
    posicao_minima    = 0 + raio
    posicao_maxima    = tamanho_parede - raio
    velocidade_minima = - velocidade_maxima # velocidades podem ser positivas ou negativas

    posicao_criada    = vetor_aleatorio(posicao_minima, posicao_maxima, numero_dimensoes)
    velocidade_criada = vetor_aleatorio(velocidade_minima, velocidade_maxima, numero_dimensoes)

    particula_criada = Particula(raio, massa, posicao_criada, velocidade_criada)
    return particula_criada

# criador da lista de partículas
def criar_particulas_iniciais(quantidade_particulas: int, raio: float, massa: float) -> list[Particula]: 
    """Usa a função criar_particula_aleatoria para criar a uma lista com a quantidade de partículas desejada, ajustando as velocidades de todas pro momento ser zero no final"""
    # primeiro monta uma lista inicial das particulas
    lista_particulas = [criar_particula_aleatoria(raio, massa) for i in range(quantidade_particulas)]
    
    lista_velocidades = [particula.vetor_velocidade for particula in lista_particulas]
    velocidade_para_somar = - np.sum(lista_velocidades, axis=0) / (quantidade_particulas - 1)

    for i in range(quantidade_particulas - 1):
        particula = lista_particulas[i]
        nova_velocidade = particula.vetor_velocidade + velocidade_para_somar
        lista_particulas[i] = Particula(raio, massa, particula.vetor_posicao, nova_velocidade)

    # e a ultima particula recebe o valor que falta para o momento ser zero
    lista_velocidades = [particula.vetor_velocidade for particula in lista_particulas]

    posicao_ultima_particula = lista_particulas[-1].vetor_posicao
    velocidade_ultima_particula = lista_particulas[-1].vetor_velocidade - np.sum(lista_velocidades, axis=0)

    lista_particulas[-1] = Particula(raio, massa, posicao_ultima_particula, velocidade_ultima_particula)

    return lista_particulas

# Particle system
particulas_simulacao = criar_particulas_iniciais(quantidade_particulas, raio, massa)

# declara essas variaveis
energias_cineticas = [particula.energia_cinetica for particula in particulas_simulacao]
momento_colisoes_parede = 0
energia_cinetica_media = np.mean([particula.energia_cinetica for particula in particulas_simulacao])

# Loop de simulacao
clock = pygame.time.Clock()
running = True

while running:
    tela.fill(branco)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Cria partícula aleatoria
            particulas_simulacao.append(criar_particula_aleatoria(raio, massa))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Cria 10 particulas aleatorias
                for _ in range(10):
                    particulas_simulacao.append(criar_particula_aleatoria(raio, massa))
    
    # atualiza posicoes
    for particula in particulas_simulacao:
        particula.atualizar_posicao(0.02)
    
    # Checar colisoes O(n^2)
    for i in range(len(particulas_simulacao)):
        for j in range(i + 1, len(particulas_simulacao)):
            colisao(particulas_simulacao[i], particulas_simulacao[j])
    
    # Desenha as particulas na tela
    for particula in particulas_simulacao:
        particula.desenhar(tela)
    
    # Contagem de particulas
    font = pygame.font.SysFont('Arial', 20)
    contagem_particulas_texto = font.render(f'Particulas: {len(particulas_simulacao)}', True, preto)
    energia_cinetica_media_texto = font.render(f'Energia cinética média: {energia_cinetica_media}', True, preto)
    tela.blit(contagem_particulas_texto, (10, 10))
    tela.blit(energia_cinetica_media_texto, (10, 30))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()