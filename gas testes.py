import pygame
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# a parede é um cubo/quadrado determinado pelo tamanho do lado do cubo/quadrado e numero de dimensoes
tamanho_parede: float      = 100 #nm
numero_dimensoes: int      = 2
quantidade_particulas: int = 300

raio:  float = 1 #0.250 #nm
massa: float = 1 #* 1.66053966e-15 #pg (picogramas)

velocidade_maxima: float = 100 #nm/ns
iteracaoMaxima: int      = 100

tempo_minimo_para_evento = 1e-12 # so pra ignorar eventos repetidos, caso aparecam

tempo_global = 0 #ns
tempo_sample = 0.02 #ns

numero_muito_grande = np.float64(999999999) # um valor muito grande para retornar se não for possível colisões
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
        """Usa a equação horária do movimento da partícula para atualizar sua posição um delta_t"""
        self.vetor_posicao += self.vetor_velocidade * tempo_decorrido
    
    def reflexao(self, dimensao_reflexão: int) -> None:
        """
        Edita o vetor velocidade, invertendo o sentido da velocidade na dimensão dada, como uma colisão elástica com a parede. As dimensões usam índices negativos para evitar confusão com os índices das partículas.

        Parameters
        ----------
        dimensao_reflexao : int
            -1 - Inverte o sentido da velocidade em x\n
            -2 - Inverte o sentido da velocidade em y\n
            -3 - Inverte o sentido da velocidade em z
        """
        self.vetor_velocidade[-dimensao_reflexão - 1] = - self.vetor_velocidade[-dimensao_reflexão - 1]

    def __str__(self) -> str:
        return "Raio {}, massa {}, posicao {} e velocidade {}".format(self.raio, self.massa, self.vetor_posicao, self.vetor_velocidade)
    
    def __repr__(self) -> str:
        return "Particula({}, {}, {}, {})".format(self.raio, self.massa, self.vetor_posicao.tolist(), self.vetor_velocidade.tolist())
    
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

# calculador do momento total de uma lista
def momento_total_lista(lista_de_particulas: list[Particula]) -> npt.NDArray[np.float64]:
    """Soma os vetores de momento de todas as partículas em uma lista"""
    lista_momentos = [particula.momento for particula in lista_de_particulas]
    momento_total = np.sum(lista_momentos, axis=0)
    return momento_total

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

# calcula o vetor normal a uma colisao
def vetor_normal_colisao(particula_1: Particula, particula_2: Particula) -> npt.NDArray[np.float64]:
    """
    Não usar fora da colisão!\n
    Na colisão, a magnetude do vetor distância é a soma dos raios, simplificando o cálculo do vetor normal.\n
    
    Returns
    -------
    vetor_normal: ndarray[float64]
        Nota: o vetor gerado aponta para a fora da superfície da particula_1!\n
        normal = vetor_distancia / (soma_dos_raios)
    """
    posicao_1 = particula_1.vetor_posicao
    posicao_2 = particula_2.vetor_posicao

    raio_1 = particula_1.raio
    raio_2 = particula_2.raio
    
    vetor_distancia = posicao_2 - posicao_1
    modulo_vetor_distancia = raio_1 + raio_2
    
    vetor_normal = vetor_distancia / modulo_vetor_distancia

    return vetor_normal

# calcula nova velocidade das particulas depois de colidirem
def velocidades_apos_colisao(massas: list[float], velocidades_normais: list[np.float64]) -> list[np.float64]:
    """
    No momento da colisão, podemos decompor as velocidades das partículas em suas componentes normal ao impacto (v1 e v2) e tangencial. Somente a velocidade normal é alterada.\n
    v1_nova = ( (m1 - m2) v1 -   2 m2    v2 ) / (m1 + m2)\n
    v2_nova = (   2 m1    v1 - (m1 - m2) v2 ) / (m1 + m2)

    Returns
    -------
    lista_velocidades_novas: list[float64]
        É uma lista formada pelas velocidades normais novas calculadas, na forma  [v1_nova, v2_nova]
    """
    m1, m2 = massas
    v1, v2 = velocidades_normais
    
    diff_massa = m1 - m2
    soma_massa = m1 + m2

    velocidade_1_nova = (diff_massa * v1 -    2*m2    * v2) / soma_massa
    velocidade_2_nova = (   2*m1    * v1 - diff_massa * v2) / soma_massa

    return [velocidade_1_nova, velocidade_2_nova]

# colide e atualiza as particulas para a colisao
def colisao(particula_1: Particula, particula_2: Particula) -> None: # O(1)
    """
    Atualiza as velocidades das partículas durante uma colisão. Usar somente se a colisão já for confirmada.\n
    Para mudar a velocidade normal sem alterar a velocidade tangencial, o vetor da velocidade tangencial é calculado e somado com a velocidade normal nova:\n\n
    velocidade_normal     = <velocidade_antes, vetor_normal>\n
    velocidade_tangencial = velocidade_antes - velocidade_normal\n

    velocidade_antes  = velocidade_tangencial + velocidade_normal\n
    velocidade_depois = velocidade_tangencial + velocidade_normal_nova
    """
    vetor_normal = vetor_normal_colisao(particula_1, particula_2)

    massa_1 = particula_1.massa
    massa_2 = particula_2.massa 

    v1_normal = np.vdot(particula_1.vetor_velocidade, vetor_normal)
    v2_normal = np.vdot(particula_2.vetor_velocidade, vetor_normal)

    vetor_v1_tangente = particula_1.vetor_velocidade - v1_normal * vetor_normal
    vetor_v2_tangente = particula_2.vetor_velocidade - v2_normal * vetor_normal

    v1_normal_novo, v2_normal_novo = velocidades_apos_colisao(massas = [massa_1, massa_2], velocidades_normais = [v1_normal, v2_normal])
    
    velocidade_particula_1 = vetor_v1_tangente + v1_normal_novo * vetor_normal
    velocidade_particula_2 = vetor_v2_tangente + v2_normal_novo * vetor_normal
    
    particula_1 = Particula(particula_1.raio, particula_1.massa, particula_1.vetor_posicao, velocidade_particula_1)
    particula_2 = Particula(particula_2.raio, particula_2.massa, particula_2.vetor_posicao, velocidade_particula_2)

# calculo da previsao de colisao entre duas particulas
def tempo_colisao_particulas(particula_1: Particula, particula_2: Particula) -> np.float64: # O(1)
    """
    Calcula o tempo para duas partículas se colidirem (a velocidades constantes) ou um valor bem grande, para não interferir na lista.\n
    Nota: Se as partículas estiverem dentro uma da outra, retorna resultado negativo. Tomar cuidado com isso nas primeiras colisões\n
    Podemos imaginar que a distância relativa é um vetor que cresce no sentido da velocidade relativa. Nesse caso, uma colisão seria o momento delta_t que faz a distância relativa ter magnetude igual à soma dos raios das partículas

    Returns
    -------
    tempo_para_colisao: float64
        O tempo para que ocorra a colisão entre as duas partículas.
    
    numero_muito_grande: float64
        Caso não ocorra colisão, retorna um valor muito grande (2^15 - 1) para ser ignorado na lista de prioriade (variável global).
    """
    raio_1 = particula_1.raio
    raio_2 = particula_2.raio
    distancia_colisao = raio_1 + raio_2

    posicao_1 = particula_1.vetor_posicao
    posicao_2 = particula_2.vetor_posicao
    delta_r   = posicao_1 - posicao_2
    
    velocidade_1 = particula_1.vetor_velocidade
    velocidade_2 = particula_2.vetor_velocidade
    delta_v      = velocidade_1 - velocidade_2

    produto_v_r  = np.vdot(delta_r, delta_v)

    if produto_v_r >= 0: # Se a distancia relativa estiver aumentando, elas estão se afastando
        return numero_muito_grande

    delta_v_quadrado = np.vdot(delta_v, delta_v)
    delta_r_quadrado = np.vdot(delta_r, delta_r)

    discriminante = produto_v_r * produto_v_r - delta_v_quadrado * (delta_r_quadrado - distancia_colisao * distancia_colisao)

    if discriminante < 0: # se o discriminante de baskara for menor que zero, nao ha colisão
        return numero_muito_grande
    
    else: # se tiver tudo certo pra colisao, calcula o tempo de colisao normalmente
        tempo_colisao = - (produto_v_r + np.sqrt(discriminante)) / delta_v_quadrado
        return tempo_colisao

# calcuo da previsao de colisoes entre uma particula e todas as paredes
def tempo_colisao_paredes(particula: Particula) -> npt.NDArray[np.float64]: # O(1)
    """
    Calcula o tempo para que a particula colida com uma parede, já levando em consideração o sentido da velocidade. Se não for possível uma colisão com a parede na dimensão checada, retorna um valor muito grande, para ser ignorado na lista de colisões.\n
    Nota 1: Depende das variáveis globais numero_dimensoes para calcular as colisões em cada dimensão. Não entrar particulas com vetores de dimensões diferentes.\n
    Nota 2: Checa o tempo de colisão com a parede próxima da origem, se a velocidade for negativa, e com a parede mais distante da origem, se a velocidade for positiva. 
    
    Returns
    -------
    tempos_colisoes: NDArray[float64]
        Lista formada pelos tempos para a partícula colidir com a parede da dimensão dada pelo seu índice (inteiros negativos).\n
        tempos_colisoes[-1] = tempo para colisao em x\n
        tempos_colisoes[-2] = tempo para colisao em y\n
        tempos_colisoes[-3] = tempo para colisao em z

    tempo_para_colisao: float64
        O tempo para que ocorra a colisão entre as duas partículas.
    
    numero_muito_grande: float64
        Caso não ocorra colisão, retorna um valor muito grande (2^15 - 1) para ser ignorado na lista de prioriade.
    """
    tempos_colisoes = []

    posicao    = particula.vetor_posicao
    velocidade = particula.vetor_velocidade
    raio       = particula.raio
    
    for i in range(numero_dimensoes):
        if velocidade[i] > 0:
            distancia = (tamanho_parede - posicao[i] - raio)
            tempo_calculado = distancia / velocidade[i]
        
        elif velocidade[i] < 0:
            distancia = (raio - posicao[i])
            tempo_calculado = distancia / velocidade[i]

        else: # Se a velocidade for zero, a partícula não tem como atingir nenhuma das duas paredes.
            tempo_calculado = numero_muito_grande

        tempos_colisoes.append(tempo_calculado)
    
    return tempos_colisoes[::-1]

# gera todos os pares de colisao entre duas particulas
def gerar_pares_colisoes(quantidade_particulas: int) -> list[list[int]]: # O(n^2)
    """
    Gera as combinações das possíveis condições de colisão de duas partículas, representadas como números de 0 a quantidade_particulas - 1 na lista.\n

    Example
    -------
    >>> gerar_pares_colisoes(4)
    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

    >>> # Uma forma melhor de visualizar o resultado seria como tabela:
        # [
        # [0, 1], [0, 2], [0, 3], [0, 4],
        #         [1, 2], [1, 3], [1, 4],
        #                 [2, 3], [2, 4],
        #                         [3, 4]
        # ]
    """
    pares_colisoes = []

    for i in range(0, quantidade_particulas):
        for j in range(i+1, quantidade_particulas):
            pares_colisoes.append([i,j])
    
    return pares_colisoes

# gera todos os pares de colisao entre uma particula e as paredes
def gerar_colisoes_paredes(quantidade_particulas: int, numero_dimensoes: int) -> list[list[int]]: # O(n)
    """
    Gera todos os pares de colisão das partículas com as paredes, retornando uma lista de combinações do índice da partícula (na primeira posção) com o índice da dimensão (na segunda posição).\n

    Os índices das dimensões das paredes são:\n
    -1 - Paredes em x\n
    -2 - Paredes em y\n
    -3 - Paredes em z
    
    Example
    -------
    >>> gerar_colisoes_paredes(quantidade_particulas=2, numero_dimensoes=3)
    [[0, -1], [1, -1], [0, -2], [1, -2], [0, -3], [1, -3]]
    """
    indices_paredes    = np.arange(-1, -1 - numero_dimensoes, -1)
    indices_particulas = np.arange(quantidade_particulas)
    
    colisoes_paredes = []
    for indice_parede in indices_paredes:
        for indice_particula in indices_particulas:
            colisoes_paredes.append([indice_particula, indice_parede])

    return colisoes_paredes

# aencontra as posicoes de uma particula na lista de pares de colisap
def numero_primeira_posicao(numero_desejado: int, quantidade_particulas: int) -> int:
    """
    Com os pares de colisões sendo uma lista ordenada da forma que seu output ordena naturalmente, o numero_desejado passa a se repetir na primeira posição nos pares a partir de uma posição. Essa função calcula a posição em que essa repetição começa.\n
    Nota: Retorna valores iguais ou maiores que o tamanho da lista de pares se o numero não aparecer na primeira posição.
    
    Example
    -------
    >>> gerar_pares_colisoes(4)
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    >>> len(gerar_pares_colisoes(4))
    6
    >>> numero_primeira_posicao(1, 4)
    3
    >>> numero_primeira_posicao(2, 4)
    5
    >>> numero_primeira_posicao(3, 4) # nao existe na lista, retorna um numero maior ou igual ao tamanho da lista
    6
    
    Derivação
    ---------
    Podemos considerar que a posição que o numero N aparece na primeira posição é igual à posição que N-1 aparece na primeira posição dos pares + a quantidade de vezes que N-1 permanece na primeira posição:\n
        posicao(N) = posicao(N-1) + repeticao(N-1)

    E como o valor N é contado uma vez para cada número menor que N, restam somente (valor_maximo - N) combinações com o número N ao chegar na primeira posição:\n
        repeticao(N) = (valor_maximo - N)

    Com isso, podemos usar a recursão dessa função para descobrir sua fórmula, lembrando que o valor 0 aparece na primeira posição da lista de combinações (posição 0 em python):\n
        posicao(N) = posicao(N-1) + repeticao(N-1)
        posicao(N) = posicao(N-2) + repeticao(N-2) + repeticao(N-1)
        ...
        posicao(N) = posicao(0) + repeticao(0) + repeticao(1) + ... + repeticao(N-2) + repeticao(N-1)
        posicao(0) = 0
        posicao(N) = repeticao(0) + repeticao(1) + ... + repeticao(N-2) + repeticao(N-1)

    E aplicando a formula para o calculo da repeticao temos o seguinte padrão:\n
        posicao(N) = valor_maximo + (valor_maximo - 1) + (valor_maximo - 2) + ... + (valor_maximo - (N-2)) + (valor_maximo - (N-1))
        posicao(N) = N * valor_maximo - (1 + 2 + ... + (N-2) + (N-1))
        posicao(N) = N * valor_maximo - N * (N-1)/2

    E lembrando que começa-se a contar do 0, valor_maximo vale quantidade_particulas - 1:\n
        posicao(N) = N * (quantidade_particulas - 1) - N * (N-1)/2
        posicao(N) = N * (quantidade_particulas - 1 - (N-1)/2)
        posicao(N) = N * (quantidade_particulas - (N+1)/2)
    """
    posicao_repeticao = int(numero_desejado * (quantidade_particulas - (numero_desejado + 1) / 2))
    return posicao_repeticao

# encontra todos os indices onde o a particula desejada aparece na lista de pares de colisao entre duas particulas
def indices_colisao_particulas(numero_desejado: int, quantidade_particulas: int) -> list[int]: # tecnicamente O(n), mas basicamente O(1) para nosso uso
    """
    Calcula todas as posições em que o numero_desejado aparece na lista de colisão entre as partículas, seja na primeira posição da dupla ou na segunda.
    
    Example
    -------
    >>> gerar_pares_colisoes(4)
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    >>> posicoes_lista_de_pares(1, 4)
    [0, 3, 4]
    >>> posicoes_lista_de_pares(3, 4)
    [2, 4, 5]
    """
    inicio_primeiro_lugar: int     = numero_primeira_posicao(numero_desejado, quantidade_particulas)
    quantidade_primeiro_lugar: int = quantidade_particulas - numero_desejado - 1 # O numero se repete (valor_maximo - numero_desejado) vezes
    
    # quando no primeiro lugar da dupla, o numero aparece em todas as posicoes de inicio_primeiro_lugar ate inicio_primeiro_lugar + quantidade_primeiro_lugar
    indices_primeiro_lugar: list[int] = [i for i in range(inicio_primeiro_lugar, inicio_primeiro_lugar + quantidade_primeiro_lugar)]

    # quando aparece no segundo lugar, ele o primeiro numero é menor que ele, aparecendo sempre numero_desejado - i - 1 posicoes depois de i assumir o primeiro lugar da dupla
    indices_segundo_lugar:  list[int] = [numero_primeira_posicao(i, quantidade_particulas) + numero_desejado - i - 1 for i in range(numero_desejado)]

    return indices_segundo_lugar + indices_primeiro_lugar

# encontra todos os indices onde o a particula desejada aparece na lista de pares de colisao entre uma particula e as paredes
def indices_colisao_paredes(numero_desejado: int, quantidade_particulas: int, numero_dimensoes: int) -> list[int]: #O(numero_dimensoes)
    """
    Calcula todas as posições em que o numero_desejado aparece na lista de colisão das partículas com as paredes.\n
    Os indices das partículas nessa lista se repetem em intervalos de quantidade_particulas, então basta contar o primeiro índice + n * quantidade_particulas .
    
    Example
    -------
    >>> gerar_colisoes_paredes(quantidade_particulas=3, numero_dimensoes=3)
    [[0, -1], [1, -1], [2, -1], [0, -2], [1, -2], [2, -2], [0, -3], [1, -3], [2, -3]]
    >>> indices_colisao_paredes(numero_desejado=0, quantidade_particulas=3, numero_dimensoes=3)
    [0, 3, 6]
    >>> indices_colisao_paredes(numero_desejado=2, quantidade_particulas=3, numero_dimensoes=3)
    [2, 5, 8]
    """
    primeiro_indice = numero_desejado
    indices_encontrados = [primeiro_indice + n * quantidade_particulas for n in range(numero_dimensoes)]
    return indices_encontrados

# checagens para tolerancias
def checar_tempo_colisao(tempo_calculado: np.float64) -> bool:
    """
    Checa se o tempo calculado da colisão é menor que o numero_muito_grande e maior que zero. Essa checagem é para verificar se a colisão é impossível ou se já aconteceu.
    """
    resultado_checagem = tempo_calculado > 0 and tempo_calculado < numero_muito_grande
    return resultado_checagem
def checar_colisao_particulas(particula_1: Particula, particula_2: Particula, tolerancia: float) -> bool:
    """
    Checa se as duas partículas estão a uma distância menor que raio 1 + raio 2 + tolerância.\n
    A tolerância é para contrabalancear erros numéricos dos tempos calculados.
    """
    posicao_1 = particula_1.vetor_posicao
    posicao_2 = particula_2.vetor_posicao
    raio_1    = particula_1.raio
    raio_2    = particula_2.raio

    distancia_colisao = raio_1 + raio_2 + tolerancia
    distancia_quadrada_colisao = distancia_colisao * distancia_colisao

    vetor_distancia_centros = posicao_2 - posicao_1
    distancia_quadrada = np.vdot(vetor_distancia_centros, vetor_distancia_centros)

    resultado_checagem = distancia_quadrada <= distancia_quadrada_colisao
    return resultado_checagem
def checar_colisao_parede(particula: Particula, indice_parede: int, tolerancia: float) -> bool:
    """
    Checa se a partícula está a uma distância menor que raio + tolerância da parede.\n
    A tolerância é para contrabalancear erros numéricos dos tempos calculados.\n\n

    Lembrar os índices das paredes:\n
    -1 - Parede em x\n
    -2 - Parede em y\n
    -3 - Parede em z
    """
    vetor_posicao    = particula.vetor_posicao
    vetor_velocidade = particula.vetor_velocidade

    posicao_na_dimensao    = vetor_posicao[- indice_parede - 1]
    velocidade_na_dimensao = vetor_velocidade[- indice_parede - 1]

    distancia_colisao = particula.raio + tolerancia

    if(velocidade_na_dimensao == 0): # se tiver velocidade zero, nao tem como estar indo em direcao a parede
        resultado_checagem = False
        return resultado_checagem
    
    elif(velocidade_na_dimensao < 0): # se estiver indo em direcao da parede negativa (a mais proxima da origem), considerar a parede da origem
        posicao_parede  = 0

    else: # senao considera outra parede naquela dimensao
        posicao_parede  = tamanho_parede
    
    resultado_checagem  = abs(posicao_na_dimensao - posicao_parede) < distancia_colisao
    return resultado_checagem

# criacao da fila de prioridade
def criar_fila_prioridade(lista_particulas: list[Particula]) -> list[np.float64 | list[int]]: # O(quantidade_particulas^2)
    """
    Cria a fila de eventos para as colisoes, retornando uma lista de elementos (tempo de colisão e índice das partículas ou paredes na colisão) organizados pelo menor tempo de colisão. Os eventos são organizados no seguinte formato:\n
        evento = [tempo_colisao, indice_particula, indice_particula_ou_parede]
    Nota: As paredes usam índices negativos, calculados pela formula -(numero_da_dimensão + 1), para não se confundirem com os índices das partículas:\n
        -1: Parede em x\n
        -2: Parede em y\n
        -3: Parede em z
    """
    fila_eventos = []
    pares_colisoes = gerar_pares_colisoes(len(lista_particulas))

    # Primeiro calcula os eventos para as colisões das particulas
    for indices_par in pares_colisoes:
        particula_1 = lista_particulas[indices_par[0]]
        particula_2 = lista_particulas[indices_par[1]]

        # checa se as particulas nao tao uma dentro da outra, dentro de uma tolerancia
        estao_em_colisao = checar_colisao_particulas(particula_1, particula_2, 0.01 * particula_1.raio)

        if estao_em_colisao:
            tempo_calculado_particulas = numero_muito_grande # se tiverem, ignorar a colisao
        else:
            tempo_calculado_particulas = tempo_colisao_particulas(particula_1, particula_2)

        # so adiciona o evento se representar uma colisao possivel (tempo positivo e menor que numero_muito_grande) para simplificar o uso da lista no futuro
        colisao_eh_possivel = checar_tempo_colisao(tempo_calculado_particulas)
        if colisao_eh_possivel:
            evento = [tempo_calculado_particulas, indices_par]
            fila_eventos.append(evento)
    
    # depois para cada parede, lembrando que a função resolve para as tres dimensoes
    for indice_particula in range(len(lista_particulas)):
        particula = lista_particulas[indice_particula]
        tempos_calculados_paredes = tempo_colisao_paredes(particula)

        for dimensao in range(numero_dimensoes):
            indice_parede = - dimensao - 1
            
            # nao precisa checar o tempo, a probabilidade de nao ser possivel colidir com uma parede é muito pequena
            evento = [tempos_calculados_paredes[indice_parede], [indice_particula, indice_parede]]
            fila_eventos.append(evento)

    fila_eventos.sort(key=lambda item: item[0])

    return fila_eventos

# depois de uma colisao, recalcula os tempos de colisao do par
def recalcular_colisoes(lista_particulas: list[Particula], par_colisao: list[int]) -> list[np.float64 | list[int]]: # O(quantidade_particulas^2)
    """
    Cria uma fila de eventos para as colisoes de uma particula especifica, retornando uma lista de elementos (tempo de colisão e índice das partículas ou paredes na colisão) organizados pelo menor tempo de colisão. Os eventos são organizados no seguinte formato:\n
        evento = [tempo_colisao, indice_particula, indice_particula_ou_parede]
        
    Nota: As paredes usam índices negativos, calculados pela formula -(numero_da_dimensão + 1), para não se confundirem com os índices das partículas:\n
        -1: Parede em x\n
        -2: Parede em y\n
        -3: Parede em z
    """
    lista_eventos = []
    eh_colisao_parede = par_colisao[1] < 0

    if eh_colisao_parede:
        indices_pares_desejados = indices_colisao_particulas(par_colisao[0], quantidade_particulas)
        indice_parede = par_colisao[1]
    else:
        indices_pares_desejados = indices_colisao_particulas(par_colisao[0], quantidade_particulas) + indices_colisao_particulas(par_colisao[1], quantidade_particulas)

    # separa so os pares de colisao em que a particula aparece
    pares_colisoes_completo = gerar_pares_colisoes(quantidade_particulas)
    pares_colisoes = []

    for i in indices_pares_desejados:
        pares_colisoes.append(pares_colisoes_completo[i])

    # Primeiro calcula os eventos para as colisões das particulas
    for indices_particulas_colisao in pares_colisoes:
        particula_1 = lista_particulas[indices_particulas_colisao[0]]
        particula_2 = lista_particulas[indices_particulas_colisao[1]]

        tempo_calculado_particulas = tempo_colisao_particulas(particula_1, particula_2)

        if tempo_calculado_particulas < numero_muito_grande:
            evento = [tempo_calculado_particulas, indices_particulas_colisao]
            lista_eventos.append(evento)
    
    # depois para cada parede
    particula = lista_particulas[par_colisao[0]]
    tempos_calculados_paredes = tempo_colisao_paredes(particula)
    
    # se for uma colisao com uma parede, o tempo das colisoes com as paredes das outras dimensoes nao muda
    if eh_colisao_parede:
        evento = [tempos_calculados_paredes[indice_parede], [par_colisao[0], indice_parede]]
        lista_eventos.append(evento)

    # senao deve recalcular pras tres paredes
    else:
        for dimensao in range(numero_dimensoes):
            indice_parede = - dimensao - 1
            
            # nao precisa checar o tempo, a probabilidade de nao ser possivel colidir com uma parede é muito pequena
            evento = [tempos_calculados_paredes[indice_parede], [par_colisao[0], indice_parede]]
            lista_eventos.append(evento)

    return lista_eventos

# adicionar e remover eventos da fila
def remover_evento_fila(fila: list, tempo_avancado: float) -> list[np.float64 | list[int]]:
    """
    Remove o primeiro elemento da fila de prioridades, atualizando também o tempo de todos os próximos eventos. O tempo em cada elemento da lista marca quanto tempo falta para o evento acontecer
    
    Returns
    -------
    fila_atualizada: list[np.float64 | list[int]]
    A fila com o evento mais recente removido e todos os tempos atualizados.
    """
    atualiza_tempo_evento = lambda evento: [evento[0] - tempo_avancado, evento[1]]

    # remove o evento mais recente e atualiza os tempos de todos os outros eventos
    fila_atualizada = fila[1:]
    fila_atualizada = list(map(atualiza_tempo_evento, fila_atualizada))

    return fila_atualizada
def adicionar_na_fila(fila: list, tempo_calculado: np.float64, par_indices: list[int]) -> list[np.float64 | list[int]]:
    """
    Essa função edita a lista da fila de prioridade para adicionar a colisão calculada na fila já na ordem certa. Retorna 0 ao concluir.
    
    Example
    -------
    >>> fila_prioridade = [[1, [0,1]], [2, [1, 3]]]
    >>> fila_prioridade
    [[1, [0,1]], [2, [1, 3]]]

    >>> adicionar_na_fila(fila_prioridade, 0, [0, 2])
    >>> adicionar_na_fila(fila_prioridade, 1.5, [0, 3])
    >>> fila_prioridade
    [[0, [0, 2]], [1, [0, 1]], [1.5, [0, 3]]], [2, [1, 3]]]
    """
    fila_para_atualizar = fila
    lista_tempos = [evento[0] for evento in fila_para_atualizar]

    maior_posicao = len(lista_tempos) - 1
    menor_posicao = 0

    menor_tempo = lista_tempos[menor_posicao]
    maior_tempo = lista_tempos[maior_posicao]

    if tempo_calculado <= menor_tempo: # O(1)
        fila_para_atualizar.insert(0, [tempo_calculado, par_indices])
        return 0
    
    if tempo_calculado >= maior_tempo: # O(1)
        fila_para_atualizar.append([tempo_calculado, par_indices])
        return 0
    
    # Procura por bissecao o lugar na lista para botar
    while maior_tempo - menor_tempo > 1: # O(log n)
        posicao_meio = int(np.floor((menor_posicao + maior_posicao) / 2)) # checar se e um int
        tempo_meio = lista_tempos[posicao_meio]

        if tempo_calculado > tempo_meio: # checar se é maior ou maior igual
            menor_posicao = posicao_meio
            menor_tempo = tempo_meio
        
        else:
            maior_posicao = posicao_meio
            maior_tempo = posicao_meio
    
    posicao_insert = maior_posicao
    fila_para_atualizar.insert(posicao_insert, [tempo_calculado, par_indices])
    return fila_para_atualizar


# ------ LOOP DE SIMULACAO ------
particulas_simulacao = criar_particulas_iniciais(quantidade_particulas, raio, massa)
lista_prioridade = criar_fila_prioridade(particulas_simulacao)

# declara essas variaveis
energias_cineticas = [particula.energia_cinetica for particula in particulas_simulacao]
momento_colisoes_parede = 0
energia_cinetica_media = 0

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
fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=100)  # particulas
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)  # Histograma

# Get figure dimensions in display units
fig1_comprimento_polegadas = fig1.get_figwidth()

# Convert figure size to points (1 inch = 72 points)
fig1_comprimento_pontos = fig1_comprimento_polegadas * 72
fator_de_escala = fig1_comprimento_pontos / tamanho_parede
areas = np.pi * (raio * fator_de_escala)**2

# transforma de figura do matplotlib pra superficie do pygame
def transformar_em_superficie(figura):
    """Converte de figura do matplotlib pra uma superficie do pygame"""
    canvas = FigureCanvasAgg(figura)
    buffer = io.BytesIO()
    canvas.print_raw(buffer)
    buffer.seek(0)
    size = canvas.get_width_height()
    return pygame.image.fromstring(buffer.getvalue(), size, "RGBA")

def atualiza_plot(vetores_posicoes: list[npt.NDArray[np.float64]]):
    """Atualiza o plot das particulas"""
    ax1.clear()
    ax1.set_xlim(0, tamanho_parede)
    ax1.set_ylim(0, tamanho_parede)
    ax1.set_aspect('equal')
    ax1.set_title("Visualização das partículas")
    
    ax1.scatter(
        vetores_posicoes[:, 0],
        vetores_posicoes[:, 1],
        s=(raio * fator_de_escala)**2,
        c='blue',
        alpha=0.5,
        edgecolors='red',
        linewidths=0.5
    )

def atualiza_histograma(energias_cineticas: list[np.float64]):
    """Atualiza o histograma dos valores de energia cinética"""
    ax2.clear()
    ax2.set_title("Distribuição de energia cinética")
    ax2.set_xlabel("Energia cinética")
    ax2.set_ylabel("Contagem")
    
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

    while (tempo_global < tempo_sample):
        # coleta o tempo para o proximo evento
        evento_atual = lista_prioridade[0]
        tempo_para_colisao: np.float64 = evento_atual[0]

        # ignora tempos negativos e muito pequenos
        while tempo_para_colisao <= tempo_minimo_para_evento:
            lista_prioridade = lista_prioridade[1:]
            evento_atual     = lista_prioridade[0]
            tempo_para_colisao: np.float64 = evento_atual[0]
        
        # atualiza a posicao de todas as particulas
        for particula in particulas_simulacao:
            particula.atualizar_posicao(tempo_para_colisao)

        # e coleta o par para colisao
        par_colisao: list[int] = evento_atual[1]

        # se for uma colisao com a parede, usa a funcao de reflexao
        eh_colisao_parede = par_colisao[1] < 0
        
        if eh_colisao_parede:
            indice_particula, indice_parede = par_colisao

            particula = particulas_simulacao[indice_particula]

            # a parede absorve 2x o momento original da particula naquela dimensao e reflete
            momento_colisoes_parede += 2 * particula.momento[- 1 - indice_parede]
            particula.reflexao(indice_parede)
                
        # senao pega a particula 2 e colide as duas
        else:
            indice_particula_1, indice_particula_2 = par_colisao

            particula_1 = particulas_simulacao[indice_particula_1]
            particula_2 = particulas_simulacao[indice_particula_2]
            colisao(particula_1, particula_2)

        # recalcula as colisoes e remove o evento concluido da lista
        colisoes_recalculadas = recalcular_colisoes(particulas_simulacao, par_colisao)
        lista_prioridade = remover_evento_fila(lista_prioridade, tempo_para_colisao)

        # adiciona as colisoes recalculadas na lista
        lista_prioridade = lista_prioridade + colisoes_recalculadas
        lista_prioridade.sort(key = lambda evento: evento[0])
        
        # adiciona o tempo avancado no tempo global
        tempo_global += tempo_para_colisao

        # atualiza os plots
        posicoes = [particula.vetor_posicao for particula in particulas_simulacao]
        energias_cineticas = [particula.energia_cinetica for particula in particulas_simulacao]
        atualiza_plot(posicoes)
        atualiza_histograma(energias_cineticas)

        # transforma em superficies
        surperficie_particulas = transformar_em_superficie(fig1)
        superficie_histograma  = transformar_em_superficie(fig2)

        # Limpa a tela
        tela.fill(branco)

        # coloca as duas superficies na tela
        tela.blit(surperficie_particulas, (50, 50))
        tela.blit(superficie_histograma, (50, HEIGHT//2 + 50))

        pygame.display.flip()
        clock.tick(FPS)
        
    energia_cinetica_media = np.mean([particula.energia_cinetica for particula in particulas_simulacao])
    pressao_parede = momento_colisoes_parede / tempo_sample

    tempo_sample += tempo_global

pygame.quit()
plt.close('all')