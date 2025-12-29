import numpy as np
import random
from numpy.typing import NDArray
from typing import Callable
from scipy.stats import qmc

class Algoritmo_genetico:
    def __init__(self, funcao_fitness: Callable[[dict], float], tamanho_populacao: int, escolhas_discretas: list[list] | None = None, range_var_continuas: list[list] | None = None, seed: int | None = None):
        """
        Para inicializar, é preciso dizer o tamanho da população, as opções para cada variável discreta e o range das variáveis contínuas.
        Nota: Lembrar de ajustar self.funcao_fitness para ser a função objetivo desejada.
        
        Parameters
        ----------
        funcao_fitness: Callable[[dict], float]
            É a função objetivo para maximizar, que deve receber o indivíduo já desnormalizado, usar as chaves 'genes_continuos' e/ou 'genes_discretos' e retornar o valor de fitness do indivíduo.
        
        tamanho_populacao: int
            A quantidade de soluções testadas por geração. Deve ser maior ou igual a 2.
        
        escolhas_discretas: list[list]
            A lista de possibilidades para cada gene discreto. Por exemplo, se o primeiro gene pode ser 1 ou 2, enquanto o segundo pode ser 'a', 'b', ou 'c', o input seria [[1,2], ['a','b','c']]
        
        range_var_continuas: list[list]
            A lista de possibilidades para cada gene discreto. Por exemplo, se o primeiro gene pode variar de 1.0 a 2.0, enquanto o segundo pode variar de 4.2 a 9.0, o input seria [[1.0, 2.0], [4.2, 9.0]]

        seed: int | None
            A seed usada nos RNGs para garantir reprodutibilidade.

        Example
        -------
        >>> # Preciso criar um AG com 20 indivíduos, com uma var. contínua que varia de 2.0 a 5.0
        >>> AG = Algoritmo_genetico(tamanho_populacao=20, range_var_continuas=[[2.0, 5.0]])
        >>> AG.treinar(...)
        Treino realizado com sucesso em 12 gerações!
        """
        np.random.seed(seed)
        random.seed(seed)

        self.escolhas = escolhas_discretas or []
        self.limites  = np.array(range_var_continuas or [])
        
        self.qtd_var_discretas = len(self.escolhas)
        self.qtd_var_continuas = len(self.limites)
        self.tem_var_continuas = self.qtd_var_continuas > 0
        self.tem_var_discretas = self.qtd_var_discretas > 0

        self.funcao_fitness = lambda individuo: funcao_fitness(self.desnormalizar(individuo))
        
        if not self.tem_var_continuas and not self.tem_var_discretas:
            raise ValueError("Nenhuma variável foi definida.")
        
        if tamanho_populacao < 2:
            raise ValueError("A população deve ter pelo menos 2 indivíduos.")
        
        if self.qtd_var_discretas > 0:
            if np.any([len(opcoes) == 1 for opcoes in self.escolhas]):
                raise ValueError("Uma das variáveis discretas possui somente uma opção.")
        
        if self.qtd_var_continuas > 0:
            if np.any([limite[1] == limite[0] for limite in self.limites]):
                raise ValueError("Uma das variáveis contínuas possui limite inferior igual ao limite superior.")

        self.qtd_total_var     = self.qtd_var_discretas + self.qtd_var_continuas
        self.tamanho_populacao = tamanho_populacao
        
        self.cache_fitness     = {}
        self.genes_populacao   = self.inicializar_genes(tamanho_populacao)
        self.populacao         = self.avaliar_genes(self.genes_populacao)
        self.melhores          = [max(self.populacao, key=lambda x: x['fitness'])]

    def genes_iniciais_continuos(self, quantidade_individuos: int) -> NDArray:
        """
        Usa a distribuição de Latin Hypercube para garantir boa variabilidade genética inicial para as variáveis contínuas.
        Retorna uma NDArray de listas vazias se não forem definidas variáveis contínuas.
        """
        if not self.tem_var_continuas:
            return np.empty((quantidade_individuos, 0))

        sampler = qmc.LatinHypercube(d=self.qtd_var_continuas)
        genes = sampler.random(n=quantidade_individuos)
        return genes

    def genes_iniciais_discretos(self, quantidade_individuos: int) -> NDArray:
        """
        Usa escolhas aleatórias para garantir boa variabilidade genética inicial para as variáveis discretas.
        Retorna uma NDArray de listas vazias se não forem definidas variáveis discretas.
        """
        if not self.tem_var_discretas:
            return np.empty((quantidade_individuos, 0))

        genes = []

        for _ in range(quantidade_individuos):
            individuo = [random.choice(opcoes) for opcoes in self.escolhas]
            genes.append(individuo)

        return np.array(genes)

    def inicializar_genes(self, tamanho_populacao: int) -> list[dict[str, NDArray]]:
        """
        inicializa os genes contínuos e discretos da população, criando indivíduos sem nota.
        Já considera se não forem definidas variáveis contínuas ou discretas.
        """
        genes_discretos = self.genes_iniciais_discretos(tamanho_populacao)
        genes_continuos = self.genes_iniciais_continuos(tamanho_populacao)
        genes_populacao = []

        for i in range(tamanho_populacao):
            genes_populacao.append({'genes_discretos': genes_discretos[i], 'genes_continuos': genes_continuos[i]})

        return genes_populacao

    def desnormalizar(self, individuo: dict[str, NDArray]) -> dict[str, NDArray]:
        """
        Como todos os genes são normalizados, é preciso passar por esse desnormalizador antes de aplicar a função objetivo.
        """
        genes_discretos = individuo['genes_discretos']
        genes_continuos = self.limites[:, 0] + individuo['genes_continuos'] * (self.limites[:, 1] - self.limites[:, 0])
        
        individuo_desnormalizado = {'genes_discretos': genes_discretos, 'genes_continuos': genes_continuos}

        return individuo_desnormalizado
        
    def chave_individuo(self, genes: dict[str, NDArray], resolucao_relativa: float = 0.01) -> tuple:
        """
        Gera uma string única que representa o indivíduo (para uso no cache), usando quantização proporcional ao range das variáveis contínuas (proxy).

        Exemplo:
            range_gene_continuo = [0, 1]
            gene_continuo = 0.48939209...
            resolucao_continua = 0.2 (20% do range)
            gene salvo no cache: 0.4 (salvo como int(0.48939209... / 0.2) = 2)
        """
        if resolucao_relativa <= 0:
            raise ValueError("resolucao_relativa deve ser > 0.")
        
        if self.tem_var_discretas:
            genes_discretos = tuple(genes['genes_discretos'])
        else:
            genes_discretos = ()

        if self.tem_var_continuas:
            genes_quantizados = tuple(
                int(np.floor(x / resolucao_relativa)) for x in genes['genes_continuos']
            )

            genes_continuos = tuple(genes_quantizados)
        else:
            genes_continuos = ()

        return (genes_discretos, genes_continuos)

    def avaliar_genes(self, genes_avaliar: list[dict[str, NDArray]], resolucao_relativa: float = 0.01) -> list[dict]:
        """Adiciona a entrada 'fitness' em cada gene, com caching para evitar reavaliações repetidas."""
        genes_avaliar = [gene.copy() for gene in genes_avaliar]

        for genes in genes_avaliar:
            chave = self.chave_individuo(genes, resolucao_relativa)

            # Se já foi avaliado, reaproveita
            if chave in self.cache_fitness:
                fitness = self.cache_fitness[chave]
            else:
                fitness = self.funcao_fitness(genes)
                self.cache_fitness[chave] = fitness  # salva no cache

            genes['fitness'] = fitness

        return genes_avaliar

    def crossover_discreto(self, genes_discretos_pai: NDArray, genes_discretos_mae: NDArray) -> NDArray:
        """
        Usa escolhas aleatórias entre os genes dos pais para gerar os genes do cruzamento.
        """
        genes_filho = []

        for i in range(len(self.escolhas)):
            gene_escolhido = random.choice([genes_discretos_pai[i], genes_discretos_mae[i]])
            genes_filho.append(gene_escolhido)

        return np.array(genes_filho)

    def crossover_continuo(self, genes_continuos_pai: NDArray, genes_continuos_mae: NDArray, eta_c: float) -> NDArray:
        """
        Usa SBX (Simulated Binary Crossover) para calcular os genes contínuos do cruzamento dos pais.
        O SBX também garante a possibilidade de extrapolação dos para valores além dos genes dos pais.
        """
        u = np.random.rand(self.qtd_var_continuas)

        # Calcula beta
        beta_q = np.where(u <= 0.5,
            (2*u) ** (1 / (eta_c + 1)),
            (1 / (2 * (1-u))) ** (1 / (eta_c + 1))
        )

        # Calcula os filhos usando SBX
        filho_1 = 0.5 * ((1+beta_q) * genes_continuos_pai + (1-beta_q) * genes_continuos_mae)
        filho_2 = 0.5 * ((1-beta_q) * genes_continuos_pai + (1+beta_q) * genes_continuos_mae)

        # Clipa para não sair do domínio
        filho_1 = np.clip(filho_1, 0, 1)
        filho_2 = np.clip(filho_2, 0, 1)

        return filho_1 if np.random.rand() < 0.5 else filho_2

    def crossover(self, pai: dict, mae: dict, eta_c: float) -> dict[str, NDArray]:
        """
        Realiza cruzamento entre dois indivíduos e retorna os genes do cruzamento.
        Já considera se não houverem variáveis contínuas ou discretas.
        """
        if self.tem_var_discretas:
            genes_discretos_pai   = pai['genes_discretos'].copy()
            genes_discretos_mae   = mae['genes_discretos'].copy()
            genes_discretos_filho = self.crossover_discreto(genes_discretos_pai, genes_discretos_mae)
        else:
            genes_discretos_filho = np.empty(0)

        if self.tem_var_continuas:
            genes_continuos_pai   = pai['genes_continuos'].copy()
            genes_continuos_mae   = mae['genes_continuos'].copy()
            genes_continuos_filho = self.crossover_continuo(genes_continuos_pai, genes_continuos_mae, eta_c)
        else:
            genes_continuos_filho = np.empty(0)

        genes_filho = {
            'genes_discretos': genes_discretos_filho,
            'genes_continuos': genes_continuos_filho
            }

        return genes_filho

    def mutacao_discreta(self, genes_originais: NDArray) -> NDArray:
        """
        Altera um dos genes_originais para uma escolha aleatória diferente, usando a lista self.escolhas_discretas.
        Usar somente se tiverem variáveis discretas para escolha.
        """
        gene_mutar = np.random.randint(self.qtd_var_discretas)
        genes_mutados = genes_originais.copy()

        while genes_mutados[gene_mutar] == genes_originais[gene_mutar]:
            genes_mutados[gene_mutar] = random.choice(self.escolhas[gene_mutar])

        return genes_mutados

    def mutacao_continua(self, genes_originais: NDArray, proporcao_desvio: float) -> NDArray:
        """
        Altera um dos genes_originais para um valor aleatório normalmente distribuído em volta de seu valor original.
        Usar somente se tiverem variáveis contínuas para escolha.
        """
        gene_mutar = np.random.randint(self.qtd_var_continuas)
        desvio_padrao = proporcao_desvio
        genes_mutados = genes_originais.copy()

        genes_mutados[gene_mutar] = np.random.normal(genes_originais[gene_mutar], desvio_padrao)
        genes_mutados[gene_mutar] = np.clip(genes_mutados[gene_mutar], 0, 1)

        return genes_mutados

    def mutacao(self, individuo: dict, proporcao_desvio: float = 0.1) -> dict[str, NDArray]:
        """
        Realiza mutação de um dos genes_originais.
        já considera se não houverem variáveis contínuas ou discretas.
        """
        genes_discretos = individuo['genes_discretos']
        genes_continuos = individuo['genes_continuos']

        genes_discretos_mutados = genes_discretos.copy()
        genes_continuos_mutados = genes_continuos.copy()

        if self.tem_var_discretas and not self.tem_var_continuas:
            genes_discretos_mutados = self.mutacao_discreta(genes_discretos)

        elif not self.tem_var_discretas and self.tem_var_continuas:
            genes_continuos_mutados = self.mutacao_continua(genes_continuos, proporcao_desvio)

        else:
            qual_gene_mutar = np.random.randint(self.qtd_var_discretas + self.qtd_var_continuas)

            if qual_gene_mutar < self.qtd_var_discretas:
                genes_discretos_mutados = self.mutacao_discreta(genes_discretos)

            else:
                genes_continuos_mutados = self.mutacao_continua(genes_continuos, proporcao_desvio)

        genes_mutados = {
            'genes_discretos': genes_discretos_mutados,
            'genes_continuos': genes_continuos_mutados
            }

        return genes_mutados

    def roleta(self, tamanho_roleta: int) -> dict:
        """Escolhe aleatoriamente tamanho_roleta candidatos de self.populacao e usa seus valores de fitness para escolher um"""
        candidatos = random.sample(self.populacao, tamanho_roleta)

        notas = np.array([candidato['fitness'] for candidato in candidatos])
        notas = notas - np.min(notas) + 1e-9

        if np.all(notas == 0):
            probabilidade_escolha = np.ones_like(notas) / len(notas)

        else:
            probabilidade_escolha = notas / np.sum(notas)

        vencedor = random.choices(candidatos, weights=probabilidade_escolha, k=1)[0]
        return vencedor

    def treinar(self, qtd_geracoes: int, taxa_crossover: float, taxa_mutacao: float, tamanho_roleta: int, paciencia: int, proporcao_desvio: float = 0.1, range_eta_c: list[float] = [2, 10], range_resolucao: list[float] = [0.1, 0.01]):
        """
        Evolui os indivíduos do algorítmo com os parâmetros dados. Para automaticamente se não tiver melhoria por muitas gerações.
        
        Parameters
        ----------
        qtd_geracoes: int
            É a quantidade de iterações máxima para convergência. A evolução para se não chegar na geração de número qtd_geracoes.
        
        taxa_crossover: float
            É a probabilidade de acontecer crossover quando cruzando dois indivíduos, varia de 0.0 a 1.0. Uma taxa de crossover menor que 1 deixa a possibilidade de clonagem, que é bom para refino de resultados ao sofrerem mutações. Normalmente é usado 0.8.
        
        taxa_mutacao: float
            É a probabilidade de indivíduos sofrerem mutação, o que causa refino de soluções, ao aplicar pequenas alterações a uma solução existente. Usualmente é usado 0.05.
        
        tamanho_roleta: int
            É a quantidade de indivíduos escolhidos para a seleção por roleta. Escolher que nem todos participem da roleta ajuda a garantir melhor variabilidade. Sugiro usar um máximo de 20 indivíduos aqui, pois a probabilidade escolha é dividida entre cada indivíduo, e usar muitos deixa a escolha basicamente aleatória. 

        paciencia: int
            Determina a quantidade de gerações máximas sem melhoria, parando o ciclo após paciencia gerações, para garantir que o algorítmo não continue mesmo tendo já convergido para uma solução.
        
        proporcao_desvio (opcional): float
            Para mutação de variáveis contínuas, a variável contínua é alterada usando uma distribuição normal com desvio padrão de proporcao_desvio * range. Um valor de 0.1 é bom, mas depende do problema. Sugiro entre 0.1 e 0.5.
        
        range_eta_c (opcional): list[float]
            Controla os valores inicial e final do parâmetro eta_c, do crossover SBX. Durante o treino, eta_c varia de eta_c_inicial a eta_c_max. Recomendo iniciar eta_c por volta de 2 e finalizar por volta de 10 a 20.

        range_resolucao (opcional): list[float]
            Controla os valores inicial e final do parâmetro resolucao_continua, para salvar no cache. Durante o treino, esse parametro varia de resolucao_inicial a resolucao_mac. Representa proporção do range original. Por exemplo: [0.1, 0.01] inicial com resolução de 10% do range original e finaliza com resolução de 1%.

        Notes
        -----
        tamanho_roleta:
            Para a roleta, o valor de Fitness de cada indivíduo é normalizado e usado como probabilidade de escolha, mas em média, podemos ter uma ideia da chance de algum indivíduo qualquer ser escolhido usando a mesma probabilidade para cada um. Com 20 indivíduos na roleta, temos uma média de 5% de chance de ser escolhido, não fazendo sentido usar roletas maiores que 20 indivíduos.
        
        eta_c:
            Esse AG usa crossover SBX para variáveis contínuas, que possibilita refino ou extrapolação de soluções usando o parâmetro eta_c. Valores altos de eta_c fazem o cruzamento de variáveis contínuas serem próximos da média, tendo comportamento de "refino", enquanto valores baixos de eta_c deixam maior variação, tendo comportamento de "exploração".
        
        resolucao:
            Esse AG usa cache para não reavaliar soluções já calculadas, então os valores contínuos são arredondados para resolucao% do range antes de serem salvos ou procurados no cache.
        """
        geracoes_sem_melhora = 0

        eta_c = range_eta_c[0]
        resolucao = range_resolucao[0]
        passo_eta_c = (range_eta_c[1] - range_eta_c[0]) / (0.8 * paciencia)
        passo_resolucao = (range_resolucao[1] - range_resolucao[0]) / (0.8 * paciencia)

        for geracao in range(qtd_geracoes):
            # Cria uma população nova a partir da população da geração anterior
            for i in range(self.tamanho_populacao - 1):
                pai = self.roleta(tamanho_roleta)
                mae = self.roleta(tamanho_roleta)

                # Crossover ou clonagem
                if np.random.random() < taxa_crossover:
                    genes_filho = self.crossover(pai, mae, eta_c)
                else:
                    parente = random.choice([pai, mae])
                    genes_filho = {
                        'genes_discretos': parente['genes_discretos'].copy(),
                        'genes_continuos': parente['genes_continuos'].copy()
                    }

                # Mutação
                if np.random.random() < taxa_mutacao:
                    genes_filho = self.mutacao(genes_filho, proporcao_desvio)

                self.genes_populacao[i] = genes_filho.copy()

            # Elitismo
            melhor = self.melhores[-1]
            self.genes_populacao[-1] = {
                'genes_discretos': melhor['genes_discretos'].copy(),
                'genes_continuos': melhor['genes_continuos'].copy()
            }

            # Avalia a nova população
            self.populacao = self.avaliar_genes(self.genes_populacao, resolucao)
            self.melhores.append(max(self.populacao, key=lambda x: x['fitness']))

            # Se melhorou, reseta a contagem
            if self.melhores[-1]['fitness'] > self.melhores[-2]['fitness']:
                geracoes_sem_melhora = 0

            # Senão, soma mais um na contagem e aplica um equivalente do reduce_lr_on_plateau para variáveis contínuas
            else:
                eta_c = min(range_eta_c[1], eta_c + passo_eta_c)
                resolucao = min(range_resolucao[1], resolucao + passo_resolucao)
                self.cache_fitness.clear()
                
                geracoes_sem_melhora += 1

            # Early stopping
            if geracoes_sem_melhora >= paciencia:
                break

        print(f'Treino realizado com sucesso em {geracao + 1} gerações!')
        