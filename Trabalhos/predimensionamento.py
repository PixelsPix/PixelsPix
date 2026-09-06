"""
Pré-dimensionamento de uma estrutura 2D por modelo de duas barras.

Modelo preliminar:
- chapa de espessura fixa;
- duas barras ideais ligadas ao ponto de aplicação da força;
- ângulos orientados das barras e da força medidos em relação à horizontal;
- N > 0 = tração; N < 0 = compressão;
- flambagem de Euler considerada FORA DO PLANO da chapa;
- para flambagem fora do plano: I = w*t^3/12;
- o fator de segurança é aplicado explicitamente;
- permite varrer vários ângulos da força e procurar os melhores
  pares de ângulos das barras.

IMPORTANTE:
Este é um modelo de pré-dimensionamento. A peça real não é uma
treliça ideal e uma lâmina de compósito pode exigir um modelo
ortotrópico/laminado e análise de concentrações de tensão, flambagem
de placa e região do furo em uma etapa posterior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray


# ---------------------------------------------------------------------
# PROPRIEDADES
# ---------------------------------------------------------------------

PROPRIEDADES_CHAPA = {
    "espessura": 5e-3,          # m
    "sigma_tracao": 513e6,      # Pa (resistência estimada)
    "sigma_compressao": 473e6,  # Pa (resistência estimada)
    "modulo_young": 59e9,       # Pa
}


class SimuladorTrelica:
    """
    Modelo de duas barras para pré-dimensionamento.

    Convenção angular:
        theta = ângulo orientado em relação à horizontal.

    A convenção permite ângulos entre 0 e 180 graus. Assim, uma barra
    pode estar à esquerda ou à direita da projeção vertical do furo.

    Convenção dos esforços:
        N > 0 -> tração
        N < 0 -> compressão

    A força é definida por:
        theta_f = 0 graus -> força para a direita
        theta_f = 180 graus -> força para a esquerda

    Se a sua convenção de desenho usar a seta da força para a esquerda
    como theta_f = 0 graus, basta deslocar os theta_f usados na entrada
    de acordo com essa convenção; as equações internas permanecem as
    mesmas.
    """

    def __init__(
        self,
        altura_furo: float,
        forca: float,
        angulos_barras_deg: tuple[float, float],
        angulos_forca_deg: tuple[float, float, float] = (-30.0, 30.1, 5.0),
        propriedades: dict = PROPRIEDADES_CHAPA,
        fator_seguranca: float = 1.5,
        fator_k_flambagem: float = 1.0,
        b_h_min: float | None = None,
    ) -> None:

        self.h = float(altura_furo)
        self.F = float(forca)
        self.propriedades = propriedades
        self.SF = float(fator_seguranca)
        self.K = float(fator_k_flambagem)
        self.b_h_min = b_h_min

        if self.h <= 0:
            raise ValueError("altura_furo deve ser positiva.")

        if self.F <= 0:
            raise ValueError("forca deve ser positiva.")

        if self.SF <= 0:
            raise ValueError("fator_seguranca deve ser positivo.")

        if self.K <= 0:
            raise ValueError("fator_k_flambagem deve ser positivo.")

        if len(angulos_barras_deg) != 2:
            raise ValueError("São necessários exatamente dois ângulos de barras.")

        self.theta_barras_rad = np.deg2rad(
            np.asarray(angulos_barras_deg, dtype=float)
        )

        self.theta_forca_rad = np.deg2rad(
            np.arange(*angulos_forca_deg)
        )

        # Para L = h/sin(theta), evitamos theta próximo de 0 ou 180 graus.
        if np.any(np.isclose(np.sin(self.theta_barras_rad), 0.0)):
            raise ValueError(
                "Ângulos das barras muito próximos de 0 ou 180 graus "
                "produzem comprimento infinito."
            )

        # Duas barras paralelas/colineares não permitem resolver o equilíbrio.
        if np.isclose(
            np.sin(self.theta_barras_rad[1] - self.theta_barras_rad[0]),
            0.0,
        ):
            raise ValueError(
                "Os ângulos das duas barras não podem ser iguais ou colineares."
            )

    # -----------------------------------------------------------------
    # PROPRIEDADES DO MATERIAL
    # -----------------------------------------------------------------

    @property
    def sigma_adm_tracao(self) -> float:
        """Tensão admissível à tração após aplicação do fator de segurança."""
        return self.propriedades["sigma_tracao"] / self.SF

    @property
    def sigma_adm_compressao(self) -> float:
        """Tensão admissível à compressão após aplicação do fator de segurança."""
        return self.propriedades["sigma_compressao"] / self.SF

    # -----------------------------------------------------------------
    # GEOMETRIA
    # -----------------------------------------------------------------

    @property
    def comprimentos_barras(self) -> NDArray:
        """
        Comprimentos das barras.

        L = h / sin(theta)

        Os ângulos são orientados em relação à horizontal.
        """
        return self.h / np.abs(np.sin(self.theta_barras_rad))

    @property
    def coord_barras_chao(self) -> NDArray:
        """
        Coordenadas horizontais dos pontos onde as barras encontram
        o chão, tomando a projeção vertical do furo como x = 0.

        x = h / tan(theta)

        O sinal indica o lado do furo.
        """
        return self.h / np.tan(self.theta_barras_rad)

    @property
    def distancia_frontal(self) -> float:
        """
        Distância entre a projeção vertical do furo e o ponto de apoio
        mais à esquerda.

        Para uma geometria com um apoio de cada lado do furo:
            distancia_frontal = -min(x1, x2)
        """
        return float(-np.min(self.coord_barras_chao))

    def geometria_valida(self) -> bool:
        """
        Verifica condições geométricas básicas.

        1. Os apoios devem ficar em lados opostos da projeção do furo.
        2. Se b_h_min foi fornecido, o apoio mais à esquerda deve estar
           pelo menos b_h_min distante da projeção do furo.
        """
        x1, x2 = self.coord_barras_chao

        apoios_opostos = (x1 < 0.0) and (x2 > 0.0)

        if not apoios_opostos:
            return False

        if self.b_h_min is not None:
            if self.distancia_frontal < self.b_h_min:
                return False

        return True

    # -----------------------------------------------------------------
    # ESFORÇOS INTERNOS
    # -----------------------------------------------------------------

    @property
    def esforcos_internos(self) -> NDArray:
        """
        Calcula os esforços axiais pelas fórmulas analíticas:

            N1 = F * sin(theta2 - theta_f) / sin(theta2 - theta1)

            N2 = F * sin(theta_f - theta1) / sin(theta2 - theta1)

        Retorno:
            array (n_forcas, 2)

            coluna 0 -> N1
            coluna 1 -> N2

        N > 0: tração
        N < 0: compressão
        """
        theta_1, theta_2 = self.theta_barras_rad
        theta_f = self.theta_forca_rad

        denominador = np.sin(theta_2 - theta_1)

        N1 = (
            self.F
            * np.sin(theta_2 - theta_f)
            / denominador
        )

        N2 = (
            self.F
            * np.sin(theta_f - theta_1)
            / denominador
        )

        return np.column_stack((N1, N2))

    # -----------------------------------------------------------------
    # DIMENSIONAMENTO POR TENSÃO
    # -----------------------------------------------------------------

    @property
    def larguras_por_tensao(self) -> NDArray:
        """
        Largura mínima para satisfazer o critério de tensão axial:

            sigma = |N| / (w*t)

        A tensão admissível já inclui o fator de segurança.
        """
        N = self.esforcos_internos
        t = self.propriedades["espessura"]

        sigma_adm = np.where(
            N >= 0.0,
            self.sigma_adm_tracao,
            self.sigma_adm_compressao,
        )

        return np.abs(N) / (t * sigma_adm)

    # -----------------------------------------------------------------
    # FLAMBAGEM FORA DO PLANO
    # -----------------------------------------------------------------

    @property
    def larguras_por_flambagem(self) -> NDArray:
        """
        Dimensionamento por flambagem de Euler fora do plano.

        Se a seção da barra é:
            largura = w
            espessura = t

        então:

            I = w*t^3/12

        e:

            Pcr = pi^2*E*I/(K*L)^2

        Impõe-se:

            Pcr >= SF * |N|

        apenas para barras comprimidas.
        """
        N = self.esforcos_internos
        L = self.comprimentos_barras
        t = self.propriedades["espessura"]
        E = self.propriedades["modulo_young"]

        # Cada linha corresponde a um caso de força.
        # Cada coluna corresponde a uma barra.
        N_comp = np.maximum(-N, 0.0)

        largura = (
            12.0
            * self.SF
            * N_comp
            * (self.K * L[None, :]) ** 2
            / (
                np.pi**2
                * E
                * t**3
            )
        )

        return largura

    # -----------------------------------------------------------------
    # LARGURA FINAL
    # -----------------------------------------------------------------

    @property
    def larguras_minimas(self) -> NDArray:
        """
        Para cada barra e caso de carga:

            w = max(w_tensao, w_flambagem)
        """
        return np.maximum(
            self.larguras_por_tensao,
            self.larguras_por_flambagem,
        )

    # -----------------------------------------------------------------
    # MATERIAL
    # -----------------------------------------------------------------

    @property
    def area_material(self) -> NDArray:
        """
        Área da chapa no plano 2D aproximada pelos dois caminhos de carga:

            A_material = sum(w_i * L_i)
        """
        L = self.comprimentos_barras
        w = self.larguras_minimas

        return np.sum(w * L[None, :], axis=1)

    @property
    def volume_material(self) -> NDArray:
        """Volume correspondente à área 2D vezes a espessura."""
        return self.area_material * self.propriedades["espessura"]

    # -----------------------------------------------------------------
    # RESUMO
    # -----------------------------------------------------------------

    def resultados(self) -> pd.DataFrame:
        """
        Retorna uma tabela com os resultados de cada ângulo da força.
        """
        N = self.esforcos_internos
        w_sigma = self.larguras_por_tensao
        w_buck = self.larguras_por_flambagem
        w = self.larguras_minimas

        df = pd.DataFrame({
            "theta_forca_deg": np.rad2deg(self.theta_forca_rad),

            "N1_N": N[:, 0],
            "N2_N": N[:, 1],

            "w1_tensao_m": w_sigma[:, 0],
            "w2_tensao_m": w_sigma[:, 1],

            "w1_flambagem_m": w_buck[:, 0],
            "w2_flambagem_m": w_buck[:, 1],

            "w1_final_m": w[:, 0],
            "w2_final_m": w[:, 1],

            "area_material_m2": self.area_material,
            "volume_material_m3": self.volume_material,
        })

        return df


# =====================================================================
# VARREDURA DE ÂNGULOS
# =====================================================================

def varrer_angulos(
    altura_furo: float,
    forca: float,
    theta1_deg: NDArray,
    theta2_deg: NDArray,
    angulos_forca_deg: tuple[float, float, float] = (-30.0, 30.1, 5.0),
    propriedades: dict = PROPRIEDADES_CHAPA,
    fator_seguranca: float = 1.5,
    fator_k_flambagem: float = 1.0,
    b_h_min: float | None = None,
) -> pd.DataFrame:
    """
    Testa todas as combinações de theta1 e theta2.

    Uma configuração só é aceita se os dois apoios ficarem em lados
    opostos da projeção vertical do furo e se a restrição b_h_min,
    quando fornecida, for satisfeita.

    O custo de uma configuração é o pior caso de área de material
    entre todos os ângulos de força testados.
    """

    resultados = []

    for theta1 in theta1_deg:
        for theta2 in theta2_deg:

            try:
                sim = SimuladorTrelica(
                    altura_furo=altura_furo,
                    forca=forca,
                    angulos_barras_deg=(theta1, theta2),
                    angulos_forca_deg=angulos_forca_deg,
                    propriedades=propriedades,
                    fator_seguranca=fator_seguranca,
                    fator_k_flambagem=fator_k_flambagem,
                    b_h_min=b_h_min,
                )
            except ValueError:
                continue

            if not sim.geometria_valida():
                continue

            df = sim.resultados()

            # Critério conservador:
            # a geometria deve ser dimensionada para o pior caso de carga.
            idx_pior = df["area_material_m2"].idxmax()

            resultados.append({
                "theta1_deg": theta1,
                "theta2_deg": theta2,

                "x1_m": sim.coord_barras_chao[0],
                "x2_m": sim.coord_barras_chao[1],

                "L1_m": sim.comprimentos_barras[0],
                "L2_m": sim.comprimentos_barras[1],

                "area_pior_caso_m2": df.loc[
                    idx_pior, "area_material_m2"
                ],

                "volume_pior_caso_m3": df.loc[
                    idx_pior, "volume_material_m3"
                ],

                "theta_forca_pior_caso_deg": df.loc[
                    idx_pior, "theta_forca_deg"
                ],

                "w1_pior_m": df.loc[idx_pior, "w1_final_m"],
                "w2_pior_m": df.loc[idx_pior, "w2_final_m"],

                "N1_pior_N": df.loc[idx_pior, "N1_N"],
                "N2_pior_N": df.loc[idx_pior, "N2_N"],
            })

    return (
        pd.DataFrame(resultados)
        .sort_values("area_pior_caso_m2")
        .reset_index(drop=True)
    )


# =====================================================================
# EXEMPLO DE USO
# =====================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # PARÂMETROS DO PROBLEMA
    # ---------------------------------------------------------------

    h_h = 0.100       # m
    F = 1000.0        # N

    SF = 1.5
    K = 1.0

    # Se b_h representar a distância mínima do furo até o apoio
    # mais à esquerda:
    b_h = 0.020       # m

    # Forças de -30° a +30°, em passos de 5°.
    # Ajuste a convenção de theta_f conforme a sua definição angular.
    angulos_forca = (-30.0, 30.1, 5.0)

    # ---------------------------------------------------------------
    # VARREDURA
    # ---------------------------------------------------------------

    angulos = np.arange(10.0, 171.0, 1.0)

    resultados = varrer_angulos(
        altura_furo=h_h,
        forca=F,
        theta1_deg=angulos,
        theta2_deg=angulos,
        angulos_forca_deg=angulos_forca,
        propriedades=PROPRIEDADES_CHAPA,
        fator_seguranca=SF,
        fator_k_flambagem=K,
        b_h_min=b_h,
    )

    if resultados.empty:
        print("Nenhuma configuração satisfez as restrições geométricas.")
        raise SystemExit

    # ---------------------------------------------------------------
    # MELHORES CONFIGURAÇÕES
    # ---------------------------------------------------------------

    print("\n=== 10 MELHORES CONFIGURAÇÕES ===\n")

    colunas = [
        "theta1_deg",
        "theta2_deg",
        "L1_m",
        "L2_m",
        "w1_pior_m",
        "w2_pior_m",
        "area_pior_caso_m2",
        "theta_forca_pior_caso_deg",
    ]

    print(
        resultados[colunas]
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # MELHOR CONFIGURAÇÃO
    # ---------------------------------------------------------------

    melhor = resultados.iloc[0]

    print("\n=== MELHOR CONFIGURAÇÃO ===")
    print(f"theta1 = {melhor['theta1_deg']:.2f}°")
    print(f"theta2 = {melhor['theta2_deg']:.2f}°")
    print(f"L1     = {melhor['L1_m']*1000:.2f} mm")
    print(f"L2     = {melhor['L2_m']*1000:.2f} mm")
    print(f"w1     = {melhor['w1_pior_m']*1000:.2f} mm")
    print(f"w2     = {melhor['w2_pior_m']*1000:.2f} mm")
    print(
        f"Área 2D = {melhor['area_pior_caso_m2']*1e6:.2f} mm²"
    )
    print(
        f"Força crítica = "
        f"{melhor['theta_forca_pior_caso_deg']:.2f}°"
    )

    # ---------------------------------------------------------------
    # DETALHE DA MELHOR CONFIGURAÇÃO
    # ---------------------------------------------------------------

    sim_melhor = SimuladorTrelica(
        altura_furo=h_h,
        forca=F,
        angulos_barras_deg=(
            melhor["theta1_deg"],
            melhor["theta2_deg"],
        ),
        angulos_forca_deg=angulos_forca,
        propriedades=PROPRIEDADES_CHAPA,
        fator_seguranca=SF,
        fator_k_flambagem=K,
        b_h_min=b_h,
    )

    print("\n=== DETALHAMENTO DA MELHOR CONFIGURAÇÃO ===\n")
    print(sim_melhor.resultados().to_string(index=False))

    # ---------------------------------------------------------------
    # MAPA DE CALOR
    # ---------------------------------------------------------------

    # Para visualizar o espaço de projeto, usamos os resultados
    # aceitos pela restrição geométrica.
    mapa = resultados.pivot(
        index="theta1_deg",
        columns="theta2_deg",
        values="area_pior_caso_m2",
    )

    plt.figure(figsize=(9, 7))

    imagem = plt.imshow(
        mapa.values * 1e6,
        origin="lower",
        aspect="auto",
        extent=[
            mapa.columns.min(),
            mapa.columns.max(),
            mapa.index.min(),
            mapa.index.max(),
        ],
    )

    plt.colorbar(imagem, label="Área de material [mm²]")
    plt.scatter(
        melhor["theta2_deg"],
        melhor["theta1_deg"],
        marker="x",
        s=100,
    )

    plt.xlabel(r"$\theta_2$ [graus]")
    plt.ylabel(r"$\theta_1$ [graus]")
    plt.title("Área de material no pior caso de carregamento")
    plt.tight_layout()
    plt.show()
