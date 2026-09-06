import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray

propriedades_chapa = {
    'espessura':       5e-3, # m
    'tensoes_maximas': np.array([-473e6, 513e6]), # Pa
    'modulo_young':    7e9,  # Pa
}

class SimuladorHornSimples:
    def __init__(
            self, altura_furo: float, forca: float, angulos_barras_deg: list[float], theta_forca_deg: tuple[float] = (-30.0, 30.1, 5),
            propriedades=propriedades_chapa, fator_seguranca=1.5,
        ) -> None:

        self.propriedades = propriedades
        self.altura_furo  = altura_furo

        self.forca = forca
        self.fator_seguranca = fator_seguranca
        self.sigma_min, self.sigma_max = self.propriedades['tensoes_maximas'] / fator_seguranca

        self.thetas_forca_rad = np.deg2rad(np.arange(*theta_forca_deg))
        self.theta_barras_rad = np.deg2rad(np.array(angulos_barras_deg))


    @property
    def comprimento_barras(self) -> NDArray:
        return np.array([
            self.altura_furo / np.sin(theta)
            for theta in self.theta_barras_rad
        ])


    @property
    def coord_barras_chao(self) -> NDArray:
        """(0,0) é no chão logo abaixo do furo, primeiro a barra esquerda e depois a direita"""
        return np.array([
            self.altura_furo / np.tan(theta)
            for theta in self.theta_barras_rad
        ])

    
    @property
    def esforcos_internos(self) -> NDArray:
        """Esforços para cada ângulo da força"""
        theta_1_rad, theta_2_rad = self.theta_barras_rad
        fator_esforco = self.forca / np.sin(theta_2_rad - theta_1_rad) 
    
        vetores_angulos = np.array([
            [np.sin(theta_2_rad - theta_f_rad), np.sin(theta_f_rad - theta_1_rad)]
            for theta_f_rad in self.thetas_forca
        ])
    
        return fator_esforco * vetores_angulos


    def _largura_min(self, esforco_interno, comprimento):
        N = esforco_interno
        L = comprimento

        tracao = N >= 0
        if tracao:
             # Só critério de tensão
            w_tensao = N / (self.propriedades['espessura'] * self.sigma_max)
            w_flambagem = 1e-3 # define largura minima de 1 mm p/ nao quebrar algoritmo
        
        else:
            # Critério de tensão
            w_tensao = N / (self.propriedades['espessura'] * self.sigma_min)

            # Critério de flambagem - flambagem para fora do 2d
            k = 2 # engastado-livre
            num = 12 * self.fator_seguranca * -N * (k * L)**2
            den = np.pi**2 * self.propriedades['modulo_young'] * self.propriedades['espessura']**3

            w_flambagem = num / den

        return np.max(w_tensao, w_flambagem)


    def deslocamento2(self):
        """Retorna o quadrado do deslocamento total (dx^2 + dy^2)"""
        pass

    def calcular(self) -> dict[str, float]:
        """Retorna volume/área de material deflexão total"""
        pass