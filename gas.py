import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Particula:
    def __init__(self, velocidade: float, angulo_velocidade: float, raio: float, massa: float) -> None:
        self.velocidade = velocidade
        self.angulo_velocidade = angulo_velocidade
        self.massa = massa
        self.raio = raio

        self.momento = massa * velocidade
    
    def __repr__(self) -> str:
        return "Particula({}, {}, {}, {})".format(self.velocidade, self.angulo_velocidade, self.raio, self.massa)
    
    def __str__(self) -> str:
        return "Partícula de massa {}, com velocidade {} e ângulo {}".format(self.massa, self.velocidade, self.angulo_velocidade)

p1 = Particula(1,2,5,10)