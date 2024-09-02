import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Particula:
    def __init__(self, velocidade_x, velocidade_y, massa) -> None:
        velocidade_x = velocidade_x
        velocidade_y = velocidade_y
        massa = massa

        momento_x = massa * velocidade_x
        momento_y = massa * velocidade_y

