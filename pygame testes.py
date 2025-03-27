import pygame
from sys import exit

pygame.init()
pygame.display.set_caption("Simulador")

screen_width  = 800
screen_height = 900
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# coordenadas: primeiras duas var sao as coordenadas do ponto superior esquerdo
lado_caixa = 500

# superficie para desenho
test_surface = pygame.Surface((lado_caixa, lado_caixa))
test_surface.fill("White")

# imagem_surface = pygame.image.load("[path relativa pra imagem]").convert_alpha()
# test_font = pygame.font.Font(Fonte, tamanho)
# texto_surface = test_font.render(texto, AntiAliasing, cor)

test_font = pygame.font.Font(None, 30)
text_surface = test_font.render("texto", True, "Black")

funcionando = True
while funcionando:
    for evento_usuario in pygame.event.get():
        # fecha a janela quando sai do loop
        if evento_usuario.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    # funcionamento
    screen.blit(test_surface,(0,0))
    screen.blit(text_surface,(0,0))

    pygame.display.update()
    clock.tick(60)