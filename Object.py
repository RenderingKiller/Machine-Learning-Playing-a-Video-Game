##################################################################
#####################   Author: Tianshi xie   ####################
#####################       09/20/2017        ####################
##################################################################

import pygame

DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 800

class Object(pygame.sprite.Sprite):
    def __init__(self, img, rects, pos):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        for i in range(len(rects)):
            self.images.append(img.subsurface(rects[i]).convert_alpha())
        self.rect = rects[0]
        self.rect.topleft = pos
        self.speed = 10
        self.bullets = pygame.sprite.Group()
        self.img_index = 0
        self.b_hit = False
        self.radius = self.rect.width/2


class Bullet(Object):
    def __init__(self, img, pos):
        Object.__init__(self, img, img.get_rect(), pos)
        self.image = img
        self.rect = self.image.get_rect()
        self.speed = 10

class Character(Object):
    def __init__(self, img, rects, pos):
        Object.__init__(self, img, rects, pos)
        self.speed_scale = 4

    def shoot(self, bullet_img):
        bullet = Bullet(bullet_img, self.rect.midtop)
        self.bullets.add(bullet)

    def move(self, dir, delta):
        if dir[1] == 1: #left
            if self.rect.left <= 0:
                self.rect.left = 0
            else:
                self.rect.left -= self.speed * delta * self.speed_scale
        elif dir[2] == 1:
            if self.rect.left >= DISPLAY_WIDTH - self.rect.width:
                self.rect.left = DISPLAY_WIDTH - self.rect.width
            else:
                self.rect.left += self.speed * delta * self.speed_scale

class Player(Character):
    def __init__(self, img, rects, pos):
        Character.__init__(self, img, rects, pos)
        self.speed = 300

class Enemy(Character):
    def __init__(self, img, rects, pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.rect = img.get_rect()
        self.rect.topleft = pos
        self.b_hit = False
        self.speed = 50
        self.radius = self.rect.width / 2
        self.speed_scale = 4

    def move(self, dir, delta):
        self.rect.top += self.speed * delta * self.speed_scale


































































































