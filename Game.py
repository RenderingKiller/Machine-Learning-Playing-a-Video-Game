##################################################################
#####################   Author: Tianshi xie   ####################
#####################       09/20/2017        ####################
##################################################################

import pygame
from pygame.locals import *
from pygame import *
from Object import *
import sys
import random

class Pos:
    def __init__(self, m, n):
        self.x = m
        self.y = n

class MyGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.enemys = pygame.sprite.Group()
        self.score = 0
        self.max_score = 0
        self.running = True
        self.clock = pygame.time.Clock()
        self.enemy_born_cooldown = 1500
        self.last = pygame.time.get_ticks()
        self.shoot_frequency = 0
        self.isboundingbox = 0
        self.level = 0
        self.getTicksLastFrame = pygame.time.get_ticks()
        self.LoadRes()

    def LoadRes(self):
        self.bg = pygame.image.load('../res/picture/background.png').convert()
        self.over = pygame.image.load('../res/picture/gameover.png')
        self.total_image = pygame.image.load('../res/picture/sheet.png')

        self.player_area = []
        self.player_area.append(pygame.Rect(247,84,99,75))
        self.player_area.append(pygame.Rect(247,84,99,75))
        self.player_area.append(pygame.Rect(247,84,99,75))  # dead animation
        self.player_area.append(pygame.Rect(247,84,99,75))
        self.player_area.append(pygame.Rect(247,84,99,75))
        self.player_area.append(pygame.Rect(247,84,99,75))
        player_pos = [200, 600]
        self.player = Player(self.total_image, self.player_area, player_pos)

        self.bullet_img = self.total_image.subsurface(pygame.Rect(1004,987,9,21))

        self.enemy_areas = []
        self.enemy_areas.append(pygame.Rect(222,0,103,84))
        self.enemy_areas.append(pygame.Rect(267, 347, 57, 43))
        self.enemy_areas.append(pygame.Rect(873, 697, 57, 43))
        self.enemy_areas.append(pygame.Rect(267, 296, 57, 43))
        self.enemy_areas.append(pygame.Rect(930, 697, 57, 43))
        self.enemy_image = self.total_image.subsurface(self.enemy_areas[0])

    def Replay(self):
        self.running = True
        self.player.b_hit = False
        self.score = 0
        self.player.img_index = 0
        self.enemy_born_rate = 0;
        self.shoot_frequency = 0
        self.player.rect.topleft = [200, 600]
        self.getTicksLastFrame = pygame.time.get_ticks()
        self.enemys.empty()

    def ShowScoreInGame(self, name, val, pos):

        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(name + str(val), True, (128, 128, 128))
        text_rect = score_text.get_rect()
        text_rect.topleft = pos
        self.display.blit(score_text, text_rect)

    def LevelUp(self):
        self.level = int(1+self.score/5)
        if self.level >= 6:
            self.level = 6

        self.enemy_born_cooldown = 1500 - self.level * 200
        if self.enemy_born_cooldown <= 300:
            self.enemy_born_cooldown = 300

    def ShowScoreOutGame(self):
        font = pygame.font.Font(None, 48)
        text = font.render('MaxScore: ' + str(self.score), True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.centerx = self.display.get_rect().centerx
        text_rect.centery = self.display.get_rect().centery + 24
        self.display.blit(self.over, (0, 0))
        self.display.blit(text, text_rect)

        text = font.render('Score: ' + str(self.max_score), True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.centerx = self.display.get_rect().centerx
        text_rect.centery = self.display.get_rect().centery + 60
        self.display.blit(self.over, (0, 0))
        self.display.blit(text, text_rect)

    def GetDelta(self):
        t = pygame.time.get_ticks()
        # deltaTime in seconds.
        deltaTime = (t - self.getTicksLastFrame) / 1000.0
        self.getTicksLastFrame = t

        return deltaTime

    def Step(self):
        reward = 0.1
        terminal = False

        self.clock.tick(60)

        self.delta = self.GetDelta()

        now = pygame.time.get_ticks()
        if now - self.last >= self.enemy_born_cooldown:
            self.last = now
            random_pos = [random.randint(0, DISPLAY_WIDTH - self.enemy_areas[0].width), 0]
            new_enemy = Enemy(self.enemy_image, self.enemy_areas, random_pos)
            new_enemy.speed += (1500 - self.enemy_born_cooldown)*0.2
            self.enemys.add(new_enemy)

        for enemy in self.enemys:
            enemy.move(1, self.delta)

            if pygame.sprite.collide_circle(enemy, self.player):
                self.enemys.remove(enemy)
                self.player.b_hit = True
                terminal = True
                reward = -1
                break;

            if enemy.rect.top > DISPLAY_HEIGHT:
                self.enemys.remove(enemy)
                self.score += 1
                reward = 1
                if self.max_score < self.score:
                    self.max_score = self.score

        self.LevelUp()

        # render
        self.display.fill(0)
        #self.display.blit(self.bg, (0, 0))

        if not self.player.b_hit:
            self.display.blit(self.player.images[self.player.img_index], self.player.rect)
            # pygame.draw.ellipse(self.display, (255, 0, 0), self.player.rect, 2)
            if self.isboundingbox:
                pygame.draw.circle(self.display, (255, 0, 0),
                                   [int(self.player.rect.left + self.player.rect.width / 2.0),
                                    int(self.player.rect.top + self.player.rect.height / 2.0)],
                                   int(self.player.rect.width / 2.0))
            self.player.img_index = self.shoot_frequency // 8
        else:
            self.display.blit(self.player.images[self.player.img_index], self.player.rect)
            self.player.img_index += 1
            if self.player.img_index >= len(self.player.images):
                self.running = False
                terminal = True
                reward = -1

        for enemy in self.enemys:
            self.display.blit(enemy.image, enemy.rect)
            # pygame.draw.rect(self.display,(255,   0,   0), enemy.rect,1)
            # pygame.draw.ellipse(self.display, (255,   0,   0), enemy.rect, 2)
            if self.isboundingbox:
                pygame.draw.circle(self.display, (255, 0, 0),
                                   [int(enemy.rect.left + enemy.rect.width / 2.0),
                                    int(enemy.rect.top + enemy.rect.height / 2.0)],
                                   int(enemy.rect.width / 2.0))

        self.ShowScoreInGame('MaxScore: ', self.max_score, [10, 10])
        self.ShowScoreInGame('Score: ', self.score, [10, 40])
        self.ShowScoreInGame('Level: ', self.level, [10, 70])
        self.ShowScoreInGame('enemy_born_cooldown: ', self.enemy_born_cooldown, [10, 100])

        # update screen
        pygame.display.update()

        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_a] or key_pressed[K_LEFT]:
            self.player.move([0,1,0],self.delta)
        if key_pressed[K_d] or key_pressed[K_RIGHT]:
            self.player.move([0,0,1],self.delta)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        return image_data, reward, terminal

    def StepAgent(self, action):

        reward = 0.1
        terminal = False

        self.delta = self.GetDelta()
        self.player.move(action, self.delta)

        now = pygame.time.get_ticks()
        if now - self.last >= self.enemy_born_cooldown:
            self.last = now
            random_pos = [random.randint(0, DISPLAY_WIDTH - self.enemy_areas[0].width), 0]
            new_enemy = Enemy(self.enemy_image, self.enemy_areas, random_pos)
            new_enemy.speed += (1500 - self.enemy_born_cooldown)*0.2
            self.enemys.add(new_enemy)

        for enemy in self.enemys:
            enemy.move(1, self.delta)

            if pygame.sprite.collide_circle(enemy, self.player):
                self.enemys.remove(enemy)
                self.player.b_hit = True
                terminal = True
                self.Replay()
                reward = -1
                break;

            if enemy.rect.top > DISPLAY_HEIGHT:
                self.enemys.remove(enemy)
                self.score += 1
                reward = 1
                if self.max_score < self.score:
                    self.max_score = self.score

        self.LevelUp()

        #render
        self.display.fill(0)

        if not self.player.b_hit:
            # pygame.draw.ellipse(self.display, (255, 0, 0), self.player.rect, 2)
            if self.isboundingbox:
                pygame.draw.circle(self.display, (255, 0, 0),
                                   [int(self.player.rect.left + self.player.rect.width / 2.0),
                                    int(self.player.rect.top + self.player.rect.height / 2.0)],
                                   int(self.player.rect.width / 2.0))
        else:
            self.running = False
            terminal = True
            self.Replay()
            reward = -1

        self.display.blit(self.player.images[self.player.img_index], self.player.rect)
        for enemy in self.enemys:
            self.display.blit(enemy.image, enemy.rect)
            # pygame.draw.rect(self.display,(255,   0,   0), enemy.rect,1)
            # pygame.draw.ellipse(self.display, (255,   0,   0), enemy.rect, 2)
            if self.isboundingbox:
                pygame.draw.circle(self.display, (255, 0, 0),
                                   [int(enemy.rect.left + enemy.rect.width / 2.0),
                                    int(enemy.rect.top + enemy.rect.height / 2.0)],
                                   int(enemy.rect.width / 2.0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        self.ShowScoreInGame('MaxScore: ', self.max_score, [10, 10])
        self.ShowScoreInGame('Score: ', self.score, [10, 40])
        self.ShowScoreInGame('Level: ', self.level, [10, 70])
        self.ShowScoreInGame('enemy_born_cooldown: ', self.enemy_born_cooldown, [10, 100])

        self.clock.tick(60)
        # update screen
        pygame.display.flip()

        return image_data, reward, terminal

    def Run(self):
        while 1:
            while(self.running):
                self.Step()

            self.ShowScoreOutGame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYUP:
                    self.Replay()
                    break;

            pygame.display.flip()

    def playGame(self):
        while(1):
            _, _, done = self.Step()
            if done == True:
                self.Replay()

if __name__ == "__main__":
    game = MyGame()
    game.playGame()
