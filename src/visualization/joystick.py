from typing import Tuple

import pygame
import torch


class ExitException(Exception):
    pass


class JoystickController:

    def __init__(self):
        
        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self) -> torch.Tensor:
        pygame.event.get()
        
        if self.joystick.get_button(8):
            raise ExitException('Back button pushed. Exiting Simulation.')

        return torch.tensor([[self.joystick.get_button(3),self.joystick.get_button(2),
                                self.joystick.get_button(1),self.joystick.get_button(0)]])


if __name__ == '__main__':
    joystick = JoystickController().joystick
    print('Buttons\t', end='\t')
    print('Axes')

    while True:
        _ = pygame.event.get()
        for i in range(joystick.get_numbuttons()):
            print(joystick.get_button(i), end='')
        print(end='\t')

        for i in range(joystick.get_numaxes()):
            print(f'{joystick.get_axis(i):2f}', end='\t')

        print(end='\r') 
