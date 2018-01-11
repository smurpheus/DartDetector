#!/usr/bin/env python
from threading import Thread, RLock, Event
import pygame
import time
from pygame import locals
from pygame import joystick

__author__ = "Dwarfeus"
__copyright__ = "Copyright 2018, Dwarfeus"


class Controller(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.EVENTS = {"READY": Event()}
        self.running = Event()
        self.ready = False
        self.callbacks = {}
        pygame.init()
        pygame.joystick.init()

    def wait_for(self, event, timeout=None):
        if event in self.EVENTS.keys():
            self.EVENTS[event].wait(timeout)
        else:
            self.EVENTS[event] = Event()
            self.EVENTS[event].wait(timeout)
        self.EVENTS[event] = Event()
        return

    def get_available_controller(self):
        self.reinitialize()
        number = pygame.joystick.get_count()
        joys = []
        for i in range(number):
            j = pygame.joystick.Joystick(i)
            joys.append((i, j.get_name()))
        return joys

    def init_controller(self, id):
        self.controller = pygame.joystick.Joystick(id)
        self.controller.init()
        self.ready = True
        self.emit("READY")

    def register_to_event(self, event, callback):
        if event in self.callbacks.keys():
            self.callbacks[event] = self.callbacks[event].append(callback)
        else:
            self.callbacks[event] = [callback]

    def emit(self, event, *args):
        if event in self.EVENTS.keys():
            self.EVENTS[event].set()
            self.EVENTS[event] = Event()
        if event in self.callbacks.keys():
            for fnc in self.callbacks[event]:
                fnc(args)

    def run(self):
        buttons = self.controller.get_numbuttons()
        axes = self.controller.get_numaxes()
        while not self.running.is_set():
            for e in pygame.event.get():  # iterate over event stack
                # print('event : ' + str(e.type))
                if e.type == pygame.locals.JOYAXISMOTION:  # 7
                    # x, y = self.controller.get_axis(0), self.controller.get_axis(1)

                    for i in range(axes):
                        axis = self.controller.get_axis(i)
                        self.emit("Axis{}".format(i), axis)

                if e.type == pygame.locals.JOYBUTTONDOWN:
                    for i in range(buttons):
                        button = self.controller.get_button(i)
                        if button > 0:
                            self.emit("Button{}".format(i), button)
                        # print("Axis {} value: {:>6.3f}".format(i, axis))
                    # print('x and y : ' + str(x) + ' , ' + str(y))
                # elif e.type == pygame.locals.JOYBALLMOTION:  # 8
                #     print('ball motion')
                # elif e.type == pygame.locals.JOYHATMOTION:  # 9
                #     print('hat motion')
                # elif e.type == pygame.locals.JOYBUTTONDOWN:  # 10
                #     print('button down')
                # elif e.type == pygame.locals.JOYBUTTONUP:  # 11
                #     print('button up')

    def reinitialize(self):
        pygame.quit()
        pygame.init()
        pygame.joystick.init()


def printvalue(result):
    print("X{}".format(result))


def printyvalue(result):
    print("Y{}".format(result))


def printbutton(re):
    print("Button0{}".format(re))


c = Controller()

c.init_controller(0)
c.register_to_event("Axis0", printvalue)
c.register_to_event("Axis1", printyvalue)
c.register_to_event("Button0", printbutton)
c.start()
c.wait_for("Button7")
c.running.set()
# c.emit("Axis0", 0)
# c.do_run()
