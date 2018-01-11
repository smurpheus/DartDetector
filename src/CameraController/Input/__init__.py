#!/usr/bin/env python
from threading import Thread, RLock, Event, Lock
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
        self.CALLBACKS = {}
        self.running = Event()
        self.ready = False
        self.controller = None
        self.buttons = 0
        self.axes = 0
        self.eventlock = Lock()
        self.callbacklock = Lock()
        pygame.init()
        pygame.joystick.init()

    def get_available_controller(self):
        self.reinitialize()
        number = pygame.joystick.get_count()
        joys = []
        for i in range(number):
            j = pygame.joystick.Joystick(i)
            joys.append((i, j.get_name()))
        return joys

    def get_available_axes(self):
        if self.ready:
            return self.axes
        return None

    def get_available_buttons(self):
        if self.ready:
            return self.buttons
        return None

    def init_controller(self, id):
        self.controller = pygame.joystick.Joystick(id)
        self.controller.init()
        self.buttons = self.controller.get_numbuttons()
        self.axes = self.controller.get_numaxes()
        self.ready = True
        self.emit("READY")

    def wait_for(self, event, timeout=None):
        if event in self.EVENTS.keys():
            self.EVENTS[event].wait(timeout)
        else:
            with self.eventlock:
                self.EVENTS[event] = Event()
            self.EVENTS[event].wait(timeout)
        with self.eventlock:
            self.EVENTS[event] = Event()
        return

    def reinitialize(self):
        pygame.quit()
        pygame.init()
        pygame.joystick.init()

    def register_to_event(self, event, callback):
        with self.callbacklock:
            if event in self.CALLBACKS.keys():
                self.CALLBACKS[event] = self.CALLBACKS[event].append(callback)
            else:
                self.CALLBACKS[event] = [callback]

    def emit(self, event, *args):
        with self.eventlock:
            if event in self.EVENTS.keys():
                self.EVENTS[event].set()
                self.EVENTS[event] = Event()
        with self.callbacklock:
            if event in self.CALLBACKS.keys():
                for fnc in self.CALLBACKS[event]:
                    fnc(args)

    def run(self):
        while not self.running.is_set():
            with self.eventlock, self.callbacklock:
                watched_axes = [x for x in self.EVENTS.keys() if "Axis" in x]
                watched_buttons = [x for x in self.EVENTS.keys() if "Button" in x]
                watched_axes += [x for x in self.CALLBACKS.keys() if "Axis" in x]
                watched_buttons += [x for x in self.CALLBACKS.keys() if "Button" in x]
            for e in pygame.event.get():  # iterate over event stack
                # print('event : ' + str(e.type))
                if e.type == pygame.locals.JOYAXISMOTION and len(watched_axes) > 0:  # 7
                    for a in watched_axes:
                        i = int(a.replace("Axis", ""))
                        if i == e.axis:
                            self.emit("Axis{}".format(i), e.value)

                if e.type == pygame.locals.JOYBUTTONDOWN and len(watched_buttons) > 0:
                    for a in watched_buttons:
                        i = int(a.replace("Button", ""))
                        if i == e.button:
                            self.emit("Button{}".format(i), True)
