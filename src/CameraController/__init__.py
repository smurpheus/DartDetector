#!/usr/bin/env python
from sys import stdout
from threading import Lock, Event, Thread

import time

import os

from src.CameraController.Input import Controller

def servo_value(inperc):
    return inperc * 0.000005 + 0.05

class SteeringUnit(Thread):
    Deadzone = 0.18
    Sensitivity = 0.5

    def __init__(self, maxx=90, minx=-90, maxy=90, miny=-90):
        Thread.__init__(self)
        self.max_x = maxx
        self.max_y = maxy
        self.min_x = minx
        self.min_y = miny
        self.current_x = 0
        self.current_y = 0
        self.current_lock = Lock()
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_lock = Lock()
        self.running = Event()

    def run(self):
        while not self.running.is_set():
            with self.current_lock, self.velocity_lock:
                if abs(self.velocity_x) > self.Deadzone:
                    self.change_x(self.velocity_x * self.Sensitivity)
                # else:
                #     print("In Neutral X")
                if abs(self.velocity_y) > self.Deadzone:
                    self.change_y(self.velocity_y * self.Sensitivity)
                os.system("echo 17={} > /dev/pi-blaster".format(self.current_y))
                os.system("echo 18={} > /dev/pi-blaster".format(self.current_x))
                stdout.write("\rX:{:02.2f}Y:{:02.2f}".format(self.current_x, self.current_y))
                stdout.flush()
                # else:
                #     print("In Neutral Y")
            time.sleep(0.0001)


    def change_x(self, value):
        # print("Changing by: {}".format(value))
        if self.current_x + value < self.max_x and self.current_x + value > self.min_x:
            self.current_x += value
            # print("New X: {}".format(self.current_x))
        # else:
            # print("Maximum X reached!")

    def change_y(self, value):
        if self.current_y + value < self.max_y and self.current_y + value > self.min_y:
            self.current_y += value
            # print("New Y: {}".format(self.current_y))
        # else:
            # print("Maximum Y reached!")

    def set_velocity_x(self, value):
        value = value[0]
        with self.velocity_lock:
            self.velocity_x = value
    def set_velocity_y(self, value):
        value = value[0]
        with self.velocity_lock:
            self.velocity_y = value

    def reset(self, *args):
        with self.current_lock:
            self.current_x = 0
            self.current_y = 0





c = Controller()
su = SteeringUnit(maxx=1000, minx=0, maxy=1000, miny=0)
c.init_controller(0)
c.register_to_event("Axis0", su.set_velocity_x)
c.register_to_event("Axis1", su.set_velocity_y)
c.register_to_event("Button6", su.reset)
c.start()
su.start()
c.wait_for("Button7")
su.running.set()
c.running.set()
c.emit("Axis0", 0)
c.do_run()
