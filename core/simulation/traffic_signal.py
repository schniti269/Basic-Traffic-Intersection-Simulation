import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from shared.utils import signals, currentGreen, nextGreen, currentYellow, noOfSignals
from shared.utils import defaultGreen, defaultYellow, defaultRed, directionNumbers
from shared.utils import vehicles, defaultStop, updateValues, logger, MANUAL_CONTROL
import time


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(
        ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1]
    )
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)

    if MANUAL_CONTROL:
        repeat()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while (
        signals[currentGreen].green > 0
    ):  # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1  # set yellow signal on
    # reset stop coordinates of lanes and vehicles
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while (
        signals[currentGreen].yellow > 0
    ):  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 0  # set yellow signal off

    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen  # set next signal as green signal
    nextGreen = (currentGreen + 1) % noOfSignals  # set next green signal
    signals[nextGreen].red = (
        signals[currentGreen].yellow + signals[currentGreen].green
    )  # set the red time of next to next signal as (yellow time + green time) of next signal

    if MANUAL_CONTROL:
        repeat()
