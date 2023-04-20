import copy
import json
import struct
from enum import Enum
import random
import numpy as np
import cv2
from controller import Robot, DistanceSensor, PositionSensor, GPS, Camera
import tensorflow as tf
from PIL import Image

ROBOT_RADIUS = .04


def save_image(array: np.ndarray):
    name = f"D:\\mojef\\Ferdowsi-Erebus\\train\\{random.randint(0, 1000)}.png"
    im = Image.fromarray(array.astype(np.uint8))
    im.save(name)


# enums
class GameColors(Enum):
    other = 0
    black = 2
    orange = 3
    gray = 4
    green = 5
    blue = 6
    purple = 7
    red = 8
    silver = 9


class VictimTypes(Enum):
    wall = 0
    sign_or_victim = 1


class MoveState(Enum):
    forward = 0
    turnLeft = 1
    turnRight = 2
    turnBack = 3
    moveBack = 4
    stop = 5


class Direction(Enum):
    left = 0
    up = 1
    right = 2
    down = 4
    not_initialized = -1


class AIStates(Enum):
    random_searching = 0
    not_seen_searching = 1
    wall_following = 2
    returning = 3


class AroundStatus(Enum):
    is_wall = 0
    is_empty = 1
    is_seen = 2


class TurnState(Enum):
    not_started = 0
    turning = 1


class MapBonus:
    def __init__(self):
        self.MAP_SIZE = 80
        self.map = [[0 for _ in range(2 * self.MAP_SIZE + 1)] for _ in range(2 * self.MAP_SIZE + 1)]

    def update_map(self):
        self.update_walls()
        self.update_floors()

    def update_walls(self):
        pass

    def fill_start_tile(self):
        x, y = baby_location.startingTilePos
        self.set_map_floor(x, y, GameColors.green)

    def update_floors(self):
        self.fill_start_tile()
        color = baby_cam.get_color()
        x = baby_location.light_x_tile
        y = baby_location.light_y_tile
        if baby_location.direction != Direction.not_initialized:
            self.set_map_floor(x, y, color)

    def set_map_floor(self, tileX, tileY, color: GameColors):
        if color != GameColors.other and (color == GameColors.green or baby_location.light_in_tile_center()):
            # if color != GameColors.green:
            # print(
            #     f"robot-tile : {baby_location.tilePosX, baby_location.tilePosY}, light-tile : {tileX, tileY} , color:{color}")
            x1 = tileX * 2
            x2 = tileX * 2 + 1
            y1 = tileY * 2
            y2 = tileY * 2 + 1
            self.fill_map_bonus_color(x1, y1, color)
            self.fill_map_bonus_color(x1, y2, color)
            self.fill_map_bonus_color(x2, y1, color)
            self.fill_map_bonus_color(x2, y2, color)

    def fill_map_bonus_color(self, x_small_tile, y_small_tile, color: GameColors):
        self.map[2 * x_small_tile + 1][2 * y_small_tile + 1] = str(color.value)


# Control Classes
class LocationClass:
    def __init__(self, robot: Robot):
        self.MAP_SIZE = 40
        self.gps: GPS = robot.getDevice("gps")  # Retrieve the gps by device name
        self.gps.enable(timeStep)
        self.x = 0
        self.y = 0
        self.tilePosX = 0
        self.tilePosY = 0
        self.offsetX = 0
        self.offsetY = 0
        self.direction = Direction.not_initialized
        self.isStuck = False
        self.history = []
        self.map = [[0 for _ in range(2 * self.MAP_SIZE + 1)] for _ in range(2 * self.MAP_SIZE + 1)]
        self.last_tile = (0, 0)
        self.startingTilePos = (0, 0)
        self.lightX = 0
        self.lightY = 0
        self.light_x_tile = 0
        self.light_y_tile = 0
        self.NextPosForward = (0, 0)
        self.NextPosRight = (0, 0)
        self.NextPosLeft = (0, 0)
        self.NextPosBackward = (0, 0)
        self.stuckCounter = 0
        self.blockChanged = False
        self.areaBlockChanged = False
        self.passedBlocksCounter = 0
        self.repeatedBlocksCounter = 0
        self.move_straight_error = 0.0

    def set_tile_pos(self):
        self.map[2 * self.tilePosX + 1][2 * self.tilePosY + 1] = 1
        if baby_location.robot_in_tile_center() and baby_location.direction != Direction.not_initialized:

            if baby_status.front_status == AroundStatus.is_wall:
                front_tile_x, front_tile_y = self.NextPosForward
                self.map[front_tile_x + self.tilePosX + 1][front_tile_y + self.tilePosY + 1] = 2

            if baby_status.right_status == AroundStatus.is_wall:
                right_tile_x, right_tile_y = self.NextPosRight
                self.map[right_tile_x + self.tilePosX + 1][right_tile_y + self.tilePosY + 1] = 2

            if baby_status.left_status == AroundStatus.is_wall:
                left_tile_x, left_tile_y = self.NextPosLeft
                self.map[left_tile_x + self.tilePosX + 1][left_tile_y + self.tilePosY + 1] = 2

            if baby_status.behind_status == AroundStatus.is_wall:
                behind_tile_x, behind_tile_y = self.NextPosBackward
                self.map[behind_tile_x + self.tilePosX + 1][behind_tile_y + self.tilePosY + 1] = 2

    def init_parameters(self):
        self.update_parameters()
        self.offsetX = self.x % .12
        self.offsetY = self.y % .12
        self.update_parameters()
        self.startingTilePos = (self.tilePosX, self.tilePosY)

    def update_parameters(self):
        pos = self.gps.getValues()
        self.x = pos[0]
        self.y = pos[2]
        self.last_tile = (self.tilePosX, self.tilePosY)
        self.tilePosX = int((self.x - self.offsetX + 0.06) // 0.12 + self.MAP_SIZE // 2)
        self.tilePosY = int((self.y - self.offsetY + 0.06) // 0.12 + self.MAP_SIZE // 2)
        self.set_tile_pos()
        self.add_point()
        if len(self.history) > 9:
            self.history.remove(self.history[0])
        self.lightX, self.lightY = self.get_light_pos()
        self.light_x_tile = int((self.lightX - self.offsetX + 0.06) // 0.12 + self.MAP_SIZE // 2)
        self.light_y_tile = int((self.lightY - self.offsetY + 0.06) // 0.12 + self.MAP_SIZE // 2)
        if len(self.history) == 9:
            self.direction = self.get_direction()
        else:
            self.direction = Direction.not_initialized

        self.isStuck = self.is_stuck()
        self.increasing_stuck_counter()
        self.NextPosForward = self.get_next_pos_forward()
        self.NextPosBackward = self.get_next_pos_backward()
        self.NextPosLeft = self.get_next_pos_left()
        self.NextPosRight = self.get_next_pos_right()
        self.get_sub_area()

        if self.last_tile != (self.tilePosX, self.tilePosY):
            self.blockChanged = True
            self.areaBlockChanged = True
            self.passedBlocksCounter += 1
        # set_game_map(self.lightXPos, self.lightXPos, colorControl.get_color())

        if self.direction == Direction.up or self.direction == Direction.down:
            self.move_straight_error = self.history[0][0] - self.history[-1][0]
            if self.direction == Direction.down:
                self.move_straight_error *= -1
        elif self.direction == Direction.right or self.direction == Direction.left:
            self.move_straight_error = self.history[0][1] - self.history[-1][1]
            if self.direction == Direction.left:
                self.move_straight_error *= -1
        else:
            self.move_straight_error = 0

    def get_direction(self):
        deltaX = self.history[-1][0] - self.history[0][0]
        deltaY = self.history[-1][1] - self.history[0][1]
        if abs(deltaX) > abs(deltaY):
            if deltaX > 0:
                return Direction.right
            else:
                return Direction.left
        else:
            if deltaY < 0:
                return Direction.up
            else:
                return Direction.down

    def get_next_pos_forward(self):
        direction = self.get_direction()
        if direction == Direction.right:
            return [self.tilePosX + 1, self.tilePosY]
        elif direction == Direction.left:
            return [self.tilePosX - 1, self.tilePosY]
        elif direction == Direction.up:
            return [self.tilePosX, self.tilePosY - 1]
        elif direction == Direction.down:
            return [self.tilePosX, self.tilePosY + 1]

    def get_next_pos_backward(self):
        direction = self.get_direction()
        if direction == Direction.right:
            return [self.tilePosX - 1, self.tilePosY]
        elif direction == Direction.left:
            return [self.tilePosX + 1, self.tilePosY]
        elif direction == Direction.up:
            return [self.tilePosX, self.tilePosY + 1]
        elif direction == Direction.down:
            return [self.tilePosX, self.tilePosY - 1]
        else:
            return [self.tilePosX, self.tilePosY]

    def get_next_pos_right(self):
        direction = self.get_direction()
        if direction == Direction.right:
            return [self.tilePosX, self.tilePosY + 1]
        elif direction == Direction.left:
            return [self.tilePosX, self.tilePosY - 1]
        elif direction == Direction.up:
            return [self.tilePosX + 1, self.tilePosY]
        elif direction == Direction.down:
            return [self.tilePosX - 1, self.tilePosY]
        else:
            return [self.tilePosX, self.tilePosY]

    def get_next_pos_left(self):
        direction = self.get_direction()
        if direction == Direction.right:
            return [self.tilePosX, self.tilePosY - 1]
        elif direction == Direction.left:
            return [self.tilePosX, self.tilePosY + 1]
        elif direction == Direction.up:
            return [self.tilePosX - 1, self.tilePosY]
        elif direction == Direction.down:
            return [self.tilePosX + 1, self.tilePosY]
        else:
            return [self.tilePosX, self.tilePosY]

    def get_sub_area(self):
        if (self.x - self.offsetX) % .12 < .06:
            return 1 if (self.y - self.offsetY) % .12 > .06 else 3
        else:
            return 2 if (self.y - self.offsetY) % .12 > .06 else 4

    def get_light_pos(self):
        direction = self.get_direction()
        lightOffsetX = 0 if direction in (
            Direction.up,
            Direction.down) else -1.5 * ROBOT_RADIUS if direction == Direction.left else 1.5 * ROBOT_RADIUS
        lightOffsetY = 0 if direction in (
            Direction.left,
            Direction.right) else -1.5 * ROBOT_RADIUS if direction == Direction.up else 1.5 * ROBOT_RADIUS
        return self.x + lightOffsetX, self.y + lightOffsetY

    def is_stuck(self):
        deltaX = self.history[-1][0] - self.history[0][0]
        deltaY = self.history[-1][1] - self.history[0][1]
        if abs(deltaX) < 1e-5 and abs(deltaY) < 1e-5:
            return True
        else:
            return False

    def increasing_stuck_counter(self):
        if self.isStuck:
            self.stuckCounter += 1
        else:
            self.stuckCounter = 0

    def add_point(self):
        if len(self.history) == 0:
            self.history.append((self.x, self.y))
        else:
            avg_x = (self.x + self.history[-1][0]) / 2
            avg_y = (self.y + self.history[-1][1]) / 2
            self.history.append((avg_x, avg_y))

    def is_tile_seen(self, pos):
        return self.map[2 * pos[0] + 1][2 * pos[1] + 1] == 1

    def robot_in_tile_center(self):
        return (0.055 <= (self.y - self.offsetY + 0.06) % 0.12 < 0.065 and self.direction in [
            Direction.up, Direction.down]) or \
               (0.055 <= (self.x - self.offsetX + 0.06) % 0.12 < 0.065 and self.direction in [
                   Direction.left, Direction.right])

    def light_in_tile_center(self):
        return (0.055 <= (self.lightY - self.offsetY + 0.06) % 0.12 < 0.065 and self.direction in [
            Direction.up, Direction.down]) or \
               (0.055 <= (self.lightX - self.offsetX + 0.06) % 0.12 < 0.065 and self.direction in [
                   Direction.left, Direction.right])


class RobotControlClass:
    def __init__(self, robot: Robot):
        self.state = MoveState.forward
        self.max_velocity = 5
        self.turn_state = TurnState.not_started
        self.direction = Direction.not_initialized
        self.left_wheel = robot.getDevice("wheel1 motor")
        self.right_wheel = robot.getDevice("wheel2 motor")
        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))
        self.rightWheelPosSensor: PositionSensor = self.right_wheel.getPositionSensor()
        self.leftWheelPosSensor: PositionSensor = self.left_wheel.getPositionSensor()
        self.rightWheelPosSensor.enable(timeStep)
        self.leftWheelPosSensor.enable(timeStep)
        self.stopCounter = 0
        self.leftWheelPos = 0
        self.rightWheelPos = 0
        self.stopFlag = False
        self.leftWheelSpeed = 0
        self.rightWheelSpeed = 0

    def run(self):
        if self.state == MoveState.forward:
            self.move_forward()
        elif self.state == MoveState.turnLeft:
            self.turn_left()
        elif self.state == MoveState.turnRight:
            self.turn_right()
        elif self.state == MoveState.turnBack:
            self.turn_back()
        elif self.state == MoveState.moveBack:
            self.move_back()
        elif self.state == MoveState.stop:
            self.stop()

    def move(self):
        self.state = MoveState.forward

    def dont_move(self):
        self.state = MoveState.stop
        self.stopFlag = False

    def move_forward(self):
        e = baby_location.move_straight_error * 150
        e = max(e, -.3)
        e = min(e, .3)
        er = 0
        el = 0

        if .04 < baby_status.s2.getValue() < .13:
            el = .1
            er = .05

        if .04 < baby_status.s4.getValue() < .13:
            er = .1
            el = .05

        self.leftWheelSpeed = self.max_velocity * .8 - e + er
        self.rightWheelSpeed = self.max_velocity * .8 + e + el
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)

    def turn_left(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            self.left_wheel.setVelocity(self.max_velocity * 0.3)
            self.right_wheel.setVelocity(-self.max_velocity * 0.3)

            if self.leftWheelPosSensor.getValue() > (self.leftWheelPos + 4.42 / 2):
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def turn_right(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            self.left_wheel.setVelocity(-self.max_velocity * 0.3)
            self.right_wheel.setVelocity(self.max_velocity * 0.3)

            if self.leftWheelPosSensor.getValue() < (self.leftWheelPos - 4.42 / 2):
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def turn_back(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            self.left_wheel.setVelocity(self.max_velocity * 0.3)
            self.right_wheel.setVelocity(-self.max_velocity * 0.3)

            if self.leftWheelPosSensor.getValue() > (self.leftWheelPos + 4.42):
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def move_back(self):
        baby_location.history.clear()
        leftWheelSpeed = -self.max_velocity
        rightWheelSpeed = -self.max_velocity
        self.left_wheel.setVelocity(leftWheelSpeed)
        self.right_wheel.setVelocity(rightWheelSpeed)

    def stop(self):
        self.stopCounter += 1

        if self.stopCounter >= 100 - 10:
            self.stopFlag = True

        if self.stopCounter >= 100:
            self.stopCounter = 0
            self.state = MoveState.forward
            # print("Sending victim done!")

        baby_location.history.clear()
        self.left_wheel.setVelocity(0)
        self.right_wheel.setVelocity(0)


class StatusClass:
    def __init__(self, robot: Robot):
        self.left_status = AroundStatus.is_empty
        self.right_status = AroundStatus.is_empty
        self.front_status = AroundStatus.is_empty
        self.behind_status = AroundStatus.is_empty
        self.s1: DistanceSensor = robot.getDevice("ds0")
        self.s2: DistanceSensor = robot.getDevice("ds1")
        self.s3: DistanceSensor = robot.getDevice("ds2")
        self.s4: DistanceSensor = robot.getDevice("ds3")
        self.s5: DistanceSensor = robot.getDevice("ds4")
        self.s6: DistanceSensor = robot.getDevice("ds5")
        self.s1.enable(timeStep)
        self.s2.enable(timeStep)
        self.s3.enable(timeStep)
        self.s4.enable(timeStep)
        self.s5.enable(timeStep)
        self.s6.enable(timeStep)

    def update_status(self):
        if self.s1.getValue() > 0.08 and self.s5.getValue() > 0.03 and self.s6.getValue() > 0.03:
            if baby_location.is_tile_seen(baby_location.NextPosForward):
                self.front_status = AroundStatus.is_seen
            else:
                self.front_status = AroundStatus.is_empty
        else:
            self.front_status = AroundStatus.is_wall

        if self.s2.getValue() > 0.08:
            if baby_location.is_tile_seen(baby_location.NextPosRight):
                self.right_status = AroundStatus.is_seen
            else:
                self.right_status = AroundStatus.is_empty
        else:
            self.right_status = AroundStatus.is_wall

        if self.s4.getValue() > 0.08:
            if baby_location.is_tile_seen(baby_location.NextPosLeft):
                self.left_status = AroundStatus.is_seen
            else:
                self.left_status = AroundStatus.is_empty
        else:
            self.left_status = AroundStatus.is_wall

        if self.s3.getValue() > 0.08:
            if baby_location.is_tile_seen(baby_location.NextPosBackward):
                self.behind_status = AroundStatus.is_seen
            else:
                self.behind_status = AroundStatus.is_empty
        else:
            self.behind_status = AroundStatus.is_wall

        # print(
        #     f"S1:{self.s1.getValue()} , S6: {self.s6.getValue()}, S5: {self.s5.getValue()}, Forward status : {self.front_status}")


class ReturnPath:
    list_hazfi = []
    all_path = []
    startx = 0
    starty = 0
    map_size = 7

    def __init__(self, x, y):  # Saving start tile positions
        self.startx = x
        self.starty = y
        self.map_size = len(baby_location.map)

    def get_path(self, x, y):  # receiving robot current position

        if self.startx == x and self.starty == y:
            self.list_hazfi.append((x, y))
            self.all_path.append(copy.copy(self.list_hazfi))
            self.list_hazfi.remove((x, y))

        if 2 * y - 1 >= 0 and (
                baby_location.map[2 * x + 1][2 * y + 1 - 2] == 1 or (x == self.startx and y - 1 == self.starty)) and (
                x, y - 1) not in self.list_hazfi and \
                baby_location.map[2 * x + 1][2 * y + 1 - 1] != 2:
            self.list_hazfi.append((x, y))
            self.get_path(x, y - 1)
            self.list_hazfi.remove((x, y))

        if 2 * y + 3 < self.map_size and (
                baby_location.map[2 * x + 1][2 * y + 1 + 2] == 1 or (x == self.startx and y + 1 == self.starty)) and (
                x, y + 1) not in self.list_hazfi and \
                baby_location.map[2 * x + 1][2 * y + 1 + 1] != 2:
            self.list_hazfi.append((x, y))
            self.get_path(x, y + 1)
            self.list_hazfi.remove((x, y))

        if 2 * x - 1 >= 0 and (
                baby_location.map[2 * x + 1 - 2][2 * y + 1] == 1 or (x - 1 == self.startx and y == self.starty)) and (
                x - 1, y) not in self.list_hazfi and \
                baby_location.map[2 * x + 1 - 1][
                    2 * y + 1] != 2:
            self.list_hazfi.append((x, y))
            self.get_path(x - 1, y)
            self.list_hazfi.remove((x, y))

        if 2 * x + 3 < self.map_size and baby_location.map[2 * x + 1 + 2][2 * y + 1] == 1 or (
                x + 1 == self.startx and y == self.starty) and \
                (x + 1, y) not in self.list_hazfi and \
                baby_location.map[2 * x + 1 + 1][2 * y + 1] != 2:
            self.list_hazfi.append((x, y))
            self.get_path(x + 1, y)
            self.list_hazfi.remove((x, y))

        return self.all_path

    def get_best_path(self, x, y):
        self.all_path.clear()
        self.get_path(x, y)

        if len(self.all_path) == 0:
            return []
        best_path = self.all_path[0]

        for path in self.all_path:
            if len(path) < len(best_path):
                best_path = path

        return best_path


# AI CLASS
class AIPlannerClass:
    def __init__(self, robot: Robot):
        self.ai_state = AIStates.random_searching
        # Retrieve the receiver and emitter by device name
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(timeStep)
        self.not_seen_tiles = set()
        # Enable the receiver. Note that the emitter does not need to call enable()
        self.score = 0
        self.remained_time = 1000
        self.initial_time = -1
        self.area_number = 1

    def area_detect(self):
        if baby_location.areaBlockChanged:
            # print(baby_cam.get_color())
            baby_location.areaBlockChanged = False
            if self.area_number == 1:
                if baby_cam.get_color() == GameColors.blue:
                    self.area_number = 2
                elif baby_cam.get_color() == GameColors.green:
                    self.area_number = 4

            elif self.area_number == 2:
                if baby_cam.get_color() == GameColors.blue:
                    self.area_number = 1
                elif baby_cam.get_color() == GameColors.purple:
                    self.area_number = 3

            elif self.area_number == 3:
                if baby_cam.get_color() == GameColors.purple:
                    self.area_number = 2
                elif baby_cam.get_color() == GameColors.red:
                    self.area_number = 4

            elif self.area_number == 4:
                if baby_cam.get_color() == GameColors.red:
                    self.area_number = 3
                elif baby_cam.get_color() == GameColors.green:
                    self.area_number = 1
            # print(f"area detected, {self.area_number}")

    def choose_state(self):
        if self.initial_time == -1:
            self.ai_state = AIStates.random_searching
            return
        if self.remained_time / self.initial_time > .9:
            self.ai_state = AIStates.random_searching
        # elif self.remained_time / self.initial_time > .3:
        #     self.ai_state = AIStates.wall_following
        else:
            self.ai_state = AIStates.returning
        # baby_controller.state = MoveState.forward

    def plan(self):
        self.update_game_time_score()
        baby_status.update_status()
        baby_location.update_parameters()
        baby_cam.check_victim()
        baby_map_bonus.update_map()
        self.area_detect()
        self.choose_state()
        if baby_controller.state != MoveState.stop:

            if self.ai_state == AIStates.random_searching:
                self.random_search()
            elif self.ai_state == AIStates.wall_following:
                self.wall_follow()
            elif self.ai_state == AIStates.returning:
                self.return_start_tile()
            elif self.ai_state == AIStates.not_seen_searching:
                self.go_to_not_seen_tile()
        # baby_controller.dont_move()
        # print(
        # f"s1:{baby_status.s1.getValue()},s2:{baby_status.s2.getValue()},"
        # f"s3:{baby_status.s3.getValue()},s4:{baby_status.s4.getValue()},"
        # f"location:{baby_location.tilePosX,baby_location.tilePosY},x,y:{baby_location.x,baby_location.y}")
        baby_controller.run()

    def random_search(self):
        global baby_search_finder
        # print(baby_location.direction,baby_status.front_status,baby_status.right_status,baby_status.left_status,baby_status.behind_status)
        if (baby_location.tilePosX, baby_location.tilePosY) in self.not_seen_tiles:
            self.not_seen_tiles.remove((baby_location.tilePosX, baby_location.tilePosY))
        if baby_controller.state != MoveState.forward or baby_location.direction == Direction.not_initialized:
            return

        if baby_location.stuckCounter > 30:

            if baby_status.s3.getValue() < 0.2 or baby_status.s5.getValue() < 0.2:
                baby_controller.state = MoveState.turnRight

            elif baby_status.s1.getValue() < 0.2 or baby_status.s6.getValue() < 0.2:
                baby_controller.state = MoveState.turnLeft

            elif baby_status.s2.getValue() < 0.2 and baby_status.s4.getValue() < 0.2:
                baby_controller.state = MoveState.turnBack

            return

        if baby_status.front_status == AroundStatus.is_wall or (
                baby_location.robot_in_tile_center() and baby_location.blockChanged) \
                or baby_cam.get_color() == GameColors.black:
            allChoice = []
            emptyChoice = []
            baby_location.blockChanged = False
            baby_location.stuckCounter = 0

            if baby_status.front_status != AroundStatus.is_wall and baby_cam.get_color() != GameColors.black:
                allChoice.append(MoveState.forward)

            if baby_status.front_status == AroundStatus.is_empty and baby_cam.get_color() != GameColors.black:
                emptyChoice.append(MoveState.forward)

            if baby_status.left_status != AroundStatus.is_wall:
                allChoice.append(MoveState.turnLeft)

            if baby_status.left_status == AroundStatus.is_empty:
                emptyChoice.append(MoveState.turnLeft)

            if baby_status.right_status != AroundStatus.is_wall:
                allChoice.append(MoveState.turnRight)

            if baby_status.right_status == AroundStatus.is_empty:
                emptyChoice.append(MoveState.turnRight)

            if baby_status.behind_status != AroundStatus.is_wall and \
                    (baby_status.front_status == AroundStatus.is_wall
                     and baby_status.left_status == AroundStatus.is_wall
                     and baby_status.right_status == AroundStatus.is_wall):
                allChoice.append(MoveState.turnBack)

            if baby_status.left_status == AroundStatus.is_seen and baby_status.right_status == AroundStatus.is_seen \
                    and baby_status.front_status != AroundStatus.is_wall:
                baby_controller.state = MoveState.forward

            if baby_status.left_status == AroundStatus.is_seen and baby_status.right_status == AroundStatus.is_seen \
                    and baby_status.front_status == AroundStatus.is_wall:
                baby_controller.state = MoveState.turnBack

            if len(emptyChoice) > 0:
                baby_controller.state = random.choice(emptyChoice)
                if len(emptyChoice) > 1:
                    emptyChoice.remove(baby_controller.state)
                    for choice in emptyChoice:
                        if choice == MoveState.forward:
                            self.not_seen_tiles.add(tuple(baby_location.NextPosForward))

                        elif choice == MoveState.turnLeft:
                            self.not_seen_tiles.add(tuple(baby_location.NextPosLeft))

                        elif choice == MoveState.turnRight:
                            self.not_seen_tiles.add(tuple(baby_location.NextPosRight))

                        elif choice == MoveState.turnBack:
                            print("is back")
                            self.not_seen_tiles.add(tuple(baby_location.NextPosBackward))
            elif len(self.not_seen_tiles) > 0:
                target_tile = self.not_seen_tiles.pop()
                baby_search_finder = ReturnPath(*target_tile)
                self.ai_state = AIStates.not_seen_searching
                print("go to not seen section")
                print(baby_search_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY))
            elif len(allChoice) > 0:
                baby_controller.state = random.choice(allChoice)

            else:
                baby_controller.state = MoveState.forward

    def wall_follow(self):
        pass

    def return_start_tile(self):
        print("in return section *************************************************************** ")
        best_path = baby_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY)

        if baby_location.tilePosX == baby_location.startingTilePos[0] and baby_location.tilePosY == \
                baby_location.startingTilePos[1] and baby_location.robot_in_tile_center():
            self.send_finish()

        if len(best_path) < 2:
            return

        p1 = best_path[0]
        p2 = best_path[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if baby_location.robot_in_tile_center() and baby_location.blockChanged:
            baby_location.blockChanged = False

            if baby_location.direction == Direction.up:
                if dx == 1:
                    baby_controller.state = MoveState.turnRight
                if dx == -1:
                    baby_controller.state = MoveState.turnLeft
                if dy == 1:
                    baby_controller.state = MoveState.turnBack
                if dy == -1:
                    baby_controller.state = MoveState.forward

            if baby_location.direction == Direction.down:
                if dx == 1:
                    baby_controller.state = MoveState.turnLeft
                if dx == -1:
                    baby_controller.state = MoveState.turnRight
                if dy == 1:
                    baby_controller.state = MoveState.forward
                if dy == -1:
                    baby_controller.state = MoveState.turnBack

            if baby_location.direction == Direction.left:
                if dx == 1:
                    baby_controller.state = MoveState.turnBack
                if dx == -1:
                    baby_controller.state = MoveState.forward
                if dy == 1:
                    baby_controller.state = MoveState.turnLeft
                if dy == -1:
                    baby_controller.state = MoveState.turnRight

            if baby_location.direction == Direction.right:
                if dx == 1:
                    baby_controller.state = MoveState.forward
                if dx == -1:
                    baby_controller.state = MoveState.turnBack
                if dy == 1:
                    baby_controller.state = MoveState.turnRight
                if dy == -1:
                    baby_controller.state = MoveState.turnLeft

    def send_victim(self, typ):
        x, y = baby_location.x * 100, baby_location.y * 100
        victim_type = bytes(typ, "utf-8")  # The victim type being sent is the victimChar for harmed victim
        message = struct.pack("i i c", int(x), int(y), victim_type)  # Pack message
        self.emitter.send(message)  # Send out the message

    def send_finish(self):
        self.send_map_array(np.array(baby_map_bonus.map))
        exit_mes = struct.pack('c', b'E')
        self.emitter.send(exit_mes)

    def update_game_time_score(self):
        try:
            message = struct.pack('c', 'G'.encode())
            self.emitter.send(message)

            if self.receiver.getQueueLength() > 0:
                recivedData = self.receiver.getString()
                tup = struct.unpack('c f i', recivedData)
                if tup[0].decode("utf-8") == 'G':
                    self.receiver.nextPacket()
                    self.score = tup[1]
                    self.remained_time = tup[2]
                    if self.initial_time == -1:
                        self.initial_time = self.remained_time
        except:
            pass

    def send_map_array(self, array: np.ndarray):
        s = array.shape
        s_bytes = struct.pack('2i', *s)
        flat_map = ','.join(array.flatten())
        sub_bytes = flat_map.encode('utf-8')
        a_bytes = s_bytes + sub_bytes
        self.emitter.send(a_bytes)
        map_evaluate_request = struct.pack('c', b'M')
        self.emitter.send(map_evaluate_request)

    def go_to_not_seen_tile(self):
        print("in go_to_not_seen_tile section *************************************************************** ")
        best_path = baby_search_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY)

        if len(best_path) < 2:
            self.ai_state = AIStates.random_searching

        p1 = best_path[0]
        p2 = best_path[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if baby_location.robot_in_tile_center() and baby_location.blockChanged:
            baby_location.blockChanged = False

            if baby_location.direction == Direction.up:
                if dx == 1:
                    baby_controller.state = MoveState.turnRight
                if dx == -1:
                    baby_controller.state = MoveState.turnLeft
                if dy == 1:
                    baby_controller.state = MoveState.turnBack
                if dy == -1:
                    baby_controller.state = MoveState.forward

            if baby_location.direction == Direction.down:
                if dx == 1:
                    baby_controller.state = MoveState.turnLeft
                if dx == -1:
                    baby_controller.state = MoveState.turnRight
                if dy == 1:
                    baby_controller.state = MoveState.forward
                if dy == -1:
                    baby_controller.state = MoveState.turnBack

            if baby_location.direction == Direction.left:
                if dx == 1:
                    baby_controller.state = MoveState.turnBack
                if dx == -1:
                    baby_controller.state = MoveState.forward
                if dy == 1:
                    baby_controller.state = MoveState.turnLeft
                if dy == -1:
                    baby_controller.state = MoveState.turnRight

            if baby_location.direction == Direction.right:
                if dx == 1:
                    baby_controller.state = MoveState.forward
                if dx == -1:
                    baby_controller.state = MoveState.turnBack
                if dy == 1:
                    baby_controller.state = MoveState.turnRight
                if dy == -1:
                    baby_controller.state = MoveState.turnLeft


class CameraClass:
    def __init__(self, robot):
        self.colorSensor: Camera = robot.getDevice("cs")
        self.leftCam: Camera = robot.getDevice("camera2")
        self.rightCam: Camera = robot.getDevice("camera1")
        self.leftCam.enable(timeStep)
        self.rightCam.enable(timeStep)
        self.colorSensor.enable(timeStep)
        self.victim_positions = []
        model_path = 'D:\\mojef\\Ferdowsi-Erebus'
        # self.hsu_model: tf.keras.models.Model = self.get_hsu_model()
        # self.hsu_model.load_weights(f"{model_path}\\model_hsu.h5")
        self.model: tf.keras.models.Model = self.get_all_model()
        self.model.load_weights(f"{model_path}\\model.h5")
        # self.cfop_model: tf.keras.models.Model = self.get_cfop_model()
        # self.cfop_model.load_weights(f"{model_path}\\model_cfop.h5")
        self.hsu_type = ['H', 'S', 'U']
        self.cfop_type = ['C', 'F', 'O', 'P']
        self.all_type = ['C', 'F', 'H', 'O', 'P', 'S', 'U']

    def get_hsu_model(self):

        IMG_SIZE = (224, 224)
        IMG_SHAPE = IMG_SIZE + (3,)

        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')

        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        base_model.trainable = False
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        predictions = tf.keras.layers.Dense(3, activation='softmax')(x)

        # this is the model we will train
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model

    def get_cfop_model(self):

        IMG_SIZE = (224, 224)
        IMG_SHAPE = IMG_SIZE + (3,)

        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')

        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

        base_model.trainable = False
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)

        # this is the model we will train
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model

    def color_distance(self, r1, g1, b1, r2, g2, b2):
        return ((r2 - r1) * 0.3) ** 2 + ((g2 - g1) * 0.59) ** 2 + ((b2 - b1) * 0.11) ** 2

    def get_color(self):
        r, g, b = self.colorSensor.getImageArray()[0][0]
        if r < 40 and g < 40 and b < 40:
            return GameColors.black

        elif self.color_distance(r, g, b, 127, 55, 208) < 50 or self.color_distance(r, g, b, 82, 35, 148) < 50:
            return GameColors.purple

        elif self.color_distance(r, g, b, 244, 53, 53) < 50 or self.color_distance(r, g, b, 200, 35, 35) < 50:
            return GameColors.red

        elif self.color_distance(r, g, b, 55, 55, 247) < 50 or self.color_distance(r, g, b, 35, 35, 200) < 50:
            return GameColors.blue

        elif self.color_distance(r, g, b, 190, 156, 88) < 50 or self.color_distance(r, g, b, 131, 103, 56) < 50:
            return GameColors.orange

        elif self.color_distance(r, g, b, 42, 46, 60) < 50 or self.color_distance(r, g, b, 33, 38, 56) < 50:
            return GameColors.silver

        return GameColors.other

    def capture(self):
        img1 = np.array(self.leftCam.getImageArray()).astype(
            np.float32).reshape((40, 64, 3))
        img2 = np.array(self.rightCam.getImageArray()).astype(
            np.float32).reshape((40, 64, 3))
        return img1, img2

    def check_type(self, data):
        back_per = 0
        sky = 0
        for i in range(len(data)):
            color = data[i][0][0]
            value = data[i][0][2]
            per = data[i][1]

            if 80 <= color <= 120:
                back_per += per
            if 105 <= color <= 120:
                sky += per
        if back_per >= .65 or sky >= .20:
            return VictimTypes.wall

        return VictimTypes.sign_or_victim

    def get_color_data(self, sample_image):
        #
        # img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        # twoDimage = img.reshape((-1, 3))
        # twoDimage = np.float32(twoDimage)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 3
        # attempts = 7
        # ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        # center = np.uint8(center)
        # res = center[label.flatten()]
        # result_image = res.reshape((img.shape))
        # center1 = cv2.cvtColor(np.array([center]), cv2.COLOR_BGR2HSV)[0]
        # data = []
        # for i in range(K):
        #     data.append((center1[i], result_image[(result_image == center[i])].size / result_image.size))
        # return data

        # Load the image

        pixel_values = sample_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        k = 3

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        output = centers[labels.flatten()]
        hsv = cv2.cvtColor(np.array([centers]), cv2.COLOR_RGB2HSV)[0]
        data = []
        for i in range(k):
            data.append((hsv[i], output[(output == centers[i])].size / output.size))
        return data

    def check_victim(self):

        img1, img2 = self.capture()
        # save_image(img1)
        # save_image(img2)
        if (baby_location.tilePosX, baby_location.tilePosY) in self.victim_positions:
            return
        data_left = self.get_color_data(img1)
        data_right = self.get_color_data(img2)
        type_left = self.check_type(data_left)
        type_right = self.check_type(data_right)
        # img1 = cv2.resize(img1, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # img2 = cv2.resize(img2, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # img1 = np.array([np.resize(img1, (224, 224, 3))])
        # img2 = np.array([np.resize(img2, (224, 224, 3))])
        img1 = np.array([tf.keras.preprocessing.image.smart_resize(img1, (224, 224), interpolation='bilinear')])
        img2 = np.array([tf.keras.preprocessing.image.smart_resize(img2, (224, 224), interpolation='bilinear')])
        if type_left != VictimTypes.wall and baby_status.s2.getValue() < 0.12:
            if baby_controller.stopCounter == 0:
                baby_controller.dont_move()

            if baby_controller.stopFlag:
                print(data_left)
                save_image(img1[0])
                baby_planner.send_victim(self.all_type[np.argmax(self.model.predict(img1)[0])])
                self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))

        if type_right != VictimTypes.wall and baby_status.s4.getValue() < 0.12:
            if baby_controller.stopCounter == 0:
                baby_controller.dont_move()

            if baby_controller.stopFlag:
                print(data_right)
                save_image(img2[0])
                baby_planner.send_victim(self.all_type[np.argmax(self.model.predict(img2)[0])])
                self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))

        # if type_left == VictimTypes.victim  :
        #
        #     if baby_controller.stopCounter == 0:
        #         baby_controller.dont_move()
        #
        #     if baby_controller.stopFlag:
        #         baby_planner.send_victim(self.hsu_type[np.argmax(self.hsu_model.predict(img1)[0])])
        #         self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))
        #
        # if type_right == VictimTypes.victim and baby_status.s2.getValue() < 0.12:
        #
        #     if baby_controller.stopCounter == 0:
        #         baby_controller.dont_move()
        #
        #     if baby_controller.stopFlag:
        #         baby_planner.send_victim(self.hsu_type[np.argmax(self.hsu_model.predict(img2)[0])])
        #         self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))
        #
        # if type_left == VictimTypes.sign and baby_status.s4.getValue() < 0.12:
        #
        #     if baby_controller.stopCounter == 0:
        #         baby_controller.dont_move()
        #
        #     if baby_controller.stopFlag:
        #         baby_planner.send_victim(self.cfop_type[np.argmax(self.cfop_model.predict(img1)[0])])
        #         self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))
        #
        # if type_right == VictimTypes.sign and baby_status.s2.getValue() < 0.12:
        #
        #     if baby_controller.stopCounter == 0:
        #         baby_controller.dont_move()
        #
        #     if baby_controller.stopFlag:
        #         baby_planner.send_victim(self.cfop_type[np.argmax(self.cfop_model.predict(img2)[0])])
        #         self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))

    def get_all_model(self):
        IMG_SIZE = (224, 224)
        IMG_SHAPE = IMG_SIZE + (3,)

        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')

        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

        base_model.trainable = False
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        predictions = tf.keras.layers.Dense(7, activation='softmax')(x)

        # this is the model we will train
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        return model


timeStep = 32
# define Robot
baby_robot = Robot()
baby_location = LocationClass(baby_robot)
baby_controller = RobotControlClass(baby_robot)
baby_status = StatusClass(baby_robot)
baby_cam = CameraClass(baby_robot)
baby_planner = AIPlannerClass(baby_robot)
baby_map_bonus = MapBonus()
# start simulation
baby_robot.step(timeStep)
baby_location.init_parameters()
baby_finder = ReturnPath(*baby_location.startingTilePos)
baby_search_finder = ReturnPath(0, 0)

while baby_robot.step(timeStep) != -1:
    # try:
    #     print(baby_controller.state)
    baby_planner.plan()

# except:
#     pass
# print(baby_controller.state)
