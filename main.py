import copy
import json
import struct
from enum import Enum
import random
from time import time

import numpy as np
import cv2
from controller import Robot, DistanceSensor, PositionSensor, GPS, Camera, InertialUnit
import tensorflow as tf
from PIL import Image
from numpy import array, argwhere
import pprint

pi = 3.14
ROBOT_RADIUS = .04


def save_image(array: np.ndarray):
    name = f"C:\\Users\\lenovo\\Documents\\Webot\\main{random.randint(0, 1000)}.png"
    im = Image.fromarray(array.astype(np.uint8))
    im.save(name)


def max_min(value, max_val):
    return max(min(value, max_val), -1 * max_val)


# enums
class GameColors(Enum):
    other = 0
    black = 2
    orange = 3
    silver = 4
    start_tile = 5
    blue = 6
    purple = 7
    red = 8
    green = 9


class VictimTypes(Enum):
    wall = 0
    sign_or_victim = 1


class MoveState(Enum):
    forward = 0
    turnLeft = 1
    turnRight = 2
    turnBack = 3
    moveBack = 4
    turnToRightDirection = 5
    turnToLeftDirection = 6
    turnToUpDirection = 7
    turnToDownDirection = 8
    stop = 9


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
    is_black = 3


class TurnState(Enum):
    not_started = 0
    turning = 1


class MapBonus:
    def __init__(self):
        self.MAP_SIZE = baby_location.MAP_SIZE * 2 + 1
        self.map = [['0' for _ in range(2 * self.MAP_SIZE + 1)] for _ in range(2 * self.MAP_SIZE + 1)]

    def update_map(self):
        self.update_walls()
        self.update_floors()

    def update_victim(self, victim_type, side: Direction):
        current_loc = (baby_location.tilePosX, baby_location.tilePosY)

        if side == Direction.left:
            self.set_map_victim(baby_location.NextPosLeft, current_loc, victim_type)
        if side == Direction.right:
            self.set_map_victim(baby_location.NextPosRight, current_loc, victim_type)

    def update_walls(self):
        if baby_controller.state == MoveState.forward:
            current_loc = (baby_location.tilePosX, baby_location.tilePosY)
            if baby_status.left_status == AroundStatus.is_wall:
                self.set_map_wall(baby_location.NextPosLeft, current_loc)
            if baby_status.right_status == AroundStatus.is_wall:
                self.set_map_wall(baby_location.NextPosRight, current_loc)
            if baby_status.front_status == AroundStatus.is_wall:
                self.set_map_wall_forward(baby_location.NextPosForward, current_loc)

    def fill_start_tile(self):
        x, y = baby_location.startingTilePos
        self.set_map_floor(x, y, GameColors.start_tile)

    def update_floors(self):
        self.fill_start_tile()
        color = baby_cam.get_color()
        x = baby_location.light_x_tile
        y = baby_location.light_y_tile
        if baby_location.direction != Direction.not_initialized:
            self.set_map_floor(x, y, color)
            # print(f"color : {color} : {x, y}")

    def set_map_floor(self, tileX, tileY, color: GameColors):
        if color != GameColors.other and (color == GameColors.start_tile or baby_location.light_in_tile_center()):
            # if color != GameColors.start_tile:
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

    def safe_fill_wall(self, x, y):
        if self.map[x][y] == '0':
            self.map[x][y] = '1'

    def set_map_victim(self, next_tile, current_tile, type):
        if baby_location.direction in [Direction.left, Direction.right]:
            y = 2 * (next_tile[1] + current_tile[1] + 1)
            x = 4 * current_tile[0]
            if baby_planner.area_number == 1:
                self.map[x + 1][y] = type
                # self.map[x + 2][y] = type
            if baby_planner.area_number > 1:
                if baby_location.get_sub_area() in [2, 4]:
                    self.map[x + 1][y] = type
                else:
                    self.map[x + 3][y] = type
        if baby_location.direction in [Direction.up, Direction.down]:
            x = 2 * (next_tile[0] + current_tile[0] + 1)
            y = 4 * current_tile[1]
            if baby_planner.area_number == 1:
                self.map[x][y + 1] = type
                # self.map[x][y + 2] = type
            if baby_planner.area_number > 1:
                if baby_location.get_sub_area() in [1, 2]:
                    self.map[x][y + 1] = type
                else:
                    self.map[x][y + 3] = type

    def set_map_wall(self, next_tile, current_tile):
        if baby_location.direction in [Direction.left, Direction.right]:
            y = 2 * (next_tile[1] + current_tile[1] + 1)
            x = 4 * current_tile[0]
            self.safe_fill_wall(x, y)
            self.safe_fill_wall(x + 1, y)
            self.safe_fill_wall(x + 2, y)
            self.safe_fill_wall(x + 3, y)
            self.safe_fill_wall(x + 4, y)
        if baby_location.direction in [Direction.up, Direction.down]:
            x = 2 * (next_tile[0] + current_tile[0] + 1)
            y = 4 * current_tile[1]
            self.safe_fill_wall(x, y)
            self.safe_fill_wall(x, y + 1)
            self.safe_fill_wall(x, y + 2)
            self.safe_fill_wall(x, y + 3)
            self.safe_fill_wall(x, y + 4)

    def set_map_wall_forward(self, next_tile, current_tile):
        if baby_location.direction in [Direction.up, Direction.down]:
            y = 2 * (next_tile[1] + current_tile[1] + 1)
            x = 4 * current_tile[0]
            self.safe_fill_wall(x, y)
            self.safe_fill_wall(x + 1, y)
            self.safe_fill_wall(x + 2, y)
            self.safe_fill_wall(x + 3, y)
            self.safe_fill_wall(x + 4, y)
        if baby_location.direction in [Direction.left, Direction.right]:
            x = 2 * (next_tile[0] + current_tile[0] + 1)
            y = 4 * current_tile[1]
            self.safe_fill_wall(x, y)
            self.safe_fill_wall(x, y + 1)
            self.safe_fill_wall(x, y + 2)
            self.safe_fill_wall(x, y + 3)
            self.safe_fill_wall(x, y + 4)


# Control Classes
class LocationClass:
    def __init__(self, robot: Robot):
        self.MAP_SIZE = 70
        self.gps: GPS = robot.getDevice("gps")  # Retrieve the gps by device name
        self.gps.enable(timeStep)
        self.x = 0
        self.y = 0
        self.tilePosX = 0
        self.tilePosY = 0
        self.offsetX = 0
        self.offsetY = 0
        self.direction = Direction.right
        # self.estimate_direction = Direction.not_initialized
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

    def update_direction_turning(self):
        if self.direction == Direction.up:
            if baby_controller.state == MoveState.turnLeft:
                self.direction = Direction.left
            elif baby_controller.state == MoveState.turnRight:
                self.direction = Direction.right
            elif baby_controller.state == MoveState.turnBack:
                self.direction = Direction.down

        elif self.direction == Direction.left:
            if baby_controller.state == MoveState.turnLeft:
                self.direction = Direction.down
            elif baby_controller.state == MoveState.turnRight:
                self.direction = Direction.up
            elif baby_controller.state == MoveState.turnBack:
                self.direction = Direction.right

        elif self.direction == Direction.right:
            if baby_controller.state == MoveState.turnLeft:
                self.direction = Direction.up
            elif baby_controller.state == MoveState.turnRight:
                self.direction = Direction.down
            elif baby_controller.state == MoveState.turnBack:
                self.direction = Direction.left

        elif self.direction == Direction.down:
            if baby_controller.state == MoveState.turnLeft:
                self.direction = Direction.right
            elif baby_controller.state == MoveState.turnRight:
                self.direction = Direction.left
            elif baby_controller.state == MoveState.turnBack:
                self.direction = Direction.up

    def set_tile_pos(self):
        self.map[2 * self.tilePosX + 1][2 * self.tilePosY + 1] = 1
        if baby_cam.get_color() == GameColors.black:
            if self.direction != Direction.not_initialized:
                self.map[2 * self.light_x_tile + 1][2 * self.light_y_tile + 1] = 2
        if baby_location.direction != Direction.not_initialized and \
                baby_controller.state == MoveState.forward and baby_location.robot_in_tile_center():
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
        pos = self.gps.getValues()
        self.x = pos[0]
        self.y = pos[2]
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
        self.NextPosForward = self.get_next_pos_forward()
        self.NextPosBackward = self.get_next_pos_backward()
        self.NextPosLeft = self.get_next_pos_left()
        self.NextPosRight = self.get_next_pos_right()
        self.get_sub_area()
        if baby_controller.state == MoveState.forward:
            self.add_point()
            if len(self.history) > 6:
                self.history.remove(self.history[0])
        self.lightX, self.lightY = self.get_light_pos()
        self.light_x_tile = int((self.lightX - self.offsetX + 0.06) // 0.12 + self.MAP_SIZE // 2)
        self.light_y_tile = int((self.lightY - self.offsetY + 0.06) // 0.12 + self.MAP_SIZE // 2)
        self.isStuck = self.is_stuck()
        self.increasing_stuck_counter()
        self.set_tile_pos()

        if self.last_tile != (self.tilePosX, self.tilePosY):
            self.blockChanged = True
            self.areaBlockChanged = True
            self.passedBlocksCounter += 1
        # set_game_map(self.lightXPos, self.lightXPos, colorControl.get_color())

    def get_next_pos_forward(self):
        direction = self.direction
        if direction == Direction.right:
            return [self.tilePosX + 1, self.tilePosY]
        elif direction == Direction.left:
            return [self.tilePosX - 1, self.tilePosY]
        elif direction == Direction.up:
            return [self.tilePosX, self.tilePosY - 1]
        elif direction == Direction.down:
            return [self.tilePosX, self.tilePosY + 1]

    def get_next_pos_backward(self):
        direction = self.direction
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
        direction = self.direction
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
        direction = self.direction
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
        direction = self.direction
        lightOffsetX = 0 if direction in (
            Direction.up,
            Direction.down) else -1.5 * ROBOT_RADIUS if direction == Direction.left else 1.5 * ROBOT_RADIUS
        lightOffsetY = 0 if direction in (
            Direction.left,
            Direction.right) else -1.5 * ROBOT_RADIUS if direction == Direction.up else 1.5 * ROBOT_RADIUS
        return self.x + lightOffsetX, self.y + lightOffsetY

    def is_stuck(self):
        if len(self.history) == 0:
            return False
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
        if pos is None:
            return False
        return self.map[2 * pos[0] + 1][2 * pos[1] + 1] == 1

    def is_tile_black(self, pos):
        if pos is None:
            return False
        return self.map[2 * pos[0] + 1][2 * pos[1] + 1] == 2

    def robot_in_big_tile_center(self):
        thr = .02
        condition_area1 = (0.06 - thr <= (self.y - self.offsetY + 0.06) % 0.12 < .06 + thr and self.direction in [
            Direction.up, Direction.down]) or \
                          (.06 - thr <= (self.x - self.offsetX + 0.06) % 0.12 < .06 + thr and self.direction in [
                              Direction.left, Direction.right])

        condition_area2 = (0.025 <= (self.y - self.offsetY + 0.03) % 0.06 < 0.035 and self.direction in [
            Direction.up, Direction.down]) or \
                          (0.025 <= (self.x - self.offsetX + 0.03) % 0.06 < 0.035 and self.direction in [
                              Direction.left, Direction.right])

        # if baby_planner.area_number == 1:
        return condition_area1

    def robot_in_tile_center(self):
        thr = .008
        condition_area1 = (0.06 - thr <= (self.y - self.offsetY + 0.06) % 0.12 < .06 + thr and self.direction in [
            Direction.up, Direction.down]) or \
                          (.06 - thr <= (self.x - self.offsetX + 0.06) % 0.12 < .06 + thr and self.direction in [
                              Direction.left, Direction.right])

        condition_area2 = (0.025 <= (self.y - self.offsetY + 0.03) % 0.06 < 0.035 and self.direction in [
            Direction.up, Direction.down]) or \
                          (0.025 <= (self.x - self.offsetX + 0.03) % 0.06 < 0.035 and self.direction in [
                              Direction.left, Direction.right])

        # if baby_planner.area_number == 1:
        return condition_area1

        # else:
        #     return condition_area2

    def light_in_tile_center(self):
        return (0.055 <= (self.lightY - self.offsetY + 0.06) % 0.12 < 0.065 and self.direction in [
            Direction.up, Direction.down]) or \
               (0.055 <= (self.lightX - self.offsetX + 0.06) % 0.12 < 0.065 and self.direction in [
                   Direction.left, Direction.right])


class RobotControlClass:
    def __init__(self, robot: Robot):
        self.full_turn_angle = 4.42
        self.state = MoveState.turnToRightDirection
        self.max_velocity = 6.24
        self.last_state = MoveState.forward
        self.turn_state = TurnState.not_started
        self.left_wheel = robot.getDevice("wheel1 motor")
        self.right_wheel = robot.getDevice("wheel2 motor")
        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))
        self.rightWheelPosSensor: PositionSensor = self.right_wheel.getPositionSensor()
        self.leftWheelPosSensor: PositionSensor = self.left_wheel.getPositionSensor()
        self.rightWheelPosSensor.enable(timeStep)
        self.leftWheelPosSensor.enable(timeStep)
        self.iu: InertialUnit = robot.getDevice("iu")
        self.iu.enable(timeStep)
        self.stopCounter = 0
        self.leftWheelPos = 0
        self.rightWheelPos = 0
        self.stopFlag = False
        self.leftWheelSpeed = 0
        self.rightWheelSpeed = 0
        self.left_wall_counter = 0
        self.right_wall_counter = 0
        self.forward_counter = 0

    def run(self):
        if self.state != MoveState.forward:
            self.forward_counter = 0
        if self.state == MoveState.forward:
            self.forward_counter += 1
            self.move_forward()
        elif self.state == MoveState.turnLeft:
            self.turn_left()
        elif self.state == MoveState.turnRight:
            self.turn_right()
        elif self.state == MoveState.turnBack:
            self.turn_back()
        elif self.state == MoveState.moveBack:
            self.move_back()
        elif self.state == MoveState.turnToDownDirection:
            self.turn_to_down()
        elif self.state == MoveState.turnToUpDirection:
            self.turn_to_up()
        elif self.state == MoveState.turnToLeftDirection:
            self.turn_to_left()
        elif self.state == MoveState.turnToRightDirection:
            self.turn_to_right()
        elif self.state == MoveState.stop:
            self.stop()

    def move(self):
        self.state = MoveState.forward

    def dont_move(self):
        self.last_state = self.state
        self.state = MoveState.stop
        self.stopFlag = False

    def move_forward(self):
        if baby_status.s6.getValue() < .1 and baby_status.s5.getValue() > .25 :
            baby_controller.state = MoveState.turnLeft
            return
        if baby_status.s5.getValue() < .1 and baby_status.s6.getValue() > .25:
            baby_controller.state = MoveState.turnRight
            return

        e = 0
        if baby_location.direction == Direction.left:
            e = self.get_left_error(4)
        elif baby_location.direction == Direction.right:
            e = self.get_right_error(4)
        elif baby_location.direction == Direction.up:
            e = self.get_up_error(4)
        elif baby_location.direction == Direction.down:
            e = self.get_down_error(4)

        dist = .055
        e_near_wall = 0
        if baby_status.s2.getValue() < baby_status.s4.getValue():
            # print(f"left: {baby_status.s2.getValue()}")
            if baby_status.s2.getValue() < 1.8 * dist:
                self.right_wall_counter += 1
                e_near_wall = (baby_status.s2.getValue() - dist)
                if e_near_wall < 0:  # too near to wall
                    e_near_wall *= 25
                else:
                    e_near_wall *= 45
            else:
                self.right_wall_counter = 0
        else:
            if baby_status.s4.getValue() < 1.8 * dist:
                # print(f"right: {baby_status.s4.getValue()}")
                self.left_wall_counter += 1
                e_near_wall = - (baby_status.s4.getValue() - dist)
                if e_near_wall < 0:  # too near to wall
                    e_near_wall *= 25
                else:
                    e_near_wall *= 45
            else:
                self.left_wall_counter = 0
        if self.left_wall_counter < 10 and self.right_wall_counter < 10:
            e_near_wall = 0
        if baby_controller.forward_counter < 10 *  32 / timeStep:
            e = 0
        elif abs(e) > .3 * 4 and baby_status.front_status != AroundStatus.is_wall \
                and baby_cam.get_color() != GameColors.black and baby_controller.forward_counter > 40 * 32 / timeStep:
            if baby_location.direction == Direction.left:
                self.state = MoveState.turnToLeftDirection
            if baby_location.direction == Direction.right:
                self.state = MoveState.turnToRightDirection
            if baby_location.direction == Direction.up:
                self.state = MoveState.turnToUpDirection
            if baby_location.direction == Direction.down:
                self.state = MoveState.turnToDownDirection
            return

        # print(e_near_wall)
        # print(f"e: {e}, dir: {baby_location.direction}  , {self.iu.getRollPitchYaw()[2]} ")
        self.leftWheelSpeed = self.max_velocity - e - e_near_wall
        self.rightWheelSpeed = self.max_velocity + e + e_near_wall
        self.normalize_wheel_speed()
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)

    def turn_left(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            target = self.leftWheelPos + self.full_turn_angle / 2
            pid_coef = target - self.leftWheelPosSensor.getValue()
            self.leftWheelSpeed = max_min(pid_coef * 5, 3)
            self.rightWheelSpeed = - max_min(pid_coef * 5, 3)
            self.normalize_wheel_speed()
            self.left_wheel.setVelocity(self.leftWheelSpeed)
            self.right_wheel.setVelocity(self.rightWheelSpeed)
            if target - .01 < self.leftWheelPosSensor.getValue() < target + .01:
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def turn_right(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            target = self.leftWheelPos - self.full_turn_angle / 2
            pid_coef = target - self.leftWheelPosSensor.getValue()
            self.leftWheelSpeed = max_min(pid_coef * 5, 3)
            self.rightWheelSpeed = - max_min(pid_coef * 5, 3)
            self.normalize_wheel_speed()
            self.left_wheel.setVelocity(self.leftWheelSpeed)
            self.right_wheel.setVelocity(self.rightWheelSpeed)
            if target - .01 < self.leftWheelPosSensor.getValue() < target + .01:
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def turn_back(self):
        baby_location.history.clear()
        if self.turn_state == TurnState.not_started:
            self.turn_state = TurnState.turning
            self.leftWheelPos = self.leftWheelPosSensor.getValue()
            self.rightWheelPos = self.rightWheelPosSensor.getValue()
        else:
            target = self.leftWheelPos + self.full_turn_angle
            pid_coef = target - self.leftWheelPosSensor.getValue()
            self.leftWheelSpeed = max_min(pid_coef * 4, 3)
            self.rightWheelSpeed = - max_min(pid_coef * 4, 3)
            self.normalize_wheel_speed()
            self.left_wheel.setVelocity(self.leftWheelSpeed)
            self.right_wheel.setVelocity(self.rightWheelSpeed)
            if target - .01 < self.leftWheelPosSensor.getValue() < target + .01:
                self.turn_state = TurnState.not_started
                self.state = MoveState.forward

    def move_back(self):
        baby_location.history.clear()
        leftWheelSpeed = -self.max_velocity
        rightWheelSpeed = -self.max_velocity
        self.left_wheel.setVelocity(leftWheelSpeed)
        self.right_wheel.setVelocity(rightWheelSpeed)

    def stop(self):
        baby_location.history.clear()
        self.stopCounter += 1

        if self.stopCounter >= (100 - 10) * 32 / timeStep:
            self.stopFlag = True

        if self.stopCounter >= 100 * 32 / timeStep:
            self.stopCounter = 0
            self.state = self.last_state

        self.left_wheel.setVelocity(0)
        self.right_wheel.setVelocity(0)

    def turn_to_right(self):
        baby_location.direction = Direction.right
        baby_location.history.clear()
        e = self.get_right_error() * 2
        self.leftWheelSpeed = max_min(- e, self.max_velocity)
        self.rightWheelSpeed = max_min(+ e, self.max_velocity)
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)
        if abs(e) < .05:
            self.state = MoveState.forward

    def turn_to_up(self):
        baby_location.direction = Direction.up
        baby_location.history.clear()
        e = self.get_up_error() * 2
        self.leftWheelSpeed = max_min(- e, self.max_velocity)
        self.rightWheelSpeed = max_min(+ e, self.max_velocity)
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)
        if abs(e) < .05:
            self.state = MoveState.forward

    def turn_to_left(self):
        baby_location.direction = Direction.left
        baby_location.history.clear()
        e = self.get_left_error() * 2
        self.leftWheelSpeed = max_min(- e, self.max_velocity)
        self.rightWheelSpeed = max_min(+ e, self.max_velocity)
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)
        if abs(e) < .05:
            self.state = MoveState.forward

    def turn_to_down(self):
        baby_location.direction = Direction.down
        baby_location.history.clear()
        e = self.get_down_error() * 2
        self.leftWheelSpeed = max_min(- e, self.max_velocity)
        self.rightWheelSpeed = max_min(+ e, self.max_velocity)
        self.left_wheel.setVelocity(self.leftWheelSpeed)
        self.right_wheel.setVelocity(self.rightWheelSpeed)
        if abs(e) < .05:
            self.state = MoveState.forward

    def get_right_error(self, coef=2):
        angle = self.iu.getRollPitchYaw()[2]
        diff = angle - (-pi / 2)
        e = diff
        if e > pi:
            e = e - 2 * pi
        e = max_min(e * coef, self.max_velocity)
        return e

    def get_down_error(self, coef=2):
        angle = self.iu.getRollPitchYaw()[2]
        if angle > 0:
            e = angle - pi
        else:
            e = angle + pi
        e = max_min(e * coef, self.max_velocity)
        return e

    def get_left_error(self, coef=2):
        angle = self.iu.getRollPitchYaw()[2]
        diff = angle - pi / 2
        e = diff
        if e < - pi:
            e = e + 2 * pi
        e = max_min(e * coef, self.max_velocity)
        return e

    def get_up_error(self, coef=2):
        angle = self.iu.getRollPitchYaw()[2]
        diff = angle - 0
        e = diff
        e = max_min(e * coef, self.max_velocity)
        return e

    def normalize_wheel_speed(self):
        if abs(self.leftWheelSpeed) > self.max_velocity or abs(self.rightWheelSpeed) > self.max_velocity:
            max_wheel = max(abs(self.leftWheelSpeed), abs(self.rightWheelSpeed))
            self.leftWheelSpeed *= self.max_velocity / max_wheel
            self.rightWheelSpeed *= self.max_velocity / max_wheel


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
        wall_dist = .09
        if self.s1.getValue() > wall_dist:
            if baby_location.is_tile_seen(baby_location.NextPosForward):
                self.front_status = AroundStatus.is_seen
            elif baby_location.is_tile_black(baby_location.NextPosForward):
                self.front_status = AroundStatus.is_black
            else:
                self.front_status = AroundStatus.is_empty
        else:
            self.front_status = AroundStatus.is_wall

        if self.s1.getValue() > .3 and self.s5.getValue() < wall_dist and self.s6.getValue() < wall_dist:
            self.front_status = AroundStatus.is_wall

        if self.s1.getValue() > .3 and self.s5.getValue() > .3 and self.s6.getValue() < wall_dist:
            self.front_status = AroundStatus.is_wall

        if self.s1.getValue() > .3 and self.s6.getValue() > .3 and self.s5.getValue() < wall_dist:
            self.front_status = AroundStatus.is_wall

        if self.s2.getValue() > wall_dist + .02:
            if baby_location.is_tile_seen(baby_location.NextPosRight):
                self.right_status = AroundStatus.is_seen
            elif baby_location.is_tile_black(baby_location.NextPosRight):
                self.right_status = AroundStatus.is_black
            else:
                self.right_status = AroundStatus.is_empty
        else:
            self.right_status = AroundStatus.is_wall

        if self.s4.getValue() > wall_dist + .02:
            if baby_location.is_tile_seen(baby_location.NextPosLeft):
                self.left_status = AroundStatus.is_seen
            elif baby_location.is_tile_black(baby_location.NextPosLeft):
                self.left_status = AroundStatus.is_black
            else:
                self.left_status = AroundStatus.is_empty
        else:
            self.left_status = AroundStatus.is_wall

        if self.s3.getValue() > wall_dist:
            if baby_location.is_tile_seen(baby_location.NextPosBackward):
                self.behind_status = AroundStatus.is_seen
            elif baby_location.is_tile_black(baby_location.NextPosBackward):
                self.behind_status = AroundStatus.is_black
            else:
                self.behind_status = AroundStatus.is_empty
        else:
            self.behind_status = AroundStatus.is_wall


class ReturnPath:
    def __init__(self, x, y):  # Saving start tile positions
        self.list_hazfi = []
        self.all_path = []
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

        if 2 * x + 3 < self.map_size and (baby_location.map[2 * x + 1 + 2][2 * y + 1] == 1 or (
                x + 1 == self.startx and y == self.starty)) and \
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
        self.find_path = []
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
        self.start_not_seen_searching = True

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
        # print(self.remained_time / self.initial_time)
        if self.ai_state == AIStates.not_seen_searching:
            return
        if self.initial_time == -1:
            self.ai_state = AIStates.random_searching
            return
        if self.remained_time / self.initial_time > .2:
            self.ai_state = AIStates.random_searching
        # elif self.remained_time / self.initial_time > .3:
        #     self.ai_state = AIStates.wall_following
        else:
            self.ai_state = AIStates.returning
            baby_controller.state = MoveState.stop
            baby_controller.run()
            baby_robot.step(timeStep)
            self.find_path = baby_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY)
            self.start_not_seen_searching = False
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
                # print(baby_location.stuckCounter)
                self.go_to_not_seen_tile()
        # baby_controller.dont_move()
        # print(f"current is {baby_location.tilePosX, baby_location.tilePosY}")

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
            # print("is stuck")
            ## side
            if baby_status.s3.getValue() < 0.2 or baby_status.s5.getValue() < 0.2 and baby_controller.forward_counter > 40 * 32 / timeStep:
                baby_controller.state = MoveState.turnRight

            elif baby_status.s1.getValue() < 0.2 or baby_status.s6.getValue() < 0.2 and baby_controller.forward_counter > 40 * 32 / timeStep:
                baby_controller.state = MoveState.turnLeft

            elif baby_status.s2.getValue() < 0.1 and baby_status.s4.getValue() < 0.1:
                baby_controller.state = MoveState.turnBack
            baby_location.update_direction_turning()

            return

        if baby_status.s1.getValue() <= .065 or (
                baby_location.robot_in_tile_center() and baby_location.blockChanged and
                baby_cam.get_color() != GameColors.blue and baby_cam.get_color() != GameColors.purple and
                baby_cam.get_color() != GameColors.red and baby_cam.get_color() != GameColors.green and \
                baby_status.front_status != AroundStatus.is_wall) \
                or baby_cam.get_color() == GameColors.black:
            # print(
            # f"color: {baby_cam.get_color()} is center: {baby_location.robot_in_tile_center()} bloc changed:{baby_location.blockChanged}")
            allChoice = []
            emptyChoice = []
            baby_location.blockChanged = False
            baby_location.stuckCounter = 0

            if baby_status.front_status in (
                    AroundStatus.is_empty, AroundStatus.is_seen) and baby_cam.get_color() != GameColors.black:
                allChoice.append(MoveState.forward)

            if baby_status.front_status == AroundStatus.is_empty and baby_cam.get_color() != GameColors.black:
                emptyChoice.append(MoveState.forward)

            if baby_status.left_status in (AroundStatus.is_empty, AroundStatus.is_seen):
                allChoice.append(MoveState.turnLeft)

            if baby_status.left_status == AroundStatus.is_empty:
                emptyChoice.append(MoveState.turnLeft)

            if baby_status.right_status in (AroundStatus.is_empty, AroundStatus.is_seen):
                allChoice.append(MoveState.turnRight)

            if baby_status.right_status == AroundStatus.is_empty:
                emptyChoice.append(MoveState.turnRight)

            if baby_status.behind_status in (AroundStatus.is_empty, AroundStatus.is_seen):
                allChoice.append(MoveState.turnBack)

            if baby_status.left_status == AroundStatus.is_seen and baby_status.right_status == AroundStatus.is_seen \
                    and baby_status.front_status in (AroundStatus.is_empty, AroundStatus.is_seen):
                baby_controller.state = MoveState.forward

            if baby_status.left_status == AroundStatus.is_seen and baby_status.right_status == AroundStatus.is_seen \
                    and baby_status.front_status == AroundStatus.is_wall:
                baby_controller.state = MoveState.turnBack

            if len(emptyChoice) > 0:
                if MoveState.forward in emptyChoice:
                    baby_controller.state = MoveState.forward
                else:
                    baby_controller.state = random.choice(emptyChoice)

                # print(f"Random search robot choose from emptyChoices: {baby_controller.state}")
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
                            # print("is back")
                            self.not_seen_tiles.add(tuple(baby_location.NextPosBackward))


            elif len(self.not_seen_tiles) > 0 and self.area_number in [1, 2]:
                best_path = None
                for i in range(len(self.not_seen_tiles)):
                    baby_controller.state = MoveState.stop
                    baby_controller.run()
                    baby_robot.step(timeStep)
                    target_tile = list(self.not_seen_tiles)[i]
                    baby_search_finder = ReturnPath(*target_tile)
                    return_path = baby_search_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY)
                    if len(return_path) >= 2 and best_path is None:
                        best_path = return_path
                    if best_path is not None:
                        if len(best_path) > len(return_path) >= 2:
                            best_path = return_path
                if best_path is not None:
                    if len(best_path) >= 2:
                        # `test`
                        # np.savetxt("D:\\ali.csv", baby_location.map, delimiter=",", fmt='%s')
                        self.ai_state = AIStates.not_seen_searching
                        baby_planner.start_not_seen_searching = True
                        baby_location.blockChanged = True
                        # print(best_path)
                        # print("go to not seen section")
                        self.find_path = best_path
                        return

                if len(allChoice) > 0:
                    if len(allChoice) > 1 and MoveState.turnBack in allChoice:
                        allChoice.remove(MoveState.turnBack)

                    baby_controller.state = random.choice(allChoice)
                    print(f"Random search robot choose from allChoice: {baby_controller.state}")
                else:
                    print("Bug go to forward")
                    baby_controller.state = MoveState.forward
            elif len(allChoice) > 0:
                if len(allChoice) > 1 and MoveState.turnBack in allChoice:
                    allChoice.remove(MoveState.turnBack)
                baby_controller.state = random.choice(allChoice)
                print(f"Random search robot choose from allChoice: {baby_controller.state}")
            else:
                print("Bug go to forward")
                baby_controller.state = MoveState.forward
            baby_location.update_direction_turning()

    def wall_follow(self):
        pass

    def return_start_tile(self):
        print(f"returning: {self.find_path}")
        self.go_to_not_seen_tile()
        if baby_location.tilePosX == baby_location.startingTilePos[0] and baby_location.tilePosY == \
                baby_location.startingTilePos[1]:
            self.send_finish()

    def send_victim(self, typ):
        x, y = baby_location.x * 100, baby_location.y * 100
        victim_type = bytes(typ, "utf-8")  # The victim type being sent is the victimChar for harmed victim
        message = struct.pack("i i c", int(x), int(y), victim_type)  # Pack message
        self.emitter.send(message)  # Send out the message

    def send_finish(self):

        main_array = np.flipud(np.array(baby_map_bonus.map))
        # np.savetxt("D:\\ali.csv", main_array, delimiter=",", fmt='%s')
        self.send_map_array(main_array)
        exit_mes = struct.pack('c', b'E')
        self.emitter.send(exit_mes)

    def update_game_time_score(self):
        try:
            message = struct.pack('c', 'G'.encode())
            self.emitter.send(message)

            if self.receiver.getQueueLength() > 0:
                recivedData = self.receiver.getBytes()
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
        if baby_location.tilePosX == baby_search_finder.startx and baby_location.tilePosY == baby_search_finder.starty:
            self.ai_state = AIStates.random_searching
            return

        if len(self.find_path) < 2:
            self.ai_state = AIStates.random_searching
            return

        if (baby_location.robot_in_tile_center() and baby_location.blockChanged and \
            baby_location.direction != Direction.not_initialized) or self.start_not_seen_searching:
            baby_location.blockChanged = False
            self.start_not_seen_searching = False
            # self.find_path = baby_search_finder.get_best_path(baby_location.tilePosX, baby_location.tilePosY)
            # print(self.find_path)
            p1 = self.find_path[0]
            p2 = self.find_path[1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # print(
            #     f"direction is: {baby_location.direction},target is :{baby_search_finder.startx, baby_search_finder.starty},best_path:{self.find_path}")
            if dx == 1:
                baby_controller.state = MoveState.turnToRightDirection
            elif dx == -1:
                baby_controller.state = MoveState.turnToLeftDirection
            elif dy == 1:
                baby_controller.state = MoveState.turnToDownDirection
            elif dy == -1:
                baby_controller.state = MoveState.turnToUpDirection
            else:
                # print("wrong path")
                self.ai_state = AIStates.random_searching
                baby_controller.state = MoveState.forward
                return
            if len(self.find_path) > 0:
                del self.find_path[0]

            if (
                    baby_cam.get_color() == GameColors.black and baby_controller.state == MoveState.forward) or \
                    baby_location.stuckCounter > 40 or (
                    baby_status.s1.getValue() < .75 and baby_controller.state == MoveState.forward):
                self.ai_state = AIStates.random_searching
                if self.find_path[-1] in self.not_seen_tiles:
                    # print("delete old path")
                    self.not_seen_tiles.remove(self.find_path[-1])


class CameraClass:
    def __init__(self, robot):
        self.colorSensor: Camera = robot.getDevice("cs")
        self.leftCam: Camera = robot.getDevice("camera2")
        self.rightCam: Camera = robot.getDevice("camera1")
        self.leftCam.enable(timeStep)
        self.rightCam.enable(timeStep)
        self.colorSensor.enable(timeStep)
        self.victim_positions = []
        model_path = 'D:\\Ferdowsi-Erebus\\'
        self.model: tf.keras.models.Model = self.get_all_model()
        self.model.load_weights(f"{model_path}model.h5")
        self.hsu_type = ['H', 'S', 'U']
        self.cfop_type = ['C', 'F', 'O', 'P']
        self.all_type = ['C', 'F', 'H', 'O', 'P', 'S', 'U']

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

        elif self.color_distance(r, g, b, 29, 241, 29) < 50 or self.color_distance(r, g, b, 18, 189, 18) < 50:
            return GameColors.green

        return GameColors.other

    def capture(self):
        img1 = np.array(self.leftCam.getImageArray()).astype(
            np.float32).reshape((40, 64, 3))
        img2 = np.array(self.rightCam.getImageArray()).astype(
            np.float32).reshape((40, 64, 3))
        return img1, img2

    def check_type(self, data):
        wall_per = 0
        sky = 0
        obs = 0
        for i in range(len(data)):
            color = data[i][0][0]
            value = data[i][0][2]
            per = data[i][1]

            if 80 <= color <= 125:
                wall_per += per
            if 105 <= color <= 125:
                sky += per
            if 0 <= color <= 20 and (75 <= value <= 150 or 0 <= value <= 35):  # khakestari va black
                obs += per
        # print(wall_per, sky)
        if (wall_per >= .66) or (sky >= .05) or (obs >= 0.45):
            return VictimTypes.wall

        return VictimTypes.sign_or_victim

    def get_color_data(self, sample_image):
        pixel_values = sample_image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        k = 5

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 1.0)

        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        output = centers[labels.flatten()]
        hsv = cv2.cvtColor(np.array([centers]), cv2.COLOR_RGB2HSV)[0]
        data = []
        for i in range(k):
            data.append((hsv[i], output[(output == centers[i])].size / output.size))
        return data

    def check_victim(self):
        img1, img2 = self.capture()
        if (baby_location.tilePosX, baby_location.tilePosY) in self.victim_positions:
            return
        data_left = self.get_color_data(img1)
        data_right = self.get_color_data(img2)
        if baby_status.s4.getValue() < 0.15:
            type_left = self.check_type(data_left)
            if type_left != VictimTypes.wall:
                img1 = np.array([tf.keras.preprocessing.image.smart_resize(img1, (224, 224), interpolation='bilinear')])
                # inja bayad s4 bashe
                if baby_controller.stopCounter == 0:
                    baby_controller.dont_move()

                if baby_controller.stopFlag:
                    # print(data_left)
                    # save_image(img1[0])
                    vtype = self.all_type[np.argmax(self.model.predict(img1)[0])]
                    baby_planner.send_victim(vtype)
                    self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))
                    baby_map_bonus.update_victim(vtype, Direction.left)

        if baby_status.s2.getValue() < 0.15:
            type_right = self.check_type(data_right)
            if type_right != VictimTypes.wall:
                img2 = np.array([tf.keras.preprocessing.image.smart_resize(img2, (224, 224), interpolation='bilinear')])
                # inja bayad s2 bashe
                if baby_controller.stopCounter == 0:
                    baby_controller.dont_move()

                if baby_controller.stopFlag:
                    # print(data_right)
                    # save_image(img2[0])
                    vtype = self.all_type[np.argmax(self.model.predict(img2)[0])]
                    baby_planner.send_victim(vtype)
                    self.victim_positions.append((baby_location.tilePosX, baby_location.tilePosY))
                    baby_map_bonus.update_victim(vtype, Direction.right)

    def get_all_model(self):
        IMG_SIZE = (224, 224)
        IMG_SHAPE = IMG_SIZE + (3,)

        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights=None)

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


timeStep = 16
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
baby_robot.step(timeStep)
baby_robot.step(timeStep)
baby_location.init_parameters()
baby_finder = ReturnPath(*baby_location.startingTilePos)
baby_search_finder = ReturnPath(0, 0)

while baby_robot.step(timeStep) != -1:
    try:
        #     print(baby_controller.state)
        baby_planner.plan()

    # print(
    #     f"Forward status : {baby_status.front_status}, "
    #     f"Left: {baby_status.left_status},\n"
    #     f"right: {baby_status.right_status},\n "
    #     f"back: {baby_status.behind_status},\n "
    #     f"Direction: {baby_location.direction}\n"
    #     f"ai state : {baby_planner.ai_state}\n"
    #     f"robot state : {baby_controller.state}\n")
    # print(baby_location.robot_in_tile_center())

    except:
        pass
# print(baby_controller.state)
