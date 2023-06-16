# Adapted from https://github.com/bunelr/GandRL_for_NPS/blob/master/karel/world.py

import copy
from typing import Union
import numpy as np

from prog_policies.base import BaseEnvironment

MAX_API_CALLS = 10000
MAX_MARKERS_PER_SQUARE = 10

STATE_TABLE = {
    0: 'Karel facing North',
    1: 'Karel facing East',
    2: 'Karel facing South',
    3: 'Karel facing West',
    4: 'Wall',
    5: '0 marker',
    6: '1 marker',
    7: '2 markers',
    8: '3 markers',
    9: '4 markers',
    10: '5 markers',
    11: '6 markers',
    12: '7 markers',
    13: '8 markers',
    14: '9 markers',
    15: '10 markers'
}

class KarelEnvironment(BaseEnvironment):

    def __init__(self, env_height=8, env_width=8, crashable=True, leaps_behaviour=False,
                 max_calls=MAX_API_CALLS, initial_state: Union[np.ndarray, None] = None):
        self.crashable = crashable
        self.leaps_behaviour = leaps_behaviour
        actions = {
            "move": self.move,
            "turnLeft": self.turn_left,
            "turnRight": self.turn_right,
            "pickMarker": self.pick_marker,
            "putMarker": self.put_marker
        }
        bool_features = {
            "frontIsClear": self.front_is_clear,
            "leftIsClear": self.left_is_clear,
            "rightIsClear": self.right_is_clear,
            "markersPresent": self.markers_present,
            "noMarkersPresent": self.no_markers_present
        }
        int_features = {}
        if initial_state is not None:
            state_shape = initial_state.shape
        else:
            state_shape = (len(STATE_TABLE), env_height, env_width)
        super().__init__(actions, bool_features, int_features, state_shape, initial_state,
                         max_calls=max_calls)
        
    def default_state(self):
        state = np.zeros(self.state_shape, dtype=bool)
        state[4, :, :] = True
        state[0, 0, 0] = True
        return state

    def set_state(self, state: np.ndarray):
        self.state = copy.deepcopy(state)
        d, r, c = np.where(self.state[:4, :, :] > 0)
        self.hero_pos = [r[0], c[0], d[0]]
        self.markers_grid = self.state[5:, :, :].argmax(axis=0)        
        
    @classmethod
    def from_string(cls, state_str: str):
        lines = state_str.replace('|', '').split('\n')
        rows = len(lines)
        cols = len(lines[0])
        state = np.zeros((len(STATE_TABLE), rows, cols), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if lines[r][c] == '*':
                    state[4][r][c] = True
                elif lines[r][c] >= '1' and lines[r][c] <= '9':
                    state[5 + int(lines[r][c])][r][c] = True
                elif lines[r][c] == 'M':
                    state[15][r][c] = True
                else:
                    state[5][r][c] = True
                if lines[r][c] == '^':
                    state[0][r][c] = True
                elif lines[r][c] == '>':
                    state[1][r][c] = True
                elif lines[r][c] == 'v':
                    state[2][r][c] = True
                elif lines[r][c] == '<':
                    state[3][r][c] = True
        return cls(initial_state=state)
    
    def to_string(self) -> str:
        worldStr = ''
        if self.crashed: worldStr += 'CRASHED\n'
        hero_r, hero_c, hero_d = self.get_hero_pos()
        _, rows, cols = self.state_shape
        for r in range(rows):
            rowStr = '|'
            for c in range(cols):
                if self.state[4, r, c] == True:
                    rowStr += '*'
                elif r == hero_r and c == hero_c:
                    rowStr += self.get_hero_char(hero_d)
                elif self.markers_grid[r, c] > 0:
                    num_marker = self.markers_grid[r, c]
                    if num_marker > 9: rowStr += 'M'
                    else: rowStr += str(num_marker)
                else:
                    rowStr += ' '
            worldStr += rowStr + '|'
            if(r != rows-1): worldStr += '\n'
        return worldStr
    
    # TODO
    def to_image(self) -> np.ndarray:
        pass
    
    def get_state(self):
        return self.state
    
    def get_hero_pos(self):
        return self.hero_pos
    
    def get_markers_grid(self):
        return self.markers_grid
    
    def get_hero_char(self, dir) -> str:
        if(dir == 0): return '^'
        if(dir == 1): return '>'
        if(dir == 2): return 'v'
        if(dir == 3): return '<'
        raise("invalid dir")
    
    def hero_at_pos(self, r: int, c: int) -> bool:
        row, col, _ = self.hero_pos
        return row == r and col == c
    
    def is_clear(self, r: int, c: int) -> bool:
        if r < 0 or c < 0:
            return False
        if r >= self.state_shape[1] or c >= self.state_shape[2]:
            return False
        return not self.state[4, r, c]

    def front_is_clear(self) -> bool:
        row, col, d = self.hero_pos
        if d == 0:
            return self.is_clear(row - 1, col)
        elif d == 1:
            return self.is_clear(row, col + 1)
        elif d == 2:
            return self.is_clear(row + 1, col)
        elif d == 3:
            return self.is_clear(row, col - 1)
        
    def left_is_clear(self) -> bool:
        row, col, d = self.hero_pos
        if d == 0:
            return self.is_clear(row, col - 1)
        elif d == 1:
            return self.is_clear(row - 1, col)
        elif d == 2:
            return self.is_clear(row, col + 1)
        elif d == 3:
            return self.is_clear(row + 1, col)
        
    def right_is_clear(self) -> bool:
        row, col, d = self.hero_pos
        if d == 0:
            return self.is_clear(row, col + 1)
        elif d == 1:
            return self.is_clear(row + 1, col)
        elif d == 2:
            return self.is_clear(row, col - 1)
        elif d == 3:
            return self.is_clear(row - 1, col)
        
    def markers_present(self) -> bool:
        row, col, _ = self.hero_pos
        return self.markers_grid[row, col] > 0
    
    def no_markers_present(self) -> bool:
        row, col, _ = self.hero_pos
        return self.markers_grid[row, col] == 0
    
    def move(self):
        r, c, d = self.hero_pos
        new_r = r
        new_c = c
        if(d == 0): new_r = new_r - 1
        if(d == 1): new_c = new_c + 1
        if(d == 2): new_r = new_r + 1
        if(d == 3): new_c = new_c - 1
        if not self.is_clear(new_r, new_c) and self.crashable:
            self.crashed = True
        if not self.crashed and self.is_clear(new_r, new_c):
            self.state[d, r, c] = False
            self.state[d, new_r, new_c] = True
            self.hero_pos = [new_r, new_c, d]
        elif self.leaps_behaviour:
            self.turn_left()
            self.turn_left()
            
    def turn_left(self) -> None:
        r, c, d = self.hero_pos
        new_d = (d - 1) % 4
        self.state[d, r, c] = False
        self.state[new_d, r, c] = True
        self.hero_pos = [r, c, new_d]
    
    def turn_right(self) -> None:
        r, c, d = self.hero_pos
        new_d = (d + 1) % 4
        self.state[d, r, c] = False
        self.state[new_d, r, c] = True
        self.hero_pos = [r, c, new_d]
        
    def pick_marker(self) -> None:
        r, c, _ = self.hero_pos
        num_marker = self.markers_grid[r, c]
        if num_marker == 0:
            if self.crashable:
                self.crashed = True
        else:
            self.state[5 + num_marker, r, c] = False
            self.state[4 + num_marker, r, c] = True
            self.markers_grid[r, c] -= 1
            
    def put_marker(self) -> None:
        r, c, _ = self.hero_pos
        num_marker = self.markers_grid[r, c]
        if num_marker == MAX_MARKERS_PER_SQUARE:
            if self.crashable:
                self.crashed = True
        else:
            self.state[5 + num_marker, r, c] = False
            self.state[6 + num_marker, r, c] = True
            self.markers_grid[r, c] += 1
