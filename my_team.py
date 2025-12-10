# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from contest import util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import math


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    shared_data = { # data shared among team members
        'last_seen_positions': {},
        'team_plan': "search", # can be "search" or "return" or "drift" for now
        'communication': {},
        'deposited_food': {}, # total food deposited by each agent
        'last_deposited': {}, # food carried at last step by each agent
        'first_a': True, # to initialize shared data
        'first_b': True, # to initialize shared data
        'first_c': True, # To know if it is the first action
        'safe_food_list': [],
        'food_list': [],
        'time': 0,
        'time_enter': 0,
        'row_time': [],
        'top': True
    }

    # # Access shared data
    #   defensive_info = self.shared_data.get('defensive_agent_info', {})
        
    # # Update shared data
    #   self.shared_data['offensive_target'] = self.current_food_target

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state): #-6 a 6
        """
        Picks among the actions with the highest Q(s,a).
        """
        
        print("--------------- DEBUGGING ---------------\n")
        self.shared_data['time'] += 1 # general time
        print("|    Agent ", self.index," Time: ", self.shared_data['time'])
        
        if game_state.get_agent_state(self.index).is_pacman: # If I am pacman [bdta] then add time
            self.shared_data['time_enter'] += 1 # add 1 second
        else:
            self.shared_data['time_enter'] = 0 # time is 0 ( I have not entered enemy territory)
        print("|    Time since entered: ", self.shared_data['time_enter'],"s\n")

        self.shared_data['first_c'] = True # general time

        actions = game_state.get_legal_actions(self.index)
        self.last_observation = game_state

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        print("---------------- ACTIONS ----------------\n")
        print("|    Possible: \n",actions)
        for i, v in enumerate(values):
            print("|    ",actions[i]," has value ",v,"\n")
        print("-----------------------------------------\n")
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        r = random.choice(best_actions)
        print("|    FINAL DECISION: \n")
        print("|    Agent ", self.index, " choosing going ", r, " with value ", max_value, "\n\n\n\n")
        return r

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def get_border_position(self, game_state):
        """Returns the y-coordinate of the border line"""
        layout = game_state.data.layout
        width = layout.width
        # The border is at the midpoint
        border_x = width // 2
        border_x -= 1 if self.red else 0  # Adjust for which side we're on
        return border_x
    def get_enemy_border_position(self, game_state):
        """Returns the y-coordinate of the border line"""
        layout = game_state.data.layout
        width = layout.width
        # The border is at the midpoint
        border_x = width // 2
        border_x -= 1 if not self.red else 0  # Adjust for which side we're on
        return border_x
    def distance_to_border(self, game_state, position):
        layout = game_state.data.layout
        height = layout.height
        border_x = self.get_border_position(game_state)
        valid_border_positions = []
        for y in range(height):
            border_pos = (border_x, y)
            # Check if this border position is not a wall and is within bounds
            if (0 <= border_pos[0] < layout.width and 
                0 <= border_pos[1] < layout.height and
                not game_state.has_wall(border_pos[0], border_pos[1])):
                valid_border_positions.append(border_pos)
        
        if not valid_border_positions:
            x, y = position
            return abs(position[0] - border_x)
        return min([self.get_maze_distance(position, pos) for pos in valid_border_positions])
    
    
    def distance_to_td(self, game_state, position, top):
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        border_x = self.get_border_position(game_state)

        valid_boundary_positions = []
        target_y = height - 2 if top else 1

        # Find x range for our territory
        if self.red:
            # Red: left side including border
            for x in range(border_x + 1):  # 0 to border_x inclusive
                boundary_pos = (x, target_y)
                if (0 <= boundary_pos[0] < width and 
                    0 <= boundary_pos[1] < height and
                    not game_state.has_wall(boundary_pos[0], boundary_pos[1])):
                    valid_boundary_positions.append(boundary_pos)
        else:
            # Blue: right side including border
            for x in range(border_x, width):
                boundary_pos = (x, target_y)
                if (0 <= boundary_pos[0] < width and 
                    0 <= boundary_pos[1] < height and
                    not game_state.has_wall(boundary_pos[0], boundary_pos[1])):
                    valid_boundary_positions.append(boundary_pos)

        if not valid_boundary_positions:
            x, y = position
            return abs(y - target_y)

        return min([self.get_maze_distance(position, pos) for pos in valid_boundary_positions])

    def seek_future(self, game_state, depth): # return true if you die
        """
        Simulate future positions to avoid ghosts
        """
        # Placeholder for future seeking logic
        
        if depth == 0:
            return [False, depth]

        # if dead_end_corner(self, game_state):
        #     return [True, depth]
        
        actions = game_state.get_legal_actions(self.index)
        
        dead = True
        dead_depth = -1
        enter = False

        for a in actions: # for each action
            if a != Directions.STOP and a != Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]: # avoid stop and reverse
                enter = True
                successor = game_state.generate_successor(self.index, a)
                result = self.seek_future(successor, depth-1)
                dead = dead and result[0]
                if not dead:
                    return [False, depth]
                dead_depth = max(dead_depth, result[1])
        
        if enter:
            return [True, dead_depth]
        else:
            return [True, depth]
        


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        
        print("-----------------------------------------\n")

        layout = game_state.data.layout                     # Layout of the game (for width, height, ...)
        width = layout.width
        height = layout.height

        if self.shared_data['first_a']: # General Inicialization for both agents
            for i in self.get_team(game_state):
                self.shared_data['last_deposited'][i] = 0
                self.shared_data['deposited_food'][i] = 0
            self.shared_data['first_a'] = False
            for h in range(height):
                self.shared_data['row_time'].append(0)
        
        # Initial Variables
        features = util.Counter()                           # Counter structure (like a vector)
        successor = self.get_successor(game_state, action)  # Get Gamestate after doing the action
        food_list = self.get_food(successor).as_list()      # Get the list with the position of foods
        features['successor_score'] = -len(food_list)       # Score = - Lenght of the food (less food = better)    
        im_pac = game_state.get_agent_state(self.index).is_pacman # If I am pacman [bdta] then add time                                    # self.get_score(successor)

        team_plan = self.shared_data['team_plan']           # Team plan
        range_stuck = 1
        max_turns_stuck = 48
        depth = 5                                           # Depth of the SeekFuture Algorithm
        points_return = 4 - (self.shared_data['time_enter'] // 15)

        for_len = 1 + 2*range_stuck
        max_val_stuck = max_turns_stuck - for_len
        discount_factor = 1/max_turns_stuck                               # Discount factor
        count_factor = 1
        

        points_carrying = game_state.get_agent_state(self.index).num_carrying # Num of points carrying after doing the action
        my_pos = successor.get_agent_state(self.index).get_position()        # Get my position after doing the action
        my_state = successor.get_agent_state(self.index)                     # Get state after doing the action (adta = after doing the action)
        s = self.distance_to_border(game_state, my_pos)                      # Maze distance from Position [adta] to border (where I deposit points and become ghost)

        
        print("|    Action",action,"\n|    ",game_state.get_agent_state(self.index).get_position(),"-->",successor.get_agent_state(self.index).get_position(),"\n|    Time: ",self.shared_data['time'],"\n\n")
        if action == Directions.STOP: 
            features['stop'] = 1
            print("-----------------------------------------\n")
            return features
        
        print("-------------- OTHER VALUES --------------\n")
        print("|    <Initially>\n")
        print("|    Border Position: ",self.get_border_position(game_state),"\n")
        print("|    Plan: ",team_plan,"\n")
        print("|    Points: ",points_return,"\n")

        xz, yz = game_state.get_agent_state(self.index).get_position()
        # if xz == self.get_enemy_border_position(game_state):
        #     opponent_indices = self.get_opponents(successor)
        #     enemies_states = [successor.get_agent_state(i) for i in opponent_indices]
        #     enemies_positions = [e.get_position() for e in enemies_states]

        #     enemies_states_past = [game_state.get_agent_state(i) for i in opponent_indices]
        #     enemies_positions_past = [e.get_position() for e in enemies_states_past]

        #     if any(enemies_positions):
        #         opponent_distances = [self.get_maze_distance(my_pos, pos) for pos in enemies_positions if pos is not None]

        #         if opponent_distances and points_carrying > 0:
        #             self.shared_data['team_plan'] = 'drift' # iniciate return plan globally
        #             team_plan = 'drift'                     # iniciate return plan in this action
        #             print("|    Starting to Drift !!!\n")
        #             reset = True
        #             y_no = yz
        #             self.shared_data['top'] = y_no < height // 2
        
        y_no = height//2
        if self.shared_data['first_c']:
            reset = False
            for h in range(height):
                d = self.shared_data['row_time'][h]
                d = max(d - discount_factor, 0)
                self.shared_data['row_time'][h] = d
            if not im_pac:
                #print ("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                for i in range(for_len):
                    
                    xf, yf = game_state.get_agent_state(self.index).get_position()
                    c = math.floor(yf) - range_stuck + i
                    d = self.shared_data['row_time'][c]
                    final = d + count_factor
                    if final >= max_val_stuck and reset == False:
                        self.shared_data['team_plan'] = 'drift' # iniciate return plan globally
                        team_plan = 'drift'                     # iniciate return plan in this action
                        print("|    Starting to Drift !!!\n")
                        reset = True
                        y_no = yf
                        self.shared_data['top'] = y_no < height // 2
                    self.shared_data['row_time'][c] = final
                print("|    Height values: ",self.shared_data['row_time'] ,"\n")
        
        

        if reset or team_plan == 'drift':
            for h in range(height):
                self.shared_data['row_time'][h] = 0
            reset = False
            print("|    Height values 2: ",self.shared_data['row_time'] ,"\n")
        
        if team_plan == 'drift':
            xg, yg = my_pos
            if self.shared_data['top'] == False and yg < 2:
                self.shared_data['team_plan'] = 'search' # iniciate return plan globally
                team_plan = 'search'                     # iniciate return plan in this action
            elif self.shared_data['top'] == True and yg > height - 3:
                self.shared_data['team_plan'] = 'search' # iniciate return plan globally
                team_plan = 'search'                     # iniciate return plan in this action
        
        tow = 0
        if self.red:
            tow = 1
        else:
            tow = -1

        # if xz == self.get_enemy_border_position(game_state) or xz == self.get_enemy_border_position(game_state) + tow:
        #     if points_carrying > 0:
        #         self.shared_data['team_plan'] = 'return' # iniciate return plan globally
        #         team_plan = 'return'                     # iniciate return plan in this action
        #         print("RETURN, Close to Border with points\n")



            
        

        

        if (points_carrying >= points_return): # If I have a lot of points
            self.shared_data['team_plan'] = 'return' # iniciate return plan globally
            team_plan = 'return'                     # iniciate return plan in this action
            print("RETURN, too many points\n")

        elif self.shared_data['time_enter'] > 40:       # if it's been a while and I have not returned
            self.shared_data['team_plan'] = 'return'    # Return Glob
            team_plan = 'return'                        # Return Loc
            print("RETURN, too much time\n")

        elif points_carrying == 0 and self.shared_data['last_deposited'][self.index] != 0: # You deposited food or died
            if my_pos == self.start: # If you died (you are at the start position)
                self.shared_data['team_plan'] = 'search'
                team_plan = 'search'
                print("SEARCH, you died\n")
            else: # You deposited food
                self.shared_data['deposited_food'][self.index] = self.shared_data['last_deposited'][self.index] + self.shared_data['deposited_food'][self.index]
                self.shared_data['team_plan'] = 'drift'
                team_plan = 'drift'
                print("SEARCH, you deposited\n")

        

        self.shared_data['last_deposited'][self.index] = points_carrying
        # if action == Directions.REVERSE:
        

        # Compute distance to the nearest food
        if team_plan == 'search':
            if len(food_list) > 0:  # This should always be True,  but better safe than sorry
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance


            opponent_indices = self.get_opponents(successor)
            enemies_states = [successor.get_agent_state(i) for i in opponent_indices]
            enemies_positions = [e.get_position() for e in enemies_states]

            enemies_states_past = [game_state.get_agent_state(i) for i in opponent_indices]
            enemies_positions_past = [e.get_position() for e in enemies_states_past]
            print("|    Enemy Positions: ",enemies_positions,"\n")

            if any(enemies_positions):
                opponent_distances = [self.get_maze_distance(my_pos, pos) if pos is not None else -1 for pos in enemies_positions]

                print("|    Enemy Distances: ",opponent_distances,"\n")

                

                if opponent_distances:
                    x,_ = my_pos
                    border_x = self.get_border_position(successor)
                    border_x -= 1 if self.red else 0  # Adjust for which side we're on
                    
                    
                    gd = []
                    pd = []
                    for i, dist in enumerate(opponent_distances):
                        if dist != -1:
                            print("|    Agent ", self.index, " opponent ", i, " dist: ", dist, "enemy_position:", enemies_positions[i], "my_position", my_pos,"\n")
                            
                            
                            if enemies_positions_past[i] is not None:
                                if self.get_maze_distance(enemies_positions_past[i], enemies_positions[i]) != 0:
                                    pd.append(0)
                            print("|    MP: ", my_pos," PP: ",game_state.get_agent_state(self.index).get_position()," Dist: ",self.get_maze_distance(my_pos, game_state.get_agent_state(self.index).get_position()))
                            if self.get_maze_distance(my_pos, game_state.get_agent_state(self.index).get_position()) > 2:
                                gd.append(0)
                            
                            print("|    Enemy: ",i," is pacman ",enemies_states[i].is_pacman,"\n")
                            if enemies_states[i].is_pacman:
                                
                                if dist < 4:
                                    pd.append(dist)
                                    print("Agent ", self.index, " has seen pacman opponent ", i, " dist: ", dist)
                                # elif dist > 10:
                                #     pd.append(0)
                                #     print("Agent ", self.index, " is on top of ghost opponent ", i, " dist: ", dist)
                            else:
                                if ((self.red and x > border_x) or (not self.red and x < border_x)): # if in enemy territory
                                    result = self.seek_future(successor, depth)
                                    if result[0]: # if dead
                                        #if depth - result[1] + 3 >= min(opponent_distances) or towards_end_corner(self, game_state): # if dead soon
                                        features['dodge'] = 1
                                    print("Agent ", self.index, " seek_future result: ", result, " depth: ", depth, "\n")
                                if dist < 5 and enemies_states[i].scared_timer <= 0:
                                    print("Agent ", self.index, " has seen ghost opponent ", i, " dist: ", dist)
                                    gd.append(dist)
                                    power_pellet_positions = self.get_capsules(successor)
                                    for pellet_pos in power_pellet_positions:
                                        if self.get_maze_distance(my_pos, pellet_pos) < 4:
                                            features['power_pellet'] = self.get_maze_distance(my_pos, pellet_pos)
                            
                            self.shared_data['last_seen_positions'][i] = enemies_positions[i]
                            
                                

                    features['closest_ghost_distance'] = min(gd) if gd else 5
                    features['closest_pacman_distance'] = min(pd) if pd else 6
                        
        elif team_plan == 'return':
            my_pos = successor.get_agent_state(self.index).get_position()
            features['distance_to_border'] = self.distance_to_border(successor, my_pos)

            opponent_indices = self.get_opponents(successor)
            enemies_states = [successor.get_agent_state(i) for i in opponent_indices]
            enemies_positions = [e.get_position() for e in enemies_states]

            enemies_states_past = [game_state.get_agent_state(i) for i in opponent_indices]
            enemies_positions_past = [e.get_position() for e in enemies_states_past]

            if any(enemies_positions):
                opponent_distances = [self.get_maze_distance(my_pos, pos) if pos is not None else -1 for pos in enemies_positions]

                

                if opponent_distances:
                    x,_ = my_pos
                    border_x = self.get_border_position(successor)
                    border_x -= 1 if self.red else 0  # Adjust for which side we're on
                    
                    
                    gd = []
                    pd = []
                    for i, dist in enumerate(opponent_distances):
                        if dist != -1:
                            print("Agent ", self.index, " opponent ", i, " dist: ", dist, "enemy_position:", enemies_positions[i], "my_position", my_pos)
                            
                            if enemies_positions_past[i] is not None:
                                if self.get_maze_distance(enemies_positions_past[i], enemies_positions[i]) != 0:
                                    pd.append(0)
                            if self.get_maze_distance(my_pos, game_state.get_agent_state(self.index).get_position()) > 2:
                                gd.append(0)
                            if enemies_states[i].is_pacman:
                                
                                if dist < 4:
                                    pd.append(dist)
                                    print("Agent ", self.index, " has seen pacman opponent ", i, " dist: ", dist)
                                # elif dist > 10:
                                #     pd.append(0)
                                #     print("Agent ", self.index, " is on top of ghost opponent ", i, " dist: ", dist)
                            else:
                                if ((self.red and x > border_x) or (not self.red and x < border_x)): # if in enemy territory
                                    result = self.seek_future(successor, depth)
                                    if result[0]: # if dead
                                        #if depth - result[1] + 3 >= min(opponent_distances) or towards_end_corner(self, game_state): # if dead soon
                                        features['dodge'] = 1
                                    print("Agent ", self.index, " seek_future result: ", result, " depth: ", depth, "\n")
                                if dist < 5 and enemies_states[i].scared_timer <= 0:
                                    print("Agent ", self.index, " has seen ghost opponent ", i, " dist: ", dist)
                                    gd.append(dist)
                                    power_pellet_positions = self.get_capsules(successor)
                                    for pellet_pos in power_pellet_positions:
                                        if self.get_maze_distance(my_pos, pellet_pos) < 4:
                                            features['power_pellet'] = self.get_maze_distance(my_pos, pellet_pos)
                            
                            self.shared_data['last_seen_positions'][i] = enemies_positions[i]
                            
                                

                    features['closest_ghost_distance'] = min(gd) if gd else 5
                    features['closest_pacman_distance'] = min(pd) if pd else 6


        elif team_plan == 'drift':
            my_pos = successor.get_agent_state(self.index).get_position()
            
            features['distance_to_td'] = self.distance_to_td(game_state, my_pos, self.shared_data['top'])

            opponent_indices = self.get_opponents(successor)
            enemies_states = [successor.get_agent_state(i) for i in opponent_indices]
            enemies_positions = [e.get_position() for e in enemies_states]

            enemies_states_past = [game_state.get_agent_state(i) for i in opponent_indices]
            enemies_positions_past = [e.get_position() for e in enemies_states_past]

            if any(enemies_positions):
                opponent_distances = [self.get_maze_distance(my_pos, pos) if pos is not None else -1 for pos in enemies_positions]

                

                if opponent_distances:
                    x,_ = my_pos
                    border_x = self.get_border_position(successor)
                    border_x -= 1 if self.red else 0  # Adjust for which side we're on
                    
                    
                    gd = []
                    pd = []
                    for i, dist in enumerate(opponent_distances):
                        if dist != -1:
                            print("Agent ", self.index, " opponent ", i, " dist: ", dist, "enemy_position:", enemies_positions[i], "my_position", my_pos)
                            
                            if enemies_positions_past[i] is not None:
                                if self.get_maze_distance(enemies_positions_past[i], enemies_positions[i]) != 0:
                                    pd.append(0)
                            if self.get_maze_distance(my_pos, game_state.get_agent_state(self.index).get_position()) > 2:
                                gd.append(0)
                            if enemies_states[i].is_pacman:
                                
                                if dist < 4:
                                    pd.append(dist)
                                    print("Agent ", self.index, " has seen pacman opponent ", i, " dist: ", dist)
                                # elif dist > 10:
                                #     pd.append(0)
                                #     print("Agent ", self.index, " is on top of ghost opponent ", i, " dist: ", dist)
                            else:
                                if ((self.red and x > border_x) or (not self.red and x < border_x)): # if in enemy territory
                                    result = self.seek_future(successor, depth)
                                    if result[0]: # if dead
                                        #if depth - result[1] + 3 >= min(opponent_distances) or towards_end_corner(self, game_state): # if dead soon
                                        features['dodge'] = 1
                                    print("Agent ", self.index, " seek_future result: ", result, " depth: ", depth, "\n")
                                if dist < 5 and enemies_states[i].scared_timer <= 0:
                                    print("Agent ", self.index, " has seen ghost opponent ", i, " dist: ", dist)
                                    gd.append(dist)
                                    power_pellet_positions = self.get_capsules(successor)
                                    for pellet_pos in power_pellet_positions:
                                        if self.get_maze_distance(my_pos, pellet_pos) < 4:
                                            features['power_pellet'] = self.get_maze_distance(my_pos, pellet_pos)
                            
                            self.shared_data['last_seen_positions'][i] = enemies_positions[i]
                            
                                

                    features['closest_ghost_distance'] = min(gd) if gd else 5
                    features['closest_pacman_distance'] = min(pd) if pd else 6
        
            # noisy_distances = successor.get_agent_distances() # array of distances to all agents with noise
            
            # if noisy_distances:
            #     opponent_distances = [noisy_distances[i] for i in opponent_indices]
            #     features['closest_opponent_distance'] = min(opponent_distances)
        
        
        print("|    Features: \n",features,"\n\n")
        print("|    Other Info:\n")

        print( "Carrying: ", points_carrying, "distance_to_border:", s, "my_pos:", my_pos, "\n")
            
        
        print("")
        # if self.get_maze_distance(my_pos, game_state.get_agent_state(self.index).get_position()) > 10:
        #     features['closest_ghost_distance'] = 0.5
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'closest_ghost_distance': 20, 'closest_pacman_distance': -2, 'power_pellet' : -5, 'distance_to_border': -1, 'distance_to_td' : -1, 'stop': -1000, 'dodge': -1000}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):

        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        quarter_width = layout.width // 4

        if self.shared_data['first_b']:
            print('first')
            self.shared_data['safe_food_list'] = self.get_food_you_are_defending(game_state).as_list()
            self.shared_data['food_list'] = list(
                filter(lambda food: (
                    food[0] >= quarter_width and           # x >= quarter width
                    food[1] >= 3 and                       # y >= 3 (not too close to bottom)
                    food[1] <= height - 4 and              # y <= height - 4 (not too close to top)
                    food[0] <= width - quarter_width       # x <= width - quarter_width (not too close to right)
                ), self.shared_data['safe_food_list'])
            )
            self.shared_data['first_b'] = False
        
        #print('second')
        if len(self.shared_data['food_list']) < 2:
            self.shared_data['food_list'] = list(
                filter(lambda food: (
                    food[0] >= quarter_width and           # x >= quarter width
                    food[1] >= 3 and                       # y >= 3 (not too close to bottom)
                    food[1] <= height - 4 and              # y <= height - 4 (not too close to top)
                    food[0] <= width - 5                   # x <= width - 5 (not too close to right)
                ), self.shared_data['safe_food_list'])
            )

        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            
            if game_state.get_agent_state(self.index).scared_timer > 0 : #not work
                if features['invader_distance'] == 0:
                    features['invader_distance'] = 5 # to avoid going on top of invader when scared
        else:
            if len(self.shared_data['food_list'] ) > 0:  # This should always be True,  but better safe than sorry
                distances = [self.get_maze_distance(my_pos, food) for food in self.shared_data['food_list']]
                min_distance = min(distances)
                features['distance_to_food'] = min_distance
                if min_distance == 0:
                    min_index = distances.index(min_distance)
                    
                    self.shared_data['food_list'].pop(min_index)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'distance_to_food': -1, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


def index_from_indeces(indeces, index):
    sorted = sorted(indeces) # Sort indeces to have a consistent order
    for i in range(len(sorted)):
        if indeces[i] == index:
            return i
    return -1
def dead_end_corner(self, game_state):
    """
    Check if the agent is in a dead-end corner
    """
    my_pos = game_state.get_agent_state(self.index).get_position()
    walls = 0
    layout = game_state.data.layout
    x, y = my_pos
    directions = [(1,0), (-1,0), (0,1), (0,-1)] # right, left, up, down

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if game_state.has_wall(nx, ny):
            walls += 1

    return walls >= 3
def towards_end_corner(self, game_state):
    """
    Check if the agent is in a dead-end corner
    """
    my_pos = game_state.get_agent_state(self.index).get_position()
    walls = 0
    # layout = game_state.data.layout
    x, y = my_pos
    x = math.floor(x)
    y = math.floor(y)
    directions = [(1,0), (-1,0), (0,1), (0,-1)] # right, left, up, down

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if game_state.has_wall(nx, ny):
            walls += 1

    return walls >= 2

#### Functiones
# self.distancer.getDistance(p1, p2) : Devuelve la maze distance entre los puntos p1 y p2
# self.getFood(gameState) : Devuelve una grid con la comida del equipo contrario