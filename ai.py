# -*- coding: utf-8 -*-

# python imports
import random
import copy

# chillin imports
from chillin_client import RealtimeAI

# project imports 
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker
# import position, direction, agent, ecell, world



class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)


    def initialize(self): 
        print('initialize') 


    def check_end_game(self, state):
        if self.current_cycle >= 300 - 1:
            return True


        for _, agent in state.agents.items():
            if self.check_crash_wall(state, agent) or self.check_crash_agent(state):
                return True

        return False  
      


    def check_crash_wall(self, state, agent):
        return (state.board[agent.position.y][agent.position.x] == ECell.AreaWall or \
                state.board[agent.position.y][agent.position.x] != ECell.Empty and agent.health == 0)
    

    def check_crash_agent(self, state):
        agents = list(state.agents.values())

        # if (agents[0].prev_position == agents[1].position and
        #     agents[1].prev_position == agents[0].position):
        #     return True

        return agents[0].position == agents[1].position
    

    def construct_wall(self, state, side): 
        agent = state.agents[side]
        my_cell_name = side + "Wall"
        my_cell_type = ECell[my_cell_name]
        state.board[agent.position.y][agent.position.x] = my_cell_type
        state.scores[side] += state.constants.wall_score_coefficient


    def decrease_health(self, agent):
        agent.health -= 1 

    
    def handle_collision(self, state, side):
        enemy_side = [s for s in state.agents.keys() if s != side][0]

        agent = state.agents[side]
        # CrashEnemyAgent
        if self.check_crash_agent(state):
            agent.crashed = True
            agent.move_before_crash = state.agents[side].position == state.agents[enemy_side].position
            return 

        # CrashAreaWall 
          
        if state.board[agent.position.y][agent.position.x] == ECell.AreaWall:
            agent.crashed = True
            agent.move_before_crash = True
            state.scores[side] += state.constants.area_wall_crash_score
            return 

        # WallBreaker
        my_cell_type = ECell.get_wall_type(side)
        enemy_cell_type = ECell.get_wall_type(enemy_side)

        if (agent.wall_breaker_rem_time > 0 and \
            state.board[agent.position.y][agent.position.x] in [my_cell_type, enemy_cell_type]):
            return self.destruct_wall(state, side) 

        # CrashMyWall
        if state.board[agent.position.y][agent.position.x] == my_cell_type:
            self.decrease_health(agent)
            if agent.health == 0:
                self.crashed = True
                # my_wall_refs = state.wall_refs[side][(self.position.x, self.position.y)]
                # agent.move_before_crash = opposite(agent.direction) not in my_wall_refs.keys()
                state.scores[side] += state.constants.my_wall_crash_score
            else:
                self.destruct_wall(state, side) 

            

        # CrashEnemyWall
        if state.board[agent.position.y][agent.position.x] == enemy_cell_type:
            self.decrease_health(agent)
            if agent.health == 0:
                self.crashed = True
                # enemy_wall_refs = state.wall_refs[enemy_side][(self.position.x, self.position.y)]
                # self.move_before_crash = opposite(agent.direction) not in enemy_wall_refs.keys()
                state.scores[side] += state.constants.enemy_wall_crash_score
            else:
                self.destruct_wall(state, side)

            

    def tick(self, state): 
        for side, agent in state.agents.items():
            self.tick_wall_breaker(agent)

        for side, agent in state.agents.items():
            self.construct_wall(state, side)

        for side, agent in state.agents.items():
            self.move(state, agent)

        for side, agent in state.agents.items():
            self.handle_collision(state, side)


    
    def destruct_wall(self, state, side): 
        agent = state.agents[side]
        enemy_side = [s for s in state.agents.keys() if s != side][0]
        my_cell_type = ECell.get_wall_type(side)
        enemy_cell_type = ECell.get_wall_type(enemy_side)

        wall_side = side if state.board[agent.position.y][agent.position.x] == my_cell_type else enemy_side
        state.board[agent.position.y][agent.position.x] = ECell.Empty
        state.scores[wall_side] -= state.constants.wall_score_coefficient
            
        

    def tick_wall_breaker(self, agent):

        if agent.wall_breaker_cooldown > 0:
            agent.wall_breaker_cooldown -= 1

        if agent.wall_breaker_rem_time > 0:
            agent.wall_breaker_rem_time -= 1

 
    def result(self, state, action, side): 
        state = copy.deepcopy(state)
        agent = state.agents[side]
         

        # print("selected action is : ", action) 
        direction, wall_break = action 
        self.change_direction(agent, direction) 
        if wall_break : self.activate_wall_breaker(state, agent)

        self.tick(state)     

        # print("agent final position in this cycle is : ", agent.position)
        # print("opponent final position in this cycle is : ", state.agents[self.other_side].position) 
        # print("agent health is : ", agent.health) 
        # print("opponent health is : ", state.agents[self.other_side].health)


        return state

    def count_reachable(self, state, pos : tuple):
        visited = []
        to_visit = [pos]
        
        while len(to_visit) <= 100:
            # Get the next position to visit
            curr_pos = to_visit.pop(0)
            
            # Add the current position to the visited set
            
            neighbors = []
            if curr_pos[0] >= 1 and curr_pos[0] <= 33 and curr_pos[1] >= 1 and curr_pos[1] <= 19 :
                visited.append(curr_pos)
                neighbors = [(curr_pos[0] + 1, curr_pos[1]), (curr_pos[0], curr_pos[1] + 1), (curr_pos[0] - 1, curr_pos[1]), (curr_pos[0], curr_pos[1] - 1)]
                if curr_pos[0] == 1 :
                    neighbors.remove((curr_pos[0] - 1, curr_pos[1]))
                if curr_pos[1] == 1:
                    neighbors.remove((curr_pos[0], curr_pos[1] - 1))
                if curr_pos[0] == 33:
                    neighbors.remove((curr_pos[0] + 1, curr_pos[1]))
                if curr_pos[1] == 19:
                    neighbors.remove((curr_pos[0], curr_pos[1] + 1))
                
            print("visited222", visited)
            for neighbor in neighbors:
                if neighbor not in visited:
                    to_visit.append(neighbor)
        print(visited)
        count = sum(1 for pos in visited if state.board[pos[0]][pos[1]] == ECell.Empty)
        
        return count 
    
    def utility(self, state, side):
        enemy_side = [s for s in state.agents.keys() if s != side][0]  
        utility = (state.scores[side] - state.scores[enemy_side]) + state.agents[side].health * 20 
        # print(self.count_reachable(state, (state.agents[side].position.x, state.agents[side].position.y)))
        # utility += self.count_reachable(state, (state.agents[side].position.x, state.agents[side].position.y)) - self.count_reachable(state, (state.agents[enemy_side].position.x, state.agents[enemy_side].position.y))
        return utility 

    def move(self, state, agent):
        agent.prev_position = agent.position 
        agent.position += Position.dir_to_pos(agent.direction)
        

    def activate_wall_breaker(self, state, agent):
        if agent.wall_breaker_cooldown > 0:
            return 

        agent.wall_breaker_cooldown = state.constants.wall_breaker_cooldown
        agent.wall_breaker_rem_time = state.constants.wall_breaker_duration + 1
        
    

    def change_direction(self, agent, new_direction):
        if new_direction in [agent.direction, opposite(agent.direction)]:
            return
        
        agent.direction = new_direction
        


    def get_actions(self, state, side):
        agent = state.agents[side]
        agent_dir = agent.direction  

        directions = [(EDirection.Down, True), (EDirection.Down, False), (EDirection.Up, True), (EDirection.Up, False), (EDirection.Right, True),
                       (EDirection.Right, False), (EDirection.Left, True), (EDirection.Left, False)] 
        
        return list(filter(lambda t : t[0] != opposite(agent_dir), directions))



    def minimax(self, depth, state, maxi, alpha, beta): # return utility , move 

         
        # print(actions)
        side = self.my_side if maxi else self.other_side # check k  
        actions = self.get_actions(state, side) 
        # print("side is : ", maxi) 


        if depth == 3 or self.check_end_game(state):
            # print("termina")  
            utility = self.utility(state, self.my_side)
            # print("utilty at depth 3 : ", utility)
            return utility, None
        
        if maxi: 
            # print("current depth in max is : ", depth)
            v = float('-inf')
            best_action = None
            for a in actions:
                v1, a1 = self.minimax(depth + 1, self.result(state, a, side), False, alpha, beta)
                if v1 > v:
                    v, best_action = v1, a
                alpha = max(alpha, v)
                if beta <= v:
                    # print("beta")
                    break
            # print("max : ", v, "best action : ", best_action, " in depth : " ,depth) 
            return v, best_action
        else :
            v = float('inf')
            best_action = None
            for a in actions:
                # print("current depth in min is : ", depth)
                v1 , a1 = self.minimax(depth + 1, self.result(state, a, side), True, alpha, beta)
                if v1 < v:
                    v, best_action = v1, a
                beta = min(beta, v)
                if v <= alpha:
                    # print("alphaaa")
                    break
            # print("min : ", v, "best action : ", best_action," in depth : ", depth)
            return v, best_action
    
        
    """Genetic Algorithm"""

    def create_population(self, population_size, actions):
        population = []
        current_sequence = []
        for i in range(population_size) :
            for j in range(300 - self.current_cycle):
                current_sequence.append(random.choice(actions))
            population.append((current_sequence, self.calculate_fitness(current_sequence)))
        
        return population


    def calculate_fitness(self, individual):

        # individual is a sequence of actions 
        w1 = 1.0  # wall_score_coefficient
        w2 = -20  # area_wall_crash_score
        w3 = -40  # my_wall_crash_score
        w4 = -60  # enemy_wall_crash_score
        w5 = -100 # irregular actions

        area_wall_crash_score = 0
        my_wall_crash_score = 0
        enemy_wall_crash_score = 0
        wall_score_coefficient = 0
        irregular_actions = 0
        health = 3

        state = copy.deepcopy(self.world) 
        my_agent = state.agents[self.my_side]
        opponent = state.agents[self.other_side]
        check_wall_breaker_rem = my_agent.wall_breaker_rem_time
        check_wall_breaker_cool = my_agent.wall_breaker_cooldown
        
        for index, action in enumerate(individual): 
            state = self.result(state, action, self.my_side)
            my_agent = state.agents[self.my_side]
            
            regular_actions = self.get_actions(state, self.my_side)
            if individual[index + 1] not in regular_actions:
                irregular_actions += 1

            if self.check_crash_agent(state):
                return w1 * wall_score_coefficient + w2 * area_wall_crash_score + w3 * my_wall_crash_score + w4 * enemy_wall_crash_score + w5 * irregular_actions
            
            if state.board[my_agent.position.x][my_agent.position.y] == ECell.Empty:
                wall_score_coefficient += 1
            elif state.board[my_agent.position.x][my_agent.position.y] == ECell.AreaWall:
                area_wall_crash_score += 1
                return w1 * wall_score_coefficient + w2 * area_wall_crash_score + w3 * my_wall_crash_score + w4 * enemy_wall_crash_score + w5 * irregular_actions
            elif state.board[my_agent.position.x][my_agent.position.y] == ECell.BlueWall:
                if not action[1]:
                    if health == 1 :
                        return w1 * wall_score_coefficient + w2 * area_wall_crash_score + w3 * my_wall_crash_score + w4 * enemy_wall_crash_score + w5 * irregular_actions
                    health -= 1
                    if self.my_side == "Yellow" :
                        enemy_wall_crash_score += 1
                    else : 
                        my_wall_crash_score += 1
                else:
                    if self.my_side == "Blue":
                        my_wall_crash_score += 1

            elif state.board[my_agent.position.x][my_agent.position.y] == ECell.YellowWall:
                if not action[1]:
                    if health == 1 :
                        return w1 * wall_score_coefficient + w2 * area_wall_crash_score + w3 * my_wall_crash_score + w4 * enemy_wall_crash_score + w5 * irregular_actions
                    health -= 1
                    if self.my_side == "Yellow":
                        my_wall_crash_score += 1
                    else:
                        enemy_wall_crash_score += 1
                else:
                    if self.my_side == "Yellow":
                        my_wall_crash_score += 1

        fitness = w1 * wall_score_coefficient + w2 * area_wall_crash_score + w3 * my_wall_crash_score + w4 * enemy_wall_crash_score + w5 * irregular_actions
        
        return fitness
    
    def selection(self, population):
        # tournament_size = 5
        # tournament = random.sample(population, tournament_size)
        # tournament.sort(key=lambda x: x[1], reverse=True) 
        population1 = population.sort(key=lambda x: x[1], reverse=True)
        return population1[0], population1[1]


    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1[0]) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child


    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            state = copy.deepcopy(self.world)
            index = random.randint(0, 300 - self.current_cycle)
            action = individual[index - 1]
            next_actions = self.get_actions(state, self.my_side)
            action = random.choice(next_actions)
            individual[index] = action
        return individual
        
    def genetic_algorithm(self, population):
        # Sort the population by fitness 
        population.sort(key=lambda x: x[1], reverse=True)

        # Keep the top 10% 
        best_size = int(0.1 * len(population))
        best = population[:best_size]

        # Create a new population 
        new_population = []
        while len(new_population) < len(population):
            # Select two parents
            parent1 = self.selection(population)
            parent2 = self.selection(population)

            # Crossover
            child = self.crossover(parent1, parent2)

            # Mutate
            child = self.mutate(child)

            # Calculate the fitness of the child 
            fitness = self.calculate_fitness(child)

            # Add the child and its fitness to the new population
            new_population.append((child, fitness))

        next_generation = best + new_population[:len(population) - best_size]

        return next_generation

    def decide(self):  

        state = copy.deepcopy(self.world)    
        agent = self.world.agents[self.my_side]  

        # print("agent is in : ", agent.position)
        action = self.minimax(0, state,  True, float('-inf'), float('inf'))[1]  
        
        if action is None:
            action = (agent.direction, False)


        if self.current_cycle == 10:
                genetic_actions = self.genetic_algorithm(self.create_population(50, [(EDirection.Down, True), (EDirection.Down, False), (EDirection.Up, True), (EDirection.Up, False), (EDirection.Right, True),
                       (EDirection.Right, False), (EDirection.Left, True), (EDirection.Left, False)]))
        if self.current_cycle >= 10:
                 action = genetic_actions[self.current_cycle - 10]

        # print("action is : ", action)  

         
        # print(type(action[0]))  
        
        
        self.send_command(ChangeDirection(action[0])) 
        pos = agent.position + dir_to_pos(action[0]) 
        cell = state.board[pos.y][pos.x]  

        if (action[1] and (cell == ECell.BlueWall or cell == ECell.YellowWall) ): 
            print("herree")
            self.send_command(ActivateWallBreaker())
        # self.send_command(ChangeDirection(random.choice(list(EDirection))))
        # if self.world.agents[self.my_side].wall_breaker_cooldown == 0:
        #     self.send_command(ActivateWallBreaker())


def opposite(direction):
    if direction == EDirection.Down:
        return EDirection.Up 
    elif direction == EDirection.Up: 
        return EDirection.Down 
    elif direction == EDirection.Right:
        return EDirection.Left 
    elif direction == EDirection.Left: 
        return EDirection.Right
    


def dir_to_pos(direction):
    dir_to_pos_map = {
        EDirection.Up: Position(0, -1),
        EDirection.Down: Position(0, +1),
        EDirection.Right: Position(+1, 0),
        EDirection.Left: Position(-1, 0)
    }
    return dir_to_pos_map[direction]


def is_valid(self, world):
    return 0 <= self.x < len(world.board[0]) and 0 <= self.y < len(world.board)


# def get_neighbors(world, neighbor_type=None):
#     neighbors = {}

#     for direction in EDirection:
#         neighbor_pos = self + Position.dir_to_pos(direction)
#         if (neighbor_pos.is_valid(world) and \
#             (neighbor_type is None or world.board[neighbor_pos.y][neighbor_pos.x] == neighbor_type)):
#             neighbors[direction] = neighbor_pos

#     return neighbors


# def get_8neighbors( world, neighbor_type=None):
#     neighbors = self.get_neighbors(world, neighbor_type=neighbor_type)
#     for direction, position in list(neighbors.items()):
#         neighbors[(direction,)] = position
#         del neighbors[direction]

#     for dir1 in EDirection:
#         for dir2 in EDirection:
#             if dir1 == dir2 or dir1 == dir2.opposite():
#                 continue

#             neighbor_pos = self + Position.dir_to_pos(dir1) + Position.dir_to_pos(dir2)
#             if (neighbor_pos.is_valid(world) and \
#                 (neighbor_type is None or world.board[neighbor_pos.y][neighbor_pos.x] == neighbor_type)):
#                 neighbors[(dir1, dir2)] = neighbor_pos

    return neighbors


def __eq__(self, other):
    if isinstance(other, Position):
        return self.x == other.x and self.y == other.y
    return NotImplemented


def __ne__(self, other):
    r = self.__eq__(other)
    if r is not NotImplemented:
        return not r
    return NotImplemented


def __hash__(self):
    return hash(tuple(sorted(self.__dict__.items())))


def __add__(self, other):
    if isinstance(other, Position):
        return Position(self.x + other.x, self.y + other.y)
    return NotImplemented


def __sub__(self, other):
    if isinstance(other, Position):
        return Position(self.x - other.x, self.y - other.y)
    return NotImplemented


def __repr__(self):
    return "<x: %s, y: %s>" % (self.x, self.y)


Position.dir_to_pos = dir_to_pos
Position.is_valid = is_valid
# Position.get_neighbors = get_neighbors
# Position.get_8neighbors = get_8neighbors
Position.__eq__ = __eq__
Position.__ne__ = __ne__
Position.__hash__ = __hash__
Position.__add__ = __add__
Position.__sub__ = __sub__
Position.__repr__ = __repr__
 

# ecell #### 
# 

def get_wall_type(side):
    cell_name = side + "Wall"
    return ECell[cell_name]


ECell.get_wall_type = get_wall_type 