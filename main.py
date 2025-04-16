import random

import mesa
import networkx as nx

import numpy


def get_all_tasks(model):
    all_tasks = 0
    for agent in model.schedule.agents:
        all_tasks += agent.tasks
    return all_tasks


def compute_main_relation(model):
    main_relation = 0
    sum_tasks = 0
    sum_performance = 0

    for agent in model.schedule.agents:
        main_relation += agent.relation
        sum_tasks += agent.tasks
        sum_performance += agent.performance

    coeff = float(sum_tasks) / sum_performance
    if coeff != 0:
        main_relation /= float(model.num_agents) * coeff
    return main_relation


class ParallelGraphComputingSystem(mesa.Model):

    def create_agent(self, i):
        p = 0
        if self.performance_diff > 0:
            diff = self.random.randrange(- self.performance_diff,
                                         self.performance_diff, 1)
            p = max(1, self.performance + diff)
        else:
            p = self.performance

        agent = ComputingAgent(i, self, p, 0, 0, 0)

        self.grid.place_agent(agent, i)
        self.schedule.add(agent)

    def get_tasks_partitions(self, cur_new_tasks):
        numbers = []
        for i in range(self.num_agents):
            numbers.append(random.randint(0, cur_new_tasks))

        s = sum(numbers)

        if s == 0:
            diff = cur_new_tasks
        else:
            k = cur_new_tasks / s
            for i in range(self.num_agents):
                numbers[i] = int(numbers[i] * k)
            s = sum(numbers)
            diff = cur_new_tasks - s

        numbers[self.random.randint(0, self.num_agents - 1)] = diff

        return numbers

    def complete_graph(self):
        node_groups = list(nx.components.connected_components(self.G))
        count = len(node_groups) - 1
        group = list(node_groups[0])
        node_groups.remove(node_groups[0])
        while count > 0:
            print(group)
            group_len = len(group)
            u = list(group)[self.random.randint(0, group_len - 1)]
            second_group = node_groups[self.random.randint(0, count - 1)]
            v = list(second_group)[self.random.randint(0, len(second_group) - 1)]
            self.G.add_edge(u, v)
            group.extend(list(second_group))
            node_groups.remove(second_group)
            count -= 1

    def __init__(self, num_agents, performance, performance_diff,
                 new_tasks, new_tasks_diff, graph_density=0.95, gamma_step=1):
        self.gamma_step = gamma_step
        self.graph_density = graph_density
        self.new_tasks = new_tasks
        self.new_tasks_diff = new_tasks_diff
        self.num_agents = num_agents
        self.performance = performance
        self.performance_diff = performance_diff

        assert new_tasks > new_tasks_diff
        assert performance > performance_diff

        self.G = nx.erdos_renyi_graph(self.num_agents, self.graph_density)
        self.complete_graph()

        # adding closed property to edges
        nx.set_edge_attributes(self.G, False, 'is_closed')

        # print(self.G.edges.data())
        # for edge in self.G.edges:
        #    edge['is_closed'] = False

        self.saved_edges = []
        self.grid = mesa.space.NetworkGrid(self.G)

        self.schedule = mesa.time.RandomActivation(self)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Main_relation": compute_main_relation
            },
            agent_reporters={
                "Tasks": "tasks",
                "Relation": "relation",
                "State": "state",
                "Performance": "performance",
            },
        )

        # Create agents
        for i in range(self.num_agents):
            self.create_agent(i)

        for agent in self.schedule.agents:
            agent.get_neighbors()

        self.running = True

    def rebuild_graph(self):
        graph = self.G

        # removing then saving and reviving different edges

        rev = 0
        rem = 0
        if self.saved_edges:
            rev = self.random.randint(0, len(self.saved_edges) - 1)
        if graph.edges:
            # rem = self.random.randint(0, len(graph.edges) - 1)
            rem = self.random.randint(0, int(float(len(graph.edges)) / 6))

        # removing
        for i in range(rem):
            d = list(graph.edges.data())
            ed = self.random.randint(0, len(d) - 1)
            (u, v, _) = d[ed]
            self.saved_edges.append(d[ed])
            graph[u][v]['is_closed'] = True

        # reviving
        for i in range(rev):
            ed = self.random.randint(0, len(self.saved_edges) - 1)
            (u, v, _) = self.saved_edges.pop(ed)
            graph[u][v]['is_closed'] = False

    def step(self):

        self.datacollector.collect(self)

        # rebuild graph
        self.rebuild_graph()

        # give new tasks
        if self.new_tasks_diff > 0:
            diff = self.random.randrange(- self.new_tasks_diff, self.new_tasks_diff, 1)
            cur_new_tasks = max(self.new_tasks + diff, 0)
        else:
            cur_new_tasks = self.new_tasks

        partitions = self.get_tasks_partitions(cur_new_tasks)
        for i in range(self.num_agents):
            self.schedule.agents[i].new_tasks = partitions[i]

        self.schedule.step()

        # averaging
        m = 10000000
        for i in range(self.num_agents):
            m = min(self.schedule.agents[i].state, m)

        for agent in self.schedule.agents:
            agent.state += 1 - m


class ComputingAgent(mesa.Agent):

    def __init__(self, unique_id, model, performance, tasks, state, new_tasks):
        super().__init__(unique_id, model)
        self.performance = performance
        self.tasks = tasks
        self.state = state
        self.new_tasks = new_tasks
        self.relation = float(self.tasks) / self.performance

    def get_neighbors(self):
        self.neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        self.nstates = []

        for n in self.neighbors:
            self.nstates.append(n.state)

    def get_neighbours_information(self):

        # Update states
        for i in range(self.neighbors):
            self.nstates[i] = self.neighbors[i].state

    def exchange_tasks(self, neighbor):

        # yij - yii = xj - xi
        state_difference = neighbor.state - self.state

        # bij = bj / bi
        weight = float(neighbor.performance) / self.performance

        # ui = gamma * bij(xj - xi)
        ui = float(self.model.gamma_step) * weight * state_difference

        # state += \bar ui = ui/pi
        self.state += ui / self.performance

        tasks_change = int(ui)
        if tasks_change >= 1:
            tasks_to_pick = tasks_change
            tasks_to_pick = min(neighbor.tasks, tasks_to_pick)

            self.tasks += tasks_to_pick
            neighbor.tasks -= tasks_to_pick

        elif tasks_change <= -1:
            tasks_to_give = -tasks_change
            tasks_to_give = min(self.tasks, tasks_to_give)

            self.tasks -= tasks_to_give
            neighbor.tasks += tasks_to_give

    def step(self):
        # x(t+1) = xt + f + \bar u
        # f = -1 + z/p

        # x += f
        self.state += -1 + self.new_tasks / self.performance

        # x += \bar u
        for neighbor in self.neighbors:
            if not self.model.G[self.pos][neighbor.pos]['is_closed']:
                self.exchange_tasks(neighbor)

        # q(t+1) = q(t) - p + z + u, but u is added

        # q += z - p
        self.tasks += self.new_tasks - self.performance

        self.tasks = max(0, self.tasks)

        self.relation = float(self.tasks) / self.performance
