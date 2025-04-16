import mesa

# Data visualization tools.
import seaborn as sns

# Has multi-dimensional arrays and matrices. Has a large collection of
# mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd

import matplotlib.pyplot as plt
import copy


def compute_gini(model):
  agent_wealths = [agent.wealth for agent in model.schedule.agents]
  x = sorted(agent_wealths)
  N = model.num_agents
  B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
  return 1 + (1 / N) - 2 * B

class MoneyAgent(mesa.Agent):
  """An agent with fixed initial wealth."""

  def __init__(self, unique_id, model):
    # Pass the parameters to the parent class.
    super().__init__(unique_id, model)

    # Create the agent's variable and set the initial values.
    self.wealth = 1

  def move(self):
    possible_steps = self.model.grid.get_neighborhood(
      self.pos,
      moore=True,
      include_center=False)
    new_position = self.random.choice(possible_steps)
    self.model.grid.move_agent(self, new_position)

  def give_money(self):
    cellmates = self.model.grid.get_cell_list_contents([self.pos])
    if len(cellmates) > 1:
      other = self.random.choice(cellmates)
      other.wealth += 1
      self.wealth -= 1
      #if other == self:
        #print("I JUST GAVE MONEY TO MYSELF HEHEHE!")

  def step(self):
    # The agent's step will go here.
    # For demonstration purposes we will print the agent's unique_id
    # print(f"Hi, I am an agent, you can call me {str(self.unique_id)} and my wealth is {str(self.wealth)}."
    self.move()
    if self.wealth > 0:
      self.give_money()
      # other_agent = self.random.choice(self.model.schedule.agents)
      # if other_agent is not None:
      #  other_agent.wealth += 1
      # self.wealth -= 1

class MoneyModel(mesa.Model):
  """A model with some number of agents."""

  def __init__(self, N, width, height):
    self.num_agents = N
    self.grid = mesa.space.MultiGrid(width, height, True)

    # Create scheduler and assign it to the model
    self.schedule = mesa.time.RandomActivation(self)
    self.running = True

    # Create agents
    for i in range(self.num_agents):
      a = MoneyAgent(i, self)
      # Add the agent to the scheduler
      self.schedule.add(a)

      # Add the agent to a random grid cell
      x = self.random.randrange(self.grid.width)
      y = self.random.randrange(self.grid.height)
      self.grid.place_agent(a, (x, y))

      self.datacollector = mesa.DataCollector(
        model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth"}
      )

  def step(self):
    """Advance the model by one step."""

    self.datacollector.collect(self)

    # The model's step will go here for now this will call the step method of each agent and print the agent's
    # unique_id
    self.schedule.step()
