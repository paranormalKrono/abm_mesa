from math import sin

import mesa
from prettytable import PrettyTable

from main import ParallelGraphComputingSystem

model_params = {
    "num_agents": mesa.visualization.Slider(
        "Number of agents",
        7,
        2,
        20,
        1,
        description="Choose how many agents to include in the model",
    ),
    "performance": mesa.visualization.Slider(
        "Performance",
        11,
        1,
        30,
        1,
        description="Choose a performance for every agent",
    ),
    "performance_diff": mesa.visualization.Slider(
        "Performance difference",
        6,
        0,
        20,
        1,
        description="Choose a performance difference for every agent",
    ),
    "new_tasks": mesa.visualization.Slider(
        "New tasks",
        80,
        1,
        200,
        1,
        description="Choose a count of new tasks",
    ),
    "new_tasks_diff": mesa.visualization.Slider(
        "New tasks difference",
        25,
        0,
        80,
        1,
        description="Choose a difference of new tasks",
    ),
    "graph_density": mesa.visualization.Slider(
        "Graph density",
        0.25,
        0,
        1,
        0.05,
        description="Choose a density of graph",
    ),
    "gamma_step": mesa.visualization.Slider(
        "Gamma",
        1,
        1,
        10,
        1,
        description="Choose a multiplier when exchanging tasks",
    ),
}


def get_color_by_number(agent, tasks_count):
    if tasks_count < 1 or agent.tasks < agent.performance:
        color = "#000000"
        # print(f"{agent.unique_id} - {color}")
        return color
    norm = float(agent.tasks) / tasks_count
    # norm = float(agent.tasks) / tasks_count
    func = pow(sin(norm), 1. / 4) * 255.
    tasks = int(func)

    color = '#%02x%02x%02x' % (tasks, 0, 0)
    # print(f"{agent.unique_id} - {tasks} - {color}")
    return color


def network_portrayal(graph):
    # The model ensures there is 1 agent per node
    tasks_count = 0

    # tasks count
    for (node_id, agents) in graph.nodes.data("agent"):
        tasks_count += agents[0].tasks

    portrayal = \
        {"nodes": [
            {
                "id": node_id,
                "size": 7 if agents else 5,
                "color": get_color_by_number(agents[0], tasks_count),
                "text": f"{agents[0].unique_id} - {agents[0].tasks}",
                "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].state}<br>"
                           f"performance: {agents[0].performance}<br>tasks: {agents[0].tasks}",
            }
            # agents in one cell
            for (node_id, agents) in graph.nodes.data("agent")
        ],  "edges": [
                {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "color": "#808080" if not graph.edges[source, target]["is_closed"] else "#C5D0E6"
                }
                for edge_id, (source, target) in enumerate(graph.edges)
            ]}

    return portrayal


grid = mesa.visualization.NetworkModule(network_portrayal,
                                        500, 500)


def get_full_information(model):
    table = PrettyTable()
    table.field_names = ["unique_id", "performance", "tasks", "relation",
                         "state", "new_tasks"]
    table.padding_width = 2
    table.align = "c"
    table.border = True
    table.format = True

    for agent in model.schedule.agents:
        table.add_row([agent.unique_id, agent.performance,
                       agent.tasks, f"{agent.relation:.2f}", f"{agent.state:.2f}", agent.new_tasks])

    return table.get_html_string()


server = mesa.visualization.ModularServer(
    ParallelGraphComputingSystem, [grid, get_full_information],
    "Parallel Graph Computing System", model_params
)

server.port = 8521

server.launch(open_browser=True)
