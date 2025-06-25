import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def print_x(i):
    return f"This is new output: {i**2}"

@ray.remote
class PrintActor:
    def __init__(self, *args):
        self.args = args
    
    def execture_print(self, i):
        return print_x(i)

def create_print_engines(num_engines, *args):
    print_engines = []
    bundles = [{"CPU": 1} 
               for _ in range(num_engines)]
    shared_pg = placement_group(bundles, strategy="PACK")
    ray.get(shared_pg.ready())

    for i in range(num_engines):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=shared_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=i,
            )
        print_engines.append(
            PrintActor.options(
                scheduling_strategy= scheduling_strategy
            ).remote(
                *args
            )
        )
    return print_engines

if __name__ == "__main__":
    ray.init()
    print_engines = create_print_engines(4)
    results = [engine.execture_print.remote(i) for i, engine in enumerate(print_engines)]
    print(ray.get(results))