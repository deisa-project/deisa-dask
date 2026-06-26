from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from deisa.dask import Deisa

deisa = Deisa()


@deisa.register("temperature")
def callback(temperatures):

    latest_temperature = temperatures[-1]
    # compute the sum of the latest temperature
    sum = latest_temperature.sum().compute()
    print(f"latest temperature t={latest_temperature.timestep}, sum={sum}", flush=True)

    # plot the latest temperature
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(latest_temperature, cmap="viridis", interpolation="none")
    fig.savefig(f"heat-{latest_temperature.timestep}.png")
    fig.clear()


# wait for all tasks and for simulation to finish
deisa.execute_callbacks()
