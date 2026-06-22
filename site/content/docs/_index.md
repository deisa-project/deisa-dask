<a id="dask"></a>

# dask

<a id="dask.version"></a>

## version

<a id="dask.PackageNotFoundError"></a>

## PackageNotFoundError

<a id="dask.Bridge"></a>

## Bridge

<a id="dask.Deisa"></a>

## Deisa

<a id="dask.get_connection_info"></a>

## get\_connection\_info

<a id="dask.handshake"></a>

# dask.handshake

<a id="dask.handshake.logging"></a>

## logging

<a id="dask.handshake.Optional"></a>

## Optional

<a id="dask.handshake.Future"></a>

## Future

<a id="dask.handshake.get_client"></a>

## get\_client

<a id="dask.handshake.Event"></a>

## Event

<a id="dask.handshake.Client"></a>

## Client

<a id="dask.handshake.KEY_PREFIX"></a>

## KEY\_PREFIX

<a id="dask.handshake.logger"></a>

#### logger

<a id="dask.handshake.Handshake"></a>

## Handshake Objects

```python
class Handshake()
```

<a id="dask.handshake.Handshake.HandshakeActor"></a>

## HandshakeActor Objects

```python
class HandshakeActor()
```

<a id="dask.handshake.Handshake.HandshakeActor.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="dask.handshake.Handshake.HandshakeActor.set_bridges_ready"></a>

#### set\_bridges\_ready

```python
def set_bridges_ready(nb_bridges: int, arrays_metadata: dict) -> None
```

<a id="dask.handshake.Handshake.HandshakeActor.set_bridges_done"></a>

#### set\_bridges\_done

```python
def set_bridges_done(timestep: int) -> None
```

<a id="dask.handshake.Handshake.HandshakeActor.set_analytics_ready"></a>

#### set\_analytics\_ready

```python
def set_analytics_ready(feedback_queue_size: int) -> None
```

<a id="dask.handshake.Handshake.HandshakeActor.get_arrays_metadata"></a>

#### get\_arrays\_metadata

```python
def get_arrays_metadata() -> dict | Future
```

<a id="dask.handshake.Handshake.HandshakeActor.get_nb_bridges"></a>

#### get\_nb\_bridges

```python
def get_nb_bridges() -> int | Future
```

<a id="dask.handshake.Handshake.HandshakeActor.get_feedback_queue_size"></a>

#### get\_feedback\_queue\_size

```python
def get_feedback_queue_size() -> int | Future
```

<a id="dask.handshake.Handshake.__init__"></a>

#### \_\_init\_\_

```python
def __init__(client: Optional[Client] = None)
```

<a id="dask.handshake.Handshake.all_bridges_ready"></a>

#### all\_bridges\_ready

```python
def all_bridges_ready(nb_bridge: int,
                      arrays_metadata: dict,
                      wait_for_go=True) -> None
```

Bridge must wait for analytics to be ready.

<a id="dask.handshake.Handshake.deisa_ready"></a>

#### deisa\_ready

```python
def deisa_ready(feedback_queue_size: int = 1024,
                wait_for_go=True,
                *args,
                **kwargs) -> None
```

When analytics is ready, notify all Bridges

<a id="dask.handshake.Handshake.get_arrays_metadata"></a>

#### get\_arrays\_metadata

```python
def get_arrays_metadata() -> dict
```

<a id="dask.handshake.Handshake.get_feedback_queue_size"></a>

#### get\_feedback\_queue\_size

```python
def get_feedback_queue_size() -> int
```

<a id="dask.handshake.Handshake.get_nb_bridges"></a>

#### get\_nb\_bridges

```python
def get_nb_bridges() -> int
```

<a id="dask.handshake.Handshake.wait_for_bridges"></a>

#### wait\_for\_bridges

```python
def wait_for_bridges()
```

<a id="dask.handshake.Handshake.set_bridges_done"></a>

#### set\_bridges\_done

```python
def set_bridges_done(timestep: int)
```

<a id="dask.handshake.Handshake.wait_for_bridges_to_finish"></a>

#### wait\_for\_bridges\_to\_finish

```python
def wait_for_bridges_to_finish()
```

<a id="dask.utils"></a>

# dask.utils

<a id="dask.utils.logging"></a>

## logging

<a id="dask.utils.os"></a>

## os

<a id="dask.utils.da"></a>

## da

<a id="dask.utils.ICommunicator"></a>

## ICommunicator

<a id="dask.utils.DeisaArray"></a>

## DeisaArray

<a id="dask.utils.Client"></a>

## Client

<a id="dask.utils.Lock"></a>

## Lock

<a id="dask.utils.Variable"></a>

## Variable

<a id="dask.utils.logger"></a>

#### logger

<a id="dask.utils.get_client"></a>

#### get\_client

```python
def get_client(*args, **kwargs)
```

<a id="dask.utils.get_mpi_comm_world"></a>

#### get\_mpi\_comm\_world

```python
def get_mpi_comm_world(cart_coord_dims: int = 1) -> ICommunicator
```

Computes and returns an MPI Cartesian communicator based on the number of desired
Cartesian coordinate dimensions.

This function uses the MPI library to calculate and create a Cartesian communicator
from the global MPI communicator (MPI.COMM_WORLD). The dimensions of the Cartesian coordinate
grid are determined dynamically based on the size of the Cartesian coordinate dimensions
requested by the user and the size of the MPI communicator.

:param cart_coord_dims: Number of Cartesian coordinate dimensions used to compute the
    grid layout for the Cartesian communicator. Default is 1.
:type cart_coord_dims: int
:return: A new Cartesian communicator created from the MPI world communicator based
    on the computed dimensions.
:rtype: mpi4py.MPI.Cartcomm

<a id="dask.utils.get_connection_info"></a>

#### get\_connection\_info

```python
def get_connection_info(dask_scheduler_address: str | Client, *args,
                        **kwargs) -> Client
```

<a id="dask.utils.build_deisa_array"></a>

#### build\_deisa\_array

```python
def build_deisa_array(darr: da.Array, timestep: int) -> DeisaArray
```

<a id="dask.bridge"></a>

# dask.bridge

<a id="dask.bridge.asyncio"></a>

## asyncio

<a id="dask.bridge.logging"></a>

## logging

<a id="dask.bridge.sys"></a>

## sys

<a id="dask.bridge.uuid"></a>

## uuid

<a id="dask.bridge.deque"></a>

## deque

<a id="dask.bridge.defaultdict"></a>

## defaultdict

<a id="dask.bridge.Number"></a>

## Number

<a id="dask.bridge.Any"></a>

## Any

<a id="dask.bridge.Iterator"></a>

## Iterator

<a id="dask.bridge.List"></a>

## List

<a id="dask.bridge.Dict"></a>

## Dict

<a id="dask.bridge.Optional"></a>

## Optional

<a id="dask.bridge.Union"></a>

## Union

<a id="dask.bridge.Deque"></a>

## Deque

<a id="dask.bridge.np"></a>

## np

<a id="dask.bridge.tokenize"></a>

## tokenize

<a id="dask.bridge.validate_arrays_metadata"></a>

## validate\_arrays\_metadata

<a id="dask.bridge.IBridge"></a>

## IBridge

<a id="dask.bridge.ICommunicator"></a>

## ICommunicator

<a id="dask.bridge.Queue"></a>

## Queue

<a id="dask.bridge.Client"></a>

## Client

<a id="dask.bridge.to_serialize"></a>

## to\_serialize

<a id="dask.bridge.scatter_to_workers"></a>

## scatter\_to\_workers

<a id="dask.bridge.valmap"></a>

## valmap

<a id="dask.bridge.KEY_PREFIX"></a>

## KEY\_PREFIX

<a id="dask.bridge.FEEDBACK_QUEUE_PREFIX"></a>

## FEEDBACK\_QUEUE\_PREFIX

<a id="dask.bridge.CLIENT_KEY"></a>

## CLIENT\_KEY

<a id="dask.bridge.Handshake"></a>

## Handshake

<a id="dask.bridge.get_client"></a>

## get\_client

<a id="dask.bridge.logger"></a>

#### logger

<a id="dask.bridge.Bridge"></a>

## Bridge Objects

```python
class Bridge(IBridge)
```

<a id="dask.bridge.Bridge.__init__"></a>

#### \_\_init\_\_

```python
def __init__(comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args,
             **kwargs)
```

Initializes an instance of the class, setting up communication, metadata validation,
client connection (for id=0), workers initialization, and handshake configuration for the bridge.

:param comm: An instance of ICommunicator facilitating communication between processes.
:param arrays_metadata: Dictionary containing metadata for arrays, validated during initialization.
    eg: arrays_metadata = {
            'temperature': {
                'global_shape': [20, 20],
                'chunk_shape': [10, 10],
                'chunk_position': [0, 0]
            }
            'pressure': {
                'global_shape': [20, 20],
                'chunk_shape': [10, 10],
                'chunk_position': [0, 0]
            }
:type arrays_metadata: Dict[str, Dict]
:param args: Additional positional arguments for the initialization.
:param kwargs: Additional keyword arguments for the initialization. Can include
    configuration parameters like timeout used during client setup.

<a id="dask.bridge.Bridge.__del__"></a>

#### \_\_del\_\_

```python
def __del__()
```

Cleans up resources used by the object before it gets destroyed.

This method is called when the object is about to be destroyed and ensures that
any required cleanup operations are performed. The `close` method is invoked
with a timestep set to the maximum possible value.

:param timestep: A value to specify the timestep for cleanup operations. This
    is set to the maximum integer value available in Python.
:type timestep: int

<a id="dask.bridge.Bridge.close"></a>

#### close

```python
def close(timestep: int) -> None
```

Attempts to close the bridge connection. This involves ensuring the bridge is properly cleaned up,
orchestrating communication with other bridges, and notifying the analytics of the closure.
The method ensures that it is only executed once during the lifecycle of the instance.

:param timestep: The current timestep associated with the closure action.
:type timestep: int
:return: None

<a id="dask.bridge.Bridge.send"></a>

#### send

```python
def send(array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs)
```

Handles the distribution of the given data chunk to workers in the Dask cluster.
This method sends the data directly to the workers.

:param array_name: The name of the data array being sent as a string.
    This should match what is defined in the Bridge arrays_metadata.
:param chunk: A numpy ndarray containing the data chunk to be sent to the workers.
:param timestep: The current timestep associated to the sent data chunk.
:param args: Additional positional arguments if required by the method implementation.
:param kwargs: Additional keyword arguments for optional configurations.
    Supported keys include:
    - `update_workers` (bool): If True, updates the workers' list by retrieving it from the scheduler.
    - `filter_workers` (callable): A function that filters the available workers
      and returns a list of worker names. Must return a non-empty list of strings.

:return: None. All operations are internal and side effects include sending data
    to workers, logging the event, and synchronizing worker states.

<a id="dask.bridge.Bridge.get"></a>

#### get

```python
def get(key: str,
        timestep: Optional[int] = None,
        default: Any = None) -> Optional[Union[Deque, Any]]
```

Retrieve an element associated with a specific key and optional timestep from a feedback queue.
If a queue for the key does not exist, it initializes the queue for the specified key.

:param key: The unique identifier for the feedback queue.
:type key: str
:param timestep: An optional specific timestep to look for. If None, returns the entire deque.
:type timestep: Optional[int]
:param default: The default value to return if the specified timestep is not found.
:type default: Any
:return: The element associated with the specified timestep if found, the entire deque if no
    timestep is specified, or the default value if the timestep is not found.
:rtype: Optional[Union[Deque, Any]]

<a id="dask.constants"></a>

# dask.constants

<a id="dask.constants.Final"></a>

## Final

<a id="dask.constants.FEEDBACK_QUEUE_PREFIX"></a>

#### FEEDBACK\_QUEUE\_PREFIX

<a id="dask.constants.CALLBACK_PREFIX"></a>

#### CALLBACK\_PREFIX

<a id="dask.constants.DEFAULT_SLIDING_WINDOW_SIZE"></a>

#### DEFAULT\_SLIDING\_WINDOW\_SIZE

<a id="dask.constants.CLIENT_KEY"></a>

#### CLIENT\_KEY

<a id="dask.constants.KEY_PREFIX"></a>

#### KEY\_PREFIX

<a id="dask.__version__"></a>

# dask.\_\_version\_\_

<a id="dask.__version__.__version__"></a>

#### \_\_version\_\_

<a id="dask.communicator"></a>

# dask.communicator

<a id="dask.communicator.asyncio"></a>

## asyncio

<a id="dask.communicator.logging"></a>

## logging

<a id="dask.communicator.threading"></a>

## threading

<a id="dask.communicator.uuid"></a>

## uuid

<a id="dask.communicator.Optional"></a>

## Optional

<a id="dask.communicator.np"></a>

## np

<a id="dask.communicator.ICommunicator"></a>

## ICommunicator

<a id="dask.communicator.Client"></a>

## Client

<a id="dask.communicator.logger"></a>

#### logger

<a id="dask.communicator.is_mpi_comm"></a>

#### is\_mpi\_comm

```python
def is_mpi_comm(comm)
```

<a id="dask.communicator.is_running_on_mpi"></a>

#### is\_running\_on\_mpi

```python
def is_running_on_mpi()
```

<a id="dask.communicator.resolve_comm"></a>

#### resolve\_comm

```python
def resolve_comm(comm,
                 cart_coord_dims=1,
                 use_mpi_if_available=True,
                 *args,
                 **kwargs) -> ICommunicator
```

handle 3 cases to resolve comm:
- if comm is None: use_mpi_if_available or no MPI
- if comm is an MPI Comm: use it

<a id="dask.communicator.CommActor"></a>

## CommActor Objects

```python
class CommActor()
```

<a id="dask.communicator.CommActor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(size: int)
```

<a id="dask.communicator.CommActor.register"></a>

#### register

```python
def register(cid: str) -> int
```

<a id="dask.communicator.CommActor.gather_add"></a>

#### gather\_add

```python
def gather_add(seq: int, rank: int, value)
```

<a id="dask.communicator.CommActor.gather_ready"></a>

#### gather\_ready

```python
def gather_ready(seq: int)
```

<a id="dask.communicator.CommActor.gather_get"></a>

#### gather\_get

```python
def gather_get(seq: int)
```

<a id="dask.communicator.CommActor.get_coords"></a>

#### get\_coords

```python
def get_coords(rank: int, dims)
```

<a id="dask.communicator.CommActor.bcast_set"></a>

#### bcast\_set

```python
def bcast_set(obj)
```

<a id="dask.communicator.CommActor.bcast_ready"></a>

#### bcast\_ready

```python
def bcast_ready()
```

<a id="dask.communicator.CommActor.bcast_get"></a>

#### bcast\_get

```python
def bcast_get()
```

<a id="dask.communicator.CommActor.cleanup"></a>

#### cleanup

```python
def cleanup()
```

<a id="dask.communicator.CommState"></a>

## CommState Objects

```python
class CommState()
```

<a id="dask.communicator.CommState.__init__"></a>

#### \_\_init\_\_

```python
def __init__(scheduler, size: int)
```

<a id="dask.communicator.CommState.register"></a>

#### register

```python
def register(cid: str) -> int
```

<a id="dask.communicator.CommState.get_size"></a>

#### get\_size

```python
def get_size() -> int
```

<a id="dask.communicator.CommState.gather"></a>

#### gather

```python
async def gather(seq: str, rank: int, data)
```

<a id="dask.communicator.CommState.bcast"></a>

#### bcast

```python
async def bcast(seq: str, rank: int, obj, root: int)
```

<a id="dask.communicator.setup_comm"></a>

#### setup\_comm

```python
def setup_comm(dask_scheduler, size: int)
```

<a id="dask.communicator.CommClient"></a>

## CommClient Objects

```python
class CommClient()
```

<a id="dask.communicator.CommClient.__init__"></a>

#### \_\_init\_\_

```python
def __init__(comm_state_rpc, client: Optional[Client] = None, *args, **kwargs)
```

<a id="dask.communicator.CommClient.Get_rank_async"></a>

#### Get\_rank\_async

```python
async def Get_rank_async()
```

<a id="dask.communicator.CommClient.Get_rank"></a>

#### Get\_rank

```python
def Get_rank()
```

<a id="dask.communicator.CommClient.Get_size"></a>

#### Get\_size

```python
def Get_size() -> int
```

<a id="dask.communicator.CommClient.gather"></a>

#### gather

```python
def gather(data, root=0)
```

<a id="dask.communicator.CommClient.bcast"></a>

#### bcast

```python
def bcast(obj, root=0)
```

<a id="dask.deisa"></a>

# dask.deisa

<a id="dask.deisa.asyncio"></a>

## asyncio

<a id="dask.deisa.collections"></a>

## collections

<a id="dask.deisa.logging"></a>

## logging

<a id="dask.deisa.threading"></a>

## threading

<a id="dask.deisa.time"></a>

## time

<a id="dask.deisa.weakref"></a>

## weakref

<a id="dask.deisa.Callable"></a>

## Callable

<a id="dask.deisa.Union"></a>

## Union

<a id="dask.deisa.Tuple"></a>

## Tuple

<a id="dask.deisa.List"></a>

## List

<a id="dask.deisa.Literal"></a>

## Literal

<a id="dask.deisa.Any"></a>

## Any

<a id="dask.deisa.Dict"></a>

## Dict

<a id="dask.deisa.Set"></a>

## Set

<a id="dask.deisa.Collection"></a>

## Collection

<a id="dask.deisa.da"></a>

## da

<a id="dask.deisa.np"></a>

## np

<a id="dask.deisa.CallbackArgs"></a>

## CallbackArgs

<a id="dask.deisa.Window"></a>

## Window

<a id="dask.deisa.IDeisa"></a>

## IDeisa

<a id="dask.deisa.Client"></a>

## Client

<a id="dask.deisa.Future"></a>

## Future

<a id="dask.deisa.Queue"></a>

## Queue

<a id="dask.deisa.Event"></a>

## Event

<a id="dask.deisa.KEY_PREFIX"></a>

## KEY\_PREFIX

<a id="dask.deisa.CALLBACK_PREFIX"></a>

## CALLBACK\_PREFIX

<a id="dask.deisa.CLIENT_KEY"></a>

## CLIENT\_KEY

<a id="dask.deisa.FEEDBACK_QUEUE_PREFIX"></a>

## FEEDBACK\_QUEUE\_PREFIX

<a id="dask.deisa.DEFAULT_SLIDING_WINDOW_SIZE"></a>

## DEFAULT\_SLIDING\_WINDOW\_SIZE

<a id="dask.deisa.Handshake"></a>

## Handshake

<a id="dask.deisa.get_client"></a>

## get\_client

<a id="dask.deisa.build_deisa_array"></a>

## build\_deisa\_array

<a id="dask.deisa.logger"></a>

#### logger

<a id="dask.deisa.Deisa"></a>

## Deisa Objects

```python
class Deisa(IDeisa)
```

<a id="dask.deisa.Deisa.Callback_args"></a>

#### Callback\_args

array_name, window_size

<a id="dask.deisa.Deisa.Callback_id"></a>

#### Callback\_id

<a id="dask.deisa.Deisa.__init__"></a>

#### \_\_init\_\_

```python
def __init__(feedback_queue_size: int = 1024, *args, **kwargs) -> None
```

Initializes a class instance, configuring the client and setting up the necessary
infrastructure for interactions. This includes setting up the necessary feedback
queue length, performing handshake operations with the client, and initializing
various metadata structures.

:param feedback_queue_size: The maximum size of the feedback queue. Defaults to 1024.
:type feedback_queue_size: int
:param args: Additional positional arguments passed to the initializer.
:type args: tuple
:param kwargs: Additional keyword arguments passed to the initializer.
:type kwargs: dict

<a id="dask.deisa.Deisa.__del__"></a>

#### \_\_del\_\_

```python
def __del__()
```

<a id="dask.deisa.Deisa.register"></a>

#### register

```python
def register(*callback_args: CallbackArgs,
             exception_handler: IDeisa.
             ExceptionHandler = __default_exception_handler,
             when: Literal['AND', 'OR'] = 'AND') -> Callable
```

Registers a callback function with specific arguments, exception handling, and conditional execution criteria.

This function acts as a decorator that allows you to register a callback with
parameters provided through ``callback_args``. It also handles exceptions using the
``exception_handler`` and defines the execution rules with ``when`` parameter.

Supports:
Default window size is 1.
@deisa.register("arr1")                             # default window size
@deisa.register("arr1", "arr2")                     # two arrays, default window size
@deisa.register(Window("arr1"))                     # default window size
@deisa.register(Window("arr1", 2))                  # window size 2
@deisa.register(Window("arr1", 2), Window("arr2", 5))   # window size 2 for arr1 and 5 for arr2
@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3") # window size 2 for arr1 and 5 for arr2, default window size for arr3

:param callback_args: Variable-length arguments representing callback-specific parameters.
:param exception_handler: Optional exception handler to manage errors during callback execution.
    Defaults to ``__default_exception_handler``.
:param when: Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
    Defaults to 'AND'.
:return: A callable that wraps the provided callback with the configured parameters and logic.
:rtype: Callable

<a id="dask.deisa.Deisa.register_callback"></a>

#### register\_callback

```python
def register_callback(callback: IDeisa.Callback,
                      *callback_args: CallbackArgs,
                      exception_handler: IDeisa.
                      ExceptionHandler = __default_exception_handler,
                      when: Literal['AND', 'OR'] = 'AND') -> Callable
```

Registers a callback function with specific arguments, exception handling, and conditional execution criteria.

This function allows you to register a callback with parameters provided through
``callback_args``. It also handles exceptions using the ``exception_handler``
and defines the execution rules with ``when`` parameter.

Supports:
Default window size is 1.
@deisa.register("arr1")                             # default window size
@deisa.register("arr1", "arr2")                     # two arrays, default window size
@deisa.register(Window("arr1"))                     # default window size
@deisa.register(Window("arr1", 2))                  # window size 2
@deisa.register(Window("arr1", 2), Window("arr2", 5))   # window size 2 for arr1 and 5 for arr2
@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3") # window size 2 for arr1 and 5 for arr2, default window size for arr3

:param callback: Callback function to register.
:param callback_args: Variable-length arguments representing callback-specific parameters.
:param exception_handler: Optional exception handler to manage errors during callback execution.
    Defaults to ``__default_exception_handler``.
:param when: Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
    Defaults to 'AND'.
:return: A callable that wraps the provided callback with the configured parameters and logic.
:rtype: Callable

<a id="dask.deisa.Deisa.unregister_callback"></a>

#### unregister\_callback

```python
def unregister_callback(callback_id: Callback_id) -> None
```

<a id="dask.deisa.Deisa.set"></a>

#### set

```python
def set(key: str, value: Any, timestep: int) -> None
```

Sets a value in a queue for a given key, associating it with a specific timestep. This action is
intended to store feedback or other time-specific data for the provided key.

:param key: The identifier for which the value is to be set.
:type key: str
:param value: The value to be stored, associated with the key and timestep.
:type value: Any
:param timestep: The timestamp that corresponds to when the value is set.
:type timestep: int
:return: None

<a id="dask.deisa.Deisa.execute_callbacks"></a>

#### execute\_callbacks

```python
def execute_callbacks() -> None
```

Executes a series of callbacks and waits for necessary processes to finish.

This method handles the execution of callbacks related to bridges and their
completion while also ensuring the orchestration of subsequent tasks. It is
responsible for unblocking bridges and waiting for dependencies to signal
completion.

:param self: The instance of the class invoking this method.

:return: None

<a id="dask.deisa.Deisa.run_task_sync"></a>

#### run\_task\_sync

```python
@staticmethod
def run_task_sync(coro, loop)
```

<a id="dask.deisa.Deisa.make_topic"></a>

#### make\_topic

```python
@staticmethod
def make_topic(arrays, when) -> str
```

