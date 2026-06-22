<a id="dask"></a>

# dask

<a id="dask.handshake"></a>

# dask.handshake

<a id="dask.handshake.Handshake"></a>

## Handshake Objects

```python
class Handshake()
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

<a id="dask.utils"></a>

# dask.utils

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

**Arguments**:

- `cart_coord_dims` (`int`): Number of Cartesian coordinate dimensions used to compute the
grid layout for the Cartesian communicator. Default is 1.

**Returns**:

`mpi4py.MPI.Cartcomm`: A new Cartesian communicator created from the MPI world communicator based
on the computed dimensions.

<a id="dask.bridge"></a>

# dask.bridge

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

**Arguments**:

- `comm`: An instance of ICommunicator facilitating communication between processes.
- `arrays_metadata` (`Dict[str, Dict]`): Dictionary containing metadata for arrays, validated during initialization.
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
- `args`: Additional positional arguments for the initialization.
- `kwargs`: Additional keyword arguments for the initialization. Can include
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

**Arguments**:

- `timestep` (`int`): A value to specify the timestep for cleanup operations. This
is set to the maximum integer value available in Python.

<a id="dask.bridge.Bridge.close"></a>

#### close

```python
def close(timestep: int) -> None
```

Attempts to close the bridge connection. This involves ensuring the bridge is properly cleaned up,

orchestrating communication with other bridges, and notifying the analytics of the closure.
The method ensures that it is only executed once during the lifecycle of the instance.

**Arguments**:

- `timestep` (`int`): The current timestep associated with the closure action.

**Returns**:

None

<a id="dask.bridge.Bridge.send"></a>

#### send

```python
def send(array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs)
```

Handles the distribution of the given data chunk to workers in the Dask cluster.

This method sends the data directly to the workers.

**Arguments**:

- `array_name`: The name of the data array being sent as a string.
This should match what is defined in the Bridge arrays_metadata.
- `chunk`: A numpy ndarray containing the data chunk to be sent to the workers.
- `timestep`: The current timestep associated to the sent data chunk.
- `args`: Additional positional arguments if required by the method implementation.
- `kwargs`: Additional keyword arguments for optional configurations.
Supported keys include:
- `update_workers` (bool): If True, updates the workers' list by retrieving it from the scheduler.
- `filter_workers` (callable): A function that filters the available workers
  and returns a list of worker names. Must return a non-empty list of strings.

**Returns**:

None. All operations are internal and side effects include sending data
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

**Arguments**:

- `key` (`str`): The unique identifier for the feedback queue.
- `timestep` (`Optional[int]`): An optional specific timestep to look for. If None, returns the entire deque.
- `default` (`Any`): The default value to return if the specified timestep is not found.

**Returns**:

`Optional[Union[Deque, Any]]`: The element associated with the specified timestep if found, the entire deque if no
timestep is specified, or the default value if the timestep is not found.

<a id="dask.constants"></a>

# dask.constants

<a id="dask.__version__"></a>

# dask.\_\_version\_\_

<a id="dask.communicator"></a>

# dask.communicator

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

<a id="dask.deisa"></a>

# dask.deisa

<a id="dask.deisa.Deisa"></a>

## Deisa Objects

```python
class Deisa(IDeisa)
```

<a id="dask.deisa.Deisa.Callback_args"></a>

#### Callback\_args

array_name, window_size

<a id="dask.deisa.Deisa.__init__"></a>

#### \_\_init\_\_

```python
def __init__(feedback_queue_size: int = 1024, *args, **kwargs) -> None
```

Initializes a class instance, configuring the client and setting up the necessary

infrastructure for interactions. This includes setting up the necessary feedback
queue length, performing handshake operations with the client, and initializing
various metadata structures.

**Arguments**:

- `feedback_queue_size` (`int`): The maximum size of the feedback queue. Defaults to 1024.
- `args` (`tuple`): Additional positional arguments passed to the initializer.
- `kwargs` (`dict`): Additional keyword arguments passed to the initializer.

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


Supports:  
Default window size is 1.  
`@deisa.register("arr1")`                             # default window size  
`@deisa.register("arr1", "arr2")`                     # two arrays, default window size  
`@deisa.register(Window("arr1"))`                     # default window size  
`@deisa.register(Window("arr1", 2))`                  # window size 2  
`@deisa.register(Window("arr1", 2), Window("arr2", 5))`   # window size 2 for arr1 and 5 for arr2  
`@deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3")` # window size 2 for arr1 and 5 for arr2, default window size for arr3

**Arguments**:

- `callback_args`: Variable-length arguments representing callback-specific parameters.
- `exception_handler`: Optional exception handler to manage errors during callback execution.
Defaults to ``__default_exception_handler``.
- `when`: Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
Defaults to 'AND'.

**Returns**:

`Callable`: A callable that wraps the provided callback with the configured parameters and logic.

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
`@deisa.register("arr1")`                              # default window size  
`@deisa.register("arr1","arr2")`                      # two arrays, default window size  
`@deisa.register(Window("arr1"))`                      # default window size  
`@deisa.register(Window("arr1",2))`                   # window size 2  
`@deisa.register(Window("arr1",2),Window("arr2",5))`    # window size 2 for arr1 and 5 for arr2  
`@deisa.register(Window("arr1",2),Window("arr2",5),"arr3")`  # window size 2 for arr1 and 5 for arr2, default window size for arr3

**Arguments**:

- `callback`: Callback function to register.
- `callback_args`: Variable-length arguments representing callback-specific parameters.
- `exception_handler`: Optional exception handler to manage errors during callback execution.
Defaults to ``__default_exception_handler``.
- `when`: Specifies the conditional logic for triggering the callback. Can be 'AND' or 'OR'.
Defaults to 'AND'.

**Returns**:

`Callable`: A callable that wraps the provided callback with the configured parameters and logic.

<a id="dask.deisa.Deisa.set"></a>

#### set

```python
def set(key: str, value: Any, timestep: int) -> None
```

Sets a value in a queue for a given key, associating it with a specific timestep. This action is

intended to store feedback or other time-specific data for the provided key.

**Arguments**:

- `key` (`str`): The identifier for which the value is to be set.
- `value` (`Any`): The value to be stored, associated with the key and timestep.
- `timestep` (`int`): The timestamp that corresponds to when the value is set.

**Returns**:

None

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

**Arguments**:

- `self`: The instance of the class invoking this method.

**Returns**:

None

