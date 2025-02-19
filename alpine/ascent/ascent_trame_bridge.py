import conduit
import ascent.mpi

import numpy as np
from multiprocessing.managers import BaseManager

from mpi4py import MPI


class QueueManager(BaseManager):
    pass


def main():
    # obtain a mpi4py mpi comm object
    comm = MPI.Comm.f2py(ascent_mpi_comm_id())

    # get task id and number of total tasks
    task_id = comm.Get_rank()
    num_tasks = comm.Get_size()

    # run Trame tasks
    interactive = np.array([False], bool)
    update_data = None
    if task_id == 0:
        update_data = executeMainTask(task_id, num_tasks, comm)
    else:
        executeDependentTask(task_id, num_tasks, comm)
    
    # broadcast updates to all ranks
    update_data = comm.bcast(update_data, root=0)

    # TODO: pass updates to Ascent callback
    #update_node = conduit.Node()
    #update_node['task_id'] = task_id
    #output = conduit.Node()
    #ascent.mpi.execute_callback('steeringCallback', update_node, output)


def executeMainTask(task_id, num_tasks, comm):
    interactive = False
    update_data = {}    

    # attempt to connect to Trame queue manager
    QueueManager.register('get_data_queue')
    QueueManager.register('get_signal_queue')
    mgr = QueueManager(address=('127.0.0.1', 8000 + task_id), authkey=b'ascent-trame')
    try:
        mgr.connect()
        interactive = True
    except:
        mgr = None

    if interactive:
        queue_data = mgr.get_data_queue()
        queue_signal = mgr.get_signal_queue()
        
        # send particle data to Trame app
        sendDataToTrameApp(queue_data)

        # get steering updates from Trame
        update_data = queue_signal.get()

    return update_data


def executeDependentTask(task_id, num_tasks, comm):
    interactive = False

    # attempt to connect to Trame queue manager
    QueueManager.register('get_data_queue')
    QueueManager.register('get_signal_queue')
    mgr = QueueManager(address=('127.0.0.1', 8000 + task_id), authkey=b'ascent-trame')
    try:
        mgr.connect()
        interactive = True
    except:
        mgr = None

    if interactive:
        queue_data = mgr.get_data_queue()

        # send particle data to Trame app
        sendDataToTrameApp(queue_data)


def sendDataToTrameApp(queue_data):
    mesh_data = ascent_data().child(0)

    # particle positions
    x_coords = mesh_data['coordsets/particle_coords/values/x'].astype(dtype=np.float32)
    y_coords = mesh_data['coordsets/particle_coords/values/y'].astype(dtype=np.float32)
    z_coords = mesh_data['coordsets/particle_coords/values/z'].astype(dtype=np.float32)
    particles = np.column_stack([x_coords, y_coords, z_coords])
    # particle velocities
    x_vel = mesh_data['fields/particle_velocity/values/u'].astype(dtype=np.float32)
    y_vel = mesh_data['fields/particle_velocity/values/v'].astype(dtype=np.float32)
    z_vel = mesh_data['fields/particle_velocity/values/w'].astype(dtype=np.float32)
    velocities = np.column_stack([x_vel, y_vel, z_vel])
    # particle charges
    charges = mesh_data['fields/particle_charge/values'].astype(dtype=np.float32)

    queue_data.put({'coordinates': particles, 'velocity': velocities, 'charge': charges})

"""
def gatherParticleDataOnSubsetOfTasks(task_id, num_tasks, comm):
    # Use MPI_Comm_split(color, key)
    #   color: subgroup ID (all tasks with same color become part of the same communicator
    #   key: new rank numbers assigned based on order of key
    pass
"""

if __name__ == "__main__":
    main()

