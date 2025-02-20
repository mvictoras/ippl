import sys
import asyncio
import math
import time
import random
import io
from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout
import numpy as np
import pynari as anari
from PIL import Image
from mpi4py import MPI

CUBE_DIM = 16

class QueueManager(BaseManager):
    pass


def main():
    # initialize MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    # create ANARI view
    view = AnariView(mpi_rank, mpi_size, comm)    

    # create queues from Ascent data
    queue_data = Queue()
    queue_signal = Queue()

    # start Queue Manager in new thread
    queue_mgr_thread = Process(target=runQueueManager, args=(8000 + mpi_rank, queue_data, queue_signal))
    queue_mgr_thread.daemon = True
    queue_mgr_thread.start()

    # create queues for Python app
    state_queue = Queue()
    update_queue = Queue()

    # create thread for bridging data from Ascent to Python app
    ascent_bridge_thread = Process(target=runAscentBridge, args=(queue_data, queue_signal, state_queue, update_queue))
    ascent_bridge_thread.daemon = True
    ascent_bridge_thread.start()

    runAnariApp(mpi_rank, mpi_size, comm, view, state_queue, update_queue)


def runAnariApp(mpi_rank, mpi_size, comm, view, state_queue, update_queue):
    # main task initializes Trame server
    if mpi_rank == 0:
        setupTrameServer(view, state_queue, update_queue)
    # other tasks wait for signal to rerender or quit
    else:
        finished = False
        signal = np.empty(3, dtype=np.int16)
        while not finished:
            comm.Bcast((signal, 3, MPI.INT16_T), root=0)

            if signal[0] == 0:    # quit
                finished = True
            elif signal[0] == 1:  # rerender
                view.render()
            elif signal[0] == 2:  # resize
                view.setRenderSize(int(signal[1]), int(signal[2]))
            elif signal[0] == 3:  # rotate camera
                view.rotateCamera(int(signal[1]), int(signal[2]))
            elif signal[0] == 4:  # new data from Ascent
                state_data = state_queue.get()
                view.updateData(state_data)
                update_queue.put({}) # TODO: check for accuracy


def runAscentBridge(queue_data, queue_signal, state_queue, update_queue):
    while True:
        print('waiting on data... ', end='')
        sim_data = queue_data.get()
        print(f'received!')

        state_queue.put(sim_data)
        updates = update_queue.get()

        queue_signal.put(updates)


def runQueueManager(port, queue_data, queue_signal):
    # register queues with Queue Manager
    QueueManager.register('get_data_queue', callable=lambda:queue_data)
    QueueManager.register('get_signal_queue', callable=lambda:queue_signal)

    # create Queue Manager
    mgr = QueueManager(address=('127.0.0.1', port), authkey=b'ascent-trame')

    # start Queue Manager server
    server = mgr.get_server()
    server.serve_forever()


async def mainTaskCheckForStateUpdates(state, state_queue, update_queue, view, view_handler):
    while True:
        try:
            state_data = state_queue.get(block=False)

            view.triggerReadNewData()
            view.updateData(state_data)
            view.triggerRender()
            view_handler.pushFrame()

            state.connected = True
            if state.enable_steering:
                state.allow_submit = True

            state.flush()

            if not state.enable_steering:
                update_queue.put({})
        except:
            pass
        await asyncio.sleep(0)


def setupTrameServer(view, state_queue, update_queue):
    # set up Trame application
    server = get_server(client_type='vue2')
    state = server.state
    ctrl = server.controller

    # register RCA view with Trame controller
    view_handler = None
    @ctrl.add('on_server_ready')
    def initRca(**kwargs):
        nonlocal view_handler
        view_handler = RcaViewAdapter(view, 'view')
        ctrl.rc_area_register(view_handler)
        asynchronous.create_task(mainTaskCheckForStateUpdates(state, state_queue, update_queue, view, view_handler))

    # callback for steering enabled change
    def uiStateEnableSteeringUpdate(enable_steering, **kwargs):
        if state.connected:
            state.allow_submit = enable_steering
        if not enable_steering:
            update_queue.put({})

    # callback for clicking submit button
    def submitSteeringOptions():
        steering_data = {
            'mag_field': state.mag_field,
            'threshold': state.threshold
        }
        update_queue.put(steering_data)

    #register callbacks
    state.change('enable_steering')(uiStateEnableSteeringUpdate)

    # define webpage layout
    state.allow_submit = False
    with SinglePageLayout(server) as layout:
        layout.title.set_text('Trame-ANARI')
        with layout.toolbar:
            vuetify.VDivider(vertical=True, classes='mx-2')
            vuetify.VSwitch(
                label='Enable Steering',
                v_model=('enable_steering', True),
                hide_details=True,
                dense=True
            )
            vuetify.VSpacer()
            vuetify.VSlider(
                label='Magnetic Field',
                v_model=('mag_field', 30),
                min=2,
                max=50,
                step=1,
                hide_details=True,
                dense=True
            )
            vuetify.VCol(
                '{{mag_field}}'
            )
            vuetify.VSpacer()
            vuetify.VSlider(
                label='Threshold',
                v_model=('threshold', 9.25),
                min=5.0,
                max=12.5,
                step=0.25,
                hide_details=True,
                dense=True
            )
            vuetify.VCol(
                '{{threshold.toFixed(2)}}'
            )
            vuetify.VSpacer()
            vuetify.VBtn(
                'Submit',
                color='primary',
                disabled=('!allow_submit',),
                click=submitSteeringOptions
            )
        with layout.content:
            with vuetify.VContainer(fluid=True, classes='pa-0 fill-height'):
                v = rca.RemoteControlledArea(name='view', display='image', id='rca-view')

    # start Trame server
    server.start()


# Trame RCA View Adapter
class RcaViewAdapter:
    def __init__(self, view, name):
        self._view = view
        self._streamer = None
        self._metadata = {
            'type': 'image/jpeg',
            'codec': '',
            'w': 0,
            'h': 0,
            'st': 0,
            'key': 'key'
        }
        self._mouse_pos_start = (0, 0)
        self._mouse_pos = (0, 0)
        self._mouse_down = False
        self._render_count = 0
        self._max_render_count = 64

        self.area_name = name

        asyncio.create_task(self._animate())

    def pushFrame(self):
        if self._streamer is not None:
            asynchronous.create_task(self._asyncPushFrame())

    async def _asyncPushFrame(self):
        frame_data = self._view.getFrame()
        self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)

    def _getMetadata(self):
        width, height = self._view.getSize()
        self._metadata['w'] = width
        self._metadata['h'] = height
        self._metadata['st'] = self._view.getFrameTime()
        return self._metadata

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager

    def update_size(self, origin, size):
        width = int(size.get('w', 400))
        height = int(size.get('h', 300))
        self._view.triggerResize(width, height)
        self._view.triggerRender()
        self.pushFrame()
        print(f'new size: {width}x{height}')
        sys.stdout.flush()

    def on_interaction(self, origin, event):
        event_type = event['type']
        rerender = False

        if event_type == 'LeftButtonPress':
            self._mouse_down = True
            self._mouse_pos_start = (event['x'], event['y'])
            self._mouse_pos = (event['x'], event['y'])
            #self._animate_task = asyncio.create_task(self._animate())
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], True)
        elif event_type == 'LeftButtonRelease' and self._mouse_down:
            self._mouse_down = False
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], False)
        elif event_type == 'MouseMove':
            self._mouse_pos = (event['x'], event['y'])
            rerender = self._view.onMouseMove(event['x'], event['y'])

        if rerender:
            self._view.triggerRender()
            frame_data = self._view.getFrame()
            self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)

    async def _animate(self):
        min_wait = 0.0
        while True:
            start_time = time.time()
            if self._streamer is not None and self._mouse_down:
                delta_x = self._mouse_pos[0] - self._mouse_pos_start[0]
                delta_y = self._mouse_pos[1] - self._mouse_pos_start[1]
                if delta_x != 0 or delta_y != 0:
                    self._mouse_pos_start = self._mouse_pos
                    self._view.triggerRotateCamera(delta_x, delta_y)
                    self._view.triggerRender()
                    frame_data = self._view.getFrame()
                    self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)
                    self._render_count = 1
                    min_wait = 0.0
            elif self._streamer is not None and self._render_count < self._max_render_count:
                self._view.triggerRender()
                frame_data = self._view.getFrame()
                self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)
                self._render_count += 1
                min_wait = 0.008
            duration = time.time() - start_time
            wait = max(0.033333 - duration, min_wait)
            await asyncio.sleep(wait)
        """
        while self._mouse_down:
            start_time = time.time()
            delta_x = self._mouse_pos[0] - self._mouse_pos_start[0]
            delta_y = self._mouse_pos[1] - self._mouse_pos_start[1]
            if delta_x != 0 or delta_y != 0:
                self._mouse_pos_start = self._mouse_pos
                self._view.triggerRotateCamera(delta_x, delta_y)
                self._view.triggerRender()
                frame_data = self._view.getFrame()
                self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)
            duration = time.time() - start_time
            wait = max(0.033333 - duration, 0.0)
            await asyncio.sleep(wait)
        """
# Trame custom ANARI view
class AnariView:
    def __init__(self, mpi_rank, mpi_size, comm):
        self._task_id = mpi_rank
        self._num_tasks = mpi_size
        self._mpi_comm = comm

        # store time frame is rendered at
        self._frame_time = round(time.time_ns() / 1000000)
        
        # create ANARI device
        self._device = anari.newDevice('default')

        # initial framebuffer size
        self._framebuffer_size = (512, 512)
        
        # initial camera parameters
        self._cam_theta = math.radians(-15.0)
        self._cam_phi = math.radians(90.0)
        self._cam_radius = 30.0
        #self._cam_position = (-2.5, 3.5, 7.5)
        self._cam_target = (10.0, 10.0, 10.0)
        self._cam_up = (0.0, 1.0, 0.0)
        self._fovy = math.radians(60.0)
        cam_position = self._calculateCameraPosition()

        # initial number of ray samples per pixel
        self._ray_samples = 1

        # create colormap
        colors = [
            (0.000, 0.063, 0.898),
            (0.310, 0.604, 0.980),
            (0.447, 0.851, 0.514),
            (0.859, 0.827, 0.282),
            (0.922, 0.463, 0.161),
            (0.769, 0.216, 0.090)
        ]
        self._colormap = self._makeColormap(colors, 1024)

        # add geometry to scene
        start_data = {
            'coordinates': np.array([[self._task_id, 0.0, 0.0]], dtype=np.float32),
            'velocity': np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            'charge': np.array([1.0], dtype=np.float32)
        }
        surfaces = self._createSurfaces(start_data)

        self._world = self._device.newWorld()
        self._world.setParameterArray('surface', anari.SURFACE, surfaces)
        self._world.commitParameters()

        # set up camera
        self._camera = self._device.newCamera('perspective')
        self._camera.setParameter('aspect', anari.FLOAT32, self._framebuffer_size[0] / self._framebuffer_size[1])
        self._camera.setParameter('position',anari.FLOAT32_VEC3, cam_position)
        direction = [self._cam_target[0] - cam_position[0],
                     self._cam_target[1] - cam_position[1],
                     self._cam_target[2] - cam_position[2]]
        self._camera.setParameter('direction', anari.float3, direction)
        self._camera.setParameter('up', anari.float3, self._cam_up)
        self._camera.setParameter('fovy', anari.FLOAT32, self._fovy)
        self._camera.commitParameters()

        # background gradient - dark gray to black (image 1 px wide, 2 px tall)
        bg_values = np.array(((0.25, 0.25, 0.25, 1.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float32).reshape((4, 1, 2))
        bg_gradient = self._device.newArray(anari.float4, bg_values)

        # create renderer and set background
        self._renderer = self._device.newRenderer('default')
        self._renderer.setParameter('ambientRadiance', anari.FLOAT32, 1.0)
        self._renderer.setParameter('background', anari.ARRAY, bg_gradient)
        self._renderer.setParameter('pixelSamples', anari.INT32, self._ray_samples)
        self._renderer.commitParameters()

        # create frame
        self._frame = self._device.newFrame()
        self._frame.setParameter('size', anari.uint2, self._framebuffer_size)
        self._frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_VEC4)
        self._frame.setParameter('renderer', anari.OBJECT, self._renderer)
        self._frame.setParameter('camera', anari.OBJECT, self._camera)
        self._frame.setParameter('world', anari.OBJECT, self._world)
        self._frame.commitParameters()

        # render image
        self.render()

    #
    def triggerRender(self):
        self._mpi_comm.Bcast((np.array([1, 0, 0], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req = self._mpi_comm.Ibcast((np.array([1, 0, 0], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req.wait()
        self.render()

    #
    def triggerResize(self, width, height):
        self._mpi_comm.Bcast((np.array([2, width, height], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req = self._mpi_comm.Ibcast((np.array([2, width, height], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req.wait()
        self.setRenderSize(width, height)

    #
    def triggerRotateCamera(self, delta_x, delta_y):
        self._mpi_comm.Bcast((np.array([3, delta_x, delta_y], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req = self._mpi_comm.Ibcast((np.array([3, delta_x, delta_y], dtype=np.int16), 3, MPI.INT16_T), root=0)
        #req.wait()
        self.rotateCamera(delta_x, delta_y)

    #
    def triggerReadNewData(self):
        self._mpi_comm.Bcast((np.array([4, 0, 0], dtype=np.int16), 3, MPI.INT16_T), root=0)

    # render frame
    def render(self):
        self._frame_time = round(time.time_ns() / 1000000)
        self._frame.render()

    # get width and height of frame
    def getSize(self):
        return self._framebuffer_size

    # get time frame was rendered at
    def getFrameTime(self):
        return self._frame_time

    # get frame encoded as JPEG
    def getFrame(self):
        if self._task_id == 0:
            anari_composite_image = io.BytesIO()
            pixels = np.array(self._frame.get('channel.color'))
            img = Image.fromarray(pixels)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('RGB')
            img.save(anari_composite_image, 'JPEG', quality=92)
            return np.frombuffer(anari_composite_image.getbuffer(), dtype=np.uint8)
        else:
            return None

    # set number of samples
    def setNumberOfSamples(self, num):
        self._renderer.setParameter('pixelSamples', anari.INT32, num)
        self._renderer.commitParameters()

    # set render frame size
    def setRenderSize(self, width, height):
        self._framebuffer_size = (width, height)
        self._camera.setParameter('aspect', anari.FLOAT32, self._framebuffer_size[0] / self._framebuffer_size[1])
        self._camera.commitParameters()
        self._frame.setParameter('size', anari.uint2, self._framebuffer_size)
        self._frame.commitParameters()

    # handler for left mouse button -> return whether or not rerender is required
    def onLeftMouseButton(self, mouse_x, mouse_y, pressed):
        if pressed:
            return False
        else:
            return True

    # handler for mouse movement -> return whether or not rerender is required
    def onMouseMove(self, mouse_x, mouse_y):
        return False

    # rotate camera
    def rotateCamera(self, delta_x, delta_y):
        self._cam_theta -= math.radians(delta_x * 0.1)
        self._cam_phi = min(max(self._cam_phi + math.radians(delta_y * 0.1), math.radians(1.0)), math.radians(179.0))
        cam_position = self._calculateCameraPosition()
        self._camera.setParameter('position',anari.FLOAT32_VEC3, cam_position)
        direction = [self._cam_target[0] - cam_position[0],
                     self._cam_target[1] - cam_position[1],
                     self._cam_target[2] - cam_position[2]]
        self._camera.setParameter('direction', anari.float3, direction)
        self._camera.commitParameters()

    # update particle data
    def updateData(self, data):
        surfaces = self._createSurfaces(data)

        self._world.setParameterArray('surface', anari.SURFACE, surfaces)
        self._world.commitParameters()

    # calculate camera position based on spherical coords
    def _calculateCameraPosition(self):
        x = self._cam_radius * math.sin(self._cam_phi) * math.sin(self._cam_theta) + self._cam_target[0]
        y = self._cam_radius * math.cos(self._cam_phi) + self._cam_target[1]
        z = self._cam_radius * math.sin(self._cam_phi) * math.cos(self._cam_theta) + self._cam_target[2]
        return (x, y, z)

    # create ANARI surfaces
    def _createSurfaces(self, pdata):
        radius_array = np.empty(pdata['coordinates'].shape[0], dtype=np.float32)
        radius_array.fill(0.1)
   
        vel_min = 0.0
        vel_max = 12.0 
        velocity_mag = np.linalg.norm(pdata['velocity'], axis=1)

        #colormap = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float32)
        cmap_size = self._colormap.shape[0]
        velocity_norm = np.round((cmap_size - 1) * ((velocity_mag - vel_min) / (vel_max - vel_min))).astype(dtype=np.uint16)
        color_array = self._colormap[velocity_norm]

        center = self._device.newArray(anari.FLOAT32_VEC3, pdata['coordinates'].flatten())
        radius = self._device.newArray(anari.FLOAT32, radius_array)
        color = self._device.newArray(anari.FLOAT32_VEC3, color_array.flatten())

        spheres = self._device.newGeometry('sphere')
        spheres.setParameter('vertex.position', anari.ARRAY, center)
        spheres.setParameter('vertex.radius', anari.ARRAY, radius)
        spheres.setParameter('vertex.color', anari.ARRAY, color)
        spheres.commitParameters()

        material = self._makeMaterial()
        surf = self._device.newSurface()
        surf.setParameter('geometry', anari.GEOMETRY, spheres)
        surf.setParameter('material', anari.MATERIAL, material)
        surf.commitParameters()

        return [surf]

    # create material
    def _makeMaterial(self):
        mat = self._device.newMaterial('matte')
        mat.setParameter('color', anari.STRING, 'color')
        mat.commitParameters()

        return mat

    # create colormap
    def _makeColormap(self, colors, map_size):
        colormap = np.empty((map_size, 3), np.float32)
        npg = map_size // (len(colors) - 1)
        npg_extra = map_size % (len(colors) - 1)
        for i in range(len(colors) - 1):
            c0 = colors[i]
            c1 = colors[i + 1]
            start = i * npg + min(i, npg_extra)
            count = npg if i >= npg_extra else npg + 1
            for n in range(count):
                t = n / count
                colormap[start + n][0] = (1.0 - t) * c0[0] + t * c1[0]
                colormap[start + n][1] = (1.0 - t) * c0[1] + t * c1[1]
                colormap[start + n][2] = (1.0 - t) * c0[2] + t * c1[2]

        return colormap

# run `main()` if primary script
if __name__ == '__main__':
    main()

