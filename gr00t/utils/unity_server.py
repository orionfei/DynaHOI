import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
from PIL import Image


class UnityServer:
    def __init__(self, host="0.0.0.0", port=8765, resize_size=(256, 256)):
        self.host = host
        self.port = port
        self.websocket = None
        self._recv_task = None
        self._obs_queue = asyncio.Queue()
        self._metrics_queue = asyncio.Queue()
        self._connection_event = asyncio.Event()
        self.server = None
        self.resize_size = resize_size

        self.task_des_dict = {
            "circular": "Grab the object in the video that is making a circular motion",
            "linear": "Grab the object in the video that is making a straight motion",
            "harmonic": "Grab the object in the video that is doing simple harmonic motion"
        }

        self.debug_idx = 0

    async def start(self):
        """start WebSocket server, wait for Unity client connection"""
        async def handler(websocket):
            print(f"🔌 Unity connected: {websocket.remote_address}")
            self.websocket = websocket
            self._connection_event.set()
            try:
                async for msg in websocket:
                    await self._handle_message(msg)
            except Exception as e:
                print("❌ Unity connection exception:", e)
            finally:
                self.websocket = None
                self._connection_event.clear()
                print("🔌 Unity disconnected")

        self.server = await websockets.serve(handler, self.host, self.port)
        print(f"🚀 Python WebSocket server started: ws://{self.host}:{self.port}")

    async def _handle_message(self, msg):
        try:
            msg = json.loads(msg)
            if msg["type"] == "image_and_state":
                data = json.loads(msg["data"])

                # image
                img_bytes = base64.b64decode(data["image_data"])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, self.resize_size)  # resize to 256x256

                # state
                state = np.array(data["state_data"], dtype=np.float64)  # (18,)

                # add a dimension to the obs for the GR00T interface
                obs = {
                        "video.ego_view": np.expand_dims(image_rgb, axis=0),  # [1,H,W,C]
                        "state.left_hand": state[np.newaxis, ...],  # [1,18]
                        "annotation.human.action.task_description": np.array([data["task_type"]]),
                    }
                self._obs_queue.put_nowait(obs)
                print(f"✅ received obs, put into queue")
            
            elif msg["type"] == "metrics":
                data = json.loads(msg["data"])
                metrics = {
                    "episode_id": data["episode_id"],
                    "repeat": data["repeat"],
                    "success": data["success"],
                    "waitTime": data["waitTime"],
                    "score": data["score"],
                    "min_XZ": data["min_distance_to_target"],
                    "successIndex": data["successIndex"],
                    "minJointToSurfaceDistance": data["minJointToSurfaceDistance"],
                }
                self._metrics_queue.put_nowait(metrics)
                print(f"📉 received metrics, put into queue: episode: {data['episode_id']}, repeat: {data['repeat']} metrics（{data['score']}, {data['success']}）")

            else:
                print("⚠️ received unknown message type:", msg["type"])

        except Exception as e:
            print("❌ parse Unity message failed:", e)

    async def wait_for_connection(self, timeout=30):
        print(f"⏳ waiting for Unity client connection... (timeout: {timeout} seconds)")
        try:
            await asyncio.wait_for(self._connection_event.wait(), timeout=timeout)
            print("✅ Unity client connected")
            return True
        except asyncio.TimeoutError:
            print(f"❌ waiting for Unity client connection timeout ({timeout} seconds)")
            return False

    def wait_for_connection_sync(self, timeout=30):
        """
        synchronous version of wait_for_connection
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.wait_for_connection(timeout))

    def is_connected(self):
        """check if connected"""
        if self.websocket is None:
            return False
        try:
            # in websockets 15.0.1, use state attribute to check connection status
            # OPEN = 1, CLOSING = 2, CLOSED = 3
            if hasattr(self.websocket, 'state'):
                return self.websocket.state == 1
            # try using closed attribute
            elif hasattr(self.websocket, 'closed'):
                return not self.websocket.closed
            # finally try checking close_code
            elif hasattr(self.websocket, 'close_code'):
                return self.websocket.close_code is None
            else:
                # if none of the above, assume connection is normal
                return True
        except Exception as e:
            print(f"⚠️ check connection status failed: {e}")
            return False

    def debug_connection_info(self):
        """debug connection information"""
        if self.websocket is None:
            print("🔍 WebSocket: None")
            return
        
        print(f"🔍 WebSocket object: {type(self.websocket)}")
        for attr in ['state', 'closed', 'close_code', 'open']:
            if hasattr(self.websocket, attr):
                try:
                    value = getattr(self.websocket, attr)
                    print(f"🔍 {attr}: {value}")
                except Exception as e:
                    print(f"🔍 {attr}: Error accessing - {e}")
            else:
                print(f"🔍 {attr}: Not available")

    def get_obs(self, block=True, timeout=None):
        """get one frame obs"""
        loop = asyncio.get_event_loop()
        if block:
            try:
                return loop.run_until_complete(asyncio.wait_for(self._obs_queue.get(), timeout))
            except asyncio.TimeoutError:
                return None
        else:
            if self._obs_queue.empty():
                return None
            return self._obs_queue.get_nowait()
    
    def get_metrics_from_unity(self, block=None, timeout=None):
        """get waitTime and success情况 from Unity side"""
        loop = asyncio.get_event_loop()
        if block:
            try:
                return loop.run_until_complete(asyncio.wait_for(self._metrics_queue.get(), timeout))
            except asyncio.TimeoutError:
                return None
        else:
            if self._metrics_queue.empty():
                return None
            return self._metrics_queue.get_nowait()

    async def send_start_episode(self, episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize):
        """notify Unity to start an episode"""
        if not self.is_connected():
            print("⚠️ no Unity client connected")
            return False
        try:
            msg = {
                "type": "start_episode",
                "data": json.dumps({"episode_id": int(episode_id), "task_type": task_type, "retry_time": int(repeat_num), "total_frame_episode": int(steps), "start_frame_idx": int(start_frame_idx), "windowSize": int(windowSize)})
            }
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ send start_episode failed: {e}")
            self.websocket = None  # reset connection status
            self._connection_event.clear()
            return False

    async def send_action_data(self, actions):
        """send action sequence to Unity"""
        if not self.is_connected():
            print("⚠️ no Unity client connected")
            return False
        try:
            if isinstance(actions, np.ndarray):
                actions = actions.tolist()

            # ✅ key change: wrap each frame in {"values": frame}
            wrapped_actions = [{"values": frame} for frame in actions]

            msg = {
                "type": "action_data",
                "data": json.dumps({"actions": wrapped_actions})
            }
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ send action_data failed: {e}")
            self.websocket = None  # reset connection status
            self._connection_event.clear()
            return False
        
    async def send_inference_complete(self):
        """notify Unity that the inference is complete"""
        if not self.is_connected():
            print("⚠️ no Unity client connected")
            return False
        try:
            msg = {"type": "inference_complete", "data": "{}"}
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ send inference_complete failed: {e}")
            self.websocket = None  # reset connection status
            self._connection_event.clear()
            return False

    async def send_save_results(self):
        """notify Unity to save all results"""
        if not self.is_connected():
            print("⚠️ no Unity client connected")
            return False
        try:
            msg = {"type": "save_results", "data": "{}"}
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ send save_results failed: {e}")
            self.websocket = None  # reset connection status
            self._connection_event.clear()
            return False

    # synchronous version of send methods, for evaluation script
    def send_start_episode_sync(self, episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize):
        """synchronous version of send_start_episode"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_start_episode(episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize))

    def send_action_data_sync(self, actions):
        """synchronous version of send_action_data"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_action_data(actions))

    def send_inference_complete_sync(self):
        """synchronous version of send_inference_complete"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_inference_complete())

    def send_save_results_sync(self):
        """synchronous version of send_save_results"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_save_results())

    async def stop(self):
        """stop the server"""
        self.server.close()
        await self.server.wait_closed()
        print("🛑 Python WebSocket server closed")