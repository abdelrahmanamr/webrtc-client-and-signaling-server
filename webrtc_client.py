import asyncio
import json
import websockets
import cv2
import numpy as np
import torch
import torchvision
import sys
import os
import time
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaBlackhole
from PIL import Image  # To convert NumPy arrays to PIL images

# Add the parent directory of WebRTC_Client_And_Signaling_Server to the sys.path
sys.path.append(os.path.abspath('/home/student'))

from WebRTC_Client_And_Signaling_Server.object_detector import detect_objects
from WebRTC_Client_And_Signaling_Server.dope_model import DopeNetwork

# Define the signaling server URL
SIGNALING_SERVER_URL = "ws://localhost:8082/ws"  # Adjust to your signaling server

# Candidate parsing function
def parse_candidate(sdp):
    parts = sdp.split()
    candidate = {
        "foundation": parts[0].split(':')[1],  # Remove the "candidate:" prefix
        "component": int(parts[1]),
        "protocol": parts[2].lower(),
        "priority": int(parts[3]),
        "ip": parts[4],
        "port": int(parts[5]),
        "type": parts[7]
    }
    return candidate

class SignalingClient:
    def __init__(self, server_url, web_rtc_client):
        self.server_url = server_url
        self.web_rtc_client = web_rtc_client

    async def connect(self):
        async with websockets.connect(self.server_url) as websocket:
            await self.web_rtc_client.initialize(websocket)

class WebRTCClient:
    def __init__(self):
        self.pc = RTCPeerConnection()
        self.websocket = None  # This will be set when initializing the connection
        self.last_timestamp = None  # To track the time of the previous frame
        self.total_bytes = 0        # To accumulate frame size for speed calculation
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
       
        #self.publisher_ = self.create_publisher(String, 'coordinates', 10)
        self.model = DopeNetwork()
        weights_path = os.path.join(".", "mustard_60.pth")
        state_dict = torch.load(weights_path, map_location=device)
        new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()} 
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.coordinates = []

        

    async def initialize(self, websocket):
        # Register signaling event handlers
        self.websocket = websocket
        await self._setup_peer_connection()
        
        # Handle incoming WebSocket messages
        async for message in websocket:
            data = json.loads(message)
            print("data['type']",data["type"])
            
            if data["type"] == "SessionDescription":
                # Check if it's an offer or answer based on additional fields
                sdp_type = data.get("sdp_type", "offer")  # Assuming "sdp_type" indicates the type
                await self._handle_session_description(data['payload']['sdp'], sdp_type)
            elif data["type"] == "IceCandidate":
                print(data)
                await self._handle_ice_candidate(data["payload"])

    async def _setup_peer_connection(self):
        # Handle track events (like video and audio streams)
        @self.pc.on("track")
        async def on_track(track):
            print("Receiving track:", track.kind)
            frame_count = 0
            if track.kind == "video":
                #player = MediaPlayer("/dev/video0")
                while True:
                    frame = await track.recv()
                    frame_size = len(frame.to_ndarray(format="bgr24").tobytes())  
                    self.total_bytes += frame_size
                    
                    # Get the current timestamp
                    current_timestamp = time.time()
                    #if frame_count == 0
                    print("current_timestamp",current_timestamp)

                    if self.last_timestamp is not None:
                        # Calculate time interval
                        interval = current_timestamp - self.last_timestamp
                        if interval > 0:
                            # Calculate transfer speed (bytes per second)
                            speed = frame_size / interval
                            print(f"Frame size: {frame_size} bytes, Transfer speed: {speed:.2f} bytes/sec")

                    # Update timestamp for the next frame
                    self.last_timestamp = current_timestamp
                    img = frame.to_ndarray(format="bgr24")
                    result = detect_objects(self.model, img)
                    if len(result) > 0:
                    	raw_points = result[0]['raw_points']
                    	print("raw_points",raw_points)
                    	# Send the raw points to the WebSocket server
                    	await self.send_raw_points(raw_points)
                    	int_points = []
                    	for point in raw_points:
                    	  if point is not None:
                    	    x, y = point
                    	    if x is not None and y is not None:
                    	      int_points.append([int(x), int(y)])
                    	      
                    	point_color = (0, 255, 0) 
                    	point_radius = 5
                    	point_thickness = -1
                    	image_path = os.path.join(os.getcwd(), "output_640_480_rgb_image_test.png")
                    	pil_image = Image.open(image_path)
                    	opencv_image = np.array(pil_image)
                    	for point in int_points:
                    		cv2.circle(opencv_image, point, point_radius, point_color, point_thickness)
                    		
                    	cv2.imwrite("image_with_points.jpg", opencv_image)
		
			
                    	
        # Ice Candidate Gathering
        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                await self.send_ice_candidate(caFndidate)

    async def _handle_session_description(self, sdp, sdp_type):
        # Set the remote description with the received SDP
        description = RTCSessionDescription(sdp, sdp_type)
        await self.pc.setRemoteDescription(description)
        
        if sdp_type == "offer":
            # Create and send an answer back to the signaling server
            answer = await self.pc.createAnswer()
            print("answer.sdp",answer.sdp)
            await self.pc.setLocalDescription(answer)
            await self.send_sdp(answer.sdp, "answer")

    async def _handle_ice_candidate(self, candidate):
        # Parse the candidate SDP string into components
        candidate_data = parse_candidate(candidate["sdp"])
        
        # Create and add the RTCIceCandidate using parsed data
        ice_candidate = RTCIceCandidate(
            foundation=candidate_data["foundation"],
            component=candidate_data["component"],
            protocol=candidate_data["protocol"],
            priority=candidate_data["priority"],
            ip=candidate_data["ip"],
            port=candidate_data["port"],
            type=candidate_data["type"],
            sdpMid=candidate.get("sdpMid"),           # Add sdpMid if available
            sdpMLineIndex=candidate.get("sdpMLineIndex")  # Add sdpMLineIndex if available
        )
        
        await self.pc.addIceCandidate(ice_candidate)

    async def send_sdp(self, sdp, sdp_type):
        message = json.dumps({
            "type": "SessionDescription",
            "payload": {
            	"type": sdp_type,
            	"sdp": sdp
            }
        })
        # Convert the JSON string to bytes
        message_bytes = message.encode('utf-8')
        await self.websocket.send(message_bytes)

    async def send_ice_candidate(self, candidate):
        message = json.dumps({
            "type": "IceCandidate",
            "candidate": candidate
        })
        message_bytes = message.encode('utf-8')
        await self.websocket.send(message_bytes)
        
    async def send_raw_points(self, raw_points):
        # Filter out None values and format points
        valid_points = []
        for point in raw_points:
          if point is not None:
            x, y = point
            if x is not None and y is not None:
              valid_points.append({"x": x, "y":y})
        	
        message = json.dumps({
            "type": "RawPoints",
            "payload": valid_points
        })
        message_bytes = message.encode('utf-8')
        # Send the message over the WebSocket
        if self.websocket:
            await self.websocket.send(message_bytes)

async def main():
    web_rtc_client = WebRTCClient()
    signaling_client = SignalingClient(SIGNALING_SERVER_URL, web_rtc_client)
    await signaling_client.connect()

if __name__ == "__main__":
    asyncio.run(main())
