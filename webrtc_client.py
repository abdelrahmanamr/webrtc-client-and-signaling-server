import asyncio
import json
import websockets
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaBlackhole

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
                    img = frame.to_ndarray(format="bgr24")

                    # Construct filename for each frame
                    filename = f"frame_{frame_count}.jpg"

                    # Save the image using OpenCV
                    cv2.imwrite(filename, img)
                    print(f"Saved frame {frame_count} as {filename}")

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

async def main():
    web_rtc_client = WebRTCClient()
    signaling_client = SignalingClient(SIGNALING_SERVER_URL, web_rtc_client)
    await signaling_client.connect()

if __name__ == "__main__":
    asyncio.run(main())
