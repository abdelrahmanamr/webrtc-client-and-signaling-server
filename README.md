# WebRTC Client and Signaling Server

A WebRTC client and signaling server implementation for real-time communication with peer-to-peer audio, video, and data transmission. This project demonstrates how to set up a WebRTC client that connects to a signaling server to establish and manage peer connections.

## Overview

The **WebRTC Client and Signaling Server** project includes a WebRTC client application and a signaling server. The client is responsible for creating peer-to-peer connections and transmitting data, while the signaling server handles the exchange of Session Description Protocol (SDP) and ICE candidates to enable connection establishment.

For a **demo video** showcasing the functionality of this project, please refer to this [link](https://drive.google.com/drive/folders/1SYTTRWaFQHZBB-C2ngWCZ6FboDucJ3bx?usp=sharing).

### Components
- **WebRTC Client**: Initiates and manages WebRTC peer connections, handles tracks (audio, video, or data), and processes SDP and ICE candidates.
- **Signaling Server**: Acts as a bridge to exchange connection details (SDP, ICE candidates) between peers, allowing them to establish a direct connection.

## Installation
To run the server
```bash
   git clone https://github.com/abdelrahmanamr/webrtc-client-and-signaling-server.git
   cd webrtc-client-and-signaling-server
   cd signaling/NodeJS
   npm install
   node app.js
```

To run the client
```bash
   cd webrtc-client-and-signaling-server
   pip3 install asyncio json5 websockets opencv-python numpy torch torchvision aiortc scipy pyrr tensorflow
   python webrtc_client.py
```
## Acknowledgments
The **signaling server** in this repository is based on the original work from [WebRTC-iOS](https://github.com/stasel/WebRTC-iOS). However, significant modifications have been made to the signaling server to enhance its functionality. Specifically:
- The signaling server has been updated to support communication not only with mobile phones but also with **computers**.
- Additional improvements have been made to handle multi-server multi-client feature.

These changes make the project more versatile and suitable for a wider range of use cases.
