# WebRTC Client and Signaling Server

A WebRTC client and signaling server implementation for real-time communication with peer-to-peer audio, video, and data transmission. This project demonstrates how to set up a WebRTC client that connects to a signaling server to establish and manage peer connections.

## Overview

The **WebRTC Client and Signaling Server** project includes a WebRTC client application and a signaling server. The client is responsible for creating peer-to-peer connections and transmitting data, while the signaling server handles the exchange of Session Description Protocol (SDP) and ICE candidates to enable connection establishment.

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

To run the client
```bash
   cd webrtc-client-and-signaling-server
   python webrtc_client.py
