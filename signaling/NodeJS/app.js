const WebSocket = require("ws");

const wss = new WebSocket.Server({ port: 8082 }, () => {
  console.log("Signaling server is now listening on port 8082");
});

const servers = []; // Array to store connected servers
let counter = 0;
clientMapper = new Map();
serverToClientsMapper = new Map();

// Function to get values for a key
function getClientsForServer(port) {
  return serverToClientsMapper.get(port) || []; // Return the array or an empty array if not found
}

// Function to add a value to a key
function addClientsToServer(port, value) {
  if (!serverToClientsMapper.has(port)) {
    serverToClientsMapper.set(port, []); // Initialize array if key doesn't exist
  }
  serverToClientsMapper.get(port).push(value); // Add the value
}

function addMapping(ipAddress, value) {
  if (!clientMapper.has(ipAddress)) {
    clientMapper.set(ipAddress, value);
    console.log(`Mapping added: ${ipAddress} => ${value}`);
    return true;
  } else {
    console.log(`Mapping already exists for ${ipAddress}`);
    return false;
  }
}

// Function to remove a mapping
function removeMapping(ipAddress) {
  if (clientMapper.has(ipAddress)) {
    clientMapper.delete(ipAddress);
    console.log(`Mapping for ${ipAddress} removed.`);
  } else {
    console.log(`No mapping found for ${ipAddress}.`);
  }
}

// Broadcast to all.
wss.broadcast = (ipAddress, port, data) => {
  if (ipAddress.startsWith("::ffff:127.0.0.1")) {
    const clientsList = getClientsForServer(port);
    clientsList.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  } else {
    const addressedServer = clientMapper.get(ipAddress);
    wss.clients.forEach((client) => {
      if (client === addressedServer && client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  }
};

wss.on("connection", (ws, req) => {
  console.log(`Client connected. Total connected clients: ${wss.clients.size}`);
  // Retrieve the IP address of the client
  const ipAddress = req.socket.remoteAddress;
  const port = req.socket.remotePort;

  // Check if the IP address starts with "::ffff:127.0.0.1" (IPv6-mapped IPv4 address)
  if (ipAddress && ipAddress.startsWith("::ffff:127.0.0.1")) {
    if (
      !servers.some((server) => {
        server.port === port;
      })
    ) {
      console.log(`Server connected from IP: ${ipAddress}`);

      // Store the server in the array if it matches the condition
      servers.push({
        ws, // WebSocket object
        ipAddress, // IP address
        port,
        connectedAt: Date.now(), // Timestamp for connection
      });

      console.log(`Servers list:`, servers);
    }
  } else {
    if (counter < servers.length) {
      const added = addMapping(ipAddress, servers[counter].ws);
      if (added) {
        addClientsToServer(servers[counter].port, ws);
      }
      counter++;
    } else {
      counter = 0;
      const added = addMapping(ipAddress, servers[counter].ws);
      if (added) {
        addClientsToServer(servers[counter].port, ws);
      }
    }
    console.log(clientMapper);
    console.log(`Client connected from non-local IP: ${ipAddress}`);
  }

  ws.onmessage = (message) => {
    wss.broadcast(ipAddress, port, message.data);
  };

  ws.onclose = () => {
    removeMapping(ipAddress);
    console.log(
      `Client disconnected. Total connected clients: ${wss.clients.size}`
    );
  };
});
