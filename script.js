let webcam = document.getElementById('webcam');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let startButton = document.getElementById('startButton');
let stopButton = document.getElementById('stopButton');
let resultsDiv = document.getElementById('results');
let ws;
let streaming = false;


// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        webcam.srcObject = stream;
        webcam.onloadedmetadata = (event) => {
            streaming = true
            console.log('completed get user media');
        };
    })
    .catch(err => {
        console.log("An error occurred: " + err);
    });


startButton.onclick = () => {
    if (streaming) {
        ws = new WebSocket('ws://127.0.0.1:8000/ws');

        ws.onopen = (event) => {
            console.log("WebSocket opened:", event);

            setInterval(() => {
                if(ws.readyState === WebSocket.OPEN){
                    ctx.drawImage(webcam, 0, 0, 640, 480);
                    let frameData = canvas.toDataURL('image/jpeg').split(',')[1];
                    let binaryData = new Uint8Array(atob(frameData).split("").map(char => char.charCodeAt(0)));
                    ws.send(binaryData);
                    console.log("Message sent !");
                }
                
            }, 1000);  // Send frame every 100ms (10fps)
        };

        ws.onmessage = (event) => {
            // Handle the received data (processed frame or prediction)
            // For this example, we'll assume it's a prediction text
            
            resultsDiv.innerText = event.data;
            console.log(event.data)
            console.log('message received');
        };

        ws.onerror = (error) => {
            console.log("WebSocket Error:", error);
        };

        ws.onclose = (event) => {
            if (event.wasClean) {
                console.log(`WebSocket closed cleanly, code=${event.code}, reason=${event.reason}`);
            } else {
                console.log('WebSocket connection died here');
            }
        };
    }
};

stopButton.onclick = () => {
    if (ws) {
        ws.close();
    }
};
