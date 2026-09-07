const canvas = document.getElementById("digitCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictCanvasBtn = document.getElementById("predictCanvasBtn");

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

const canvasResult = document.getElementById("canvasResult");
const canvasPrediction = document.getElementById("canvasPrediction");
const canvasConfidence = document.getElementById("canvasConfidence");
const canvasConfidenceFill = document.getElementById("canvasConfidenceFill");


ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.lineJoin = "round";


let drawing = false;


function getPosition(event) {

    const rect = canvas.getBoundingClientRect();

    const clientX = event.touches
        ? event.touches[0].clientX
        : event.clientX;

    const clientY = event.touches
        ? event.touches[0].clientY
        : event.clientY;

    return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height)
    };
}


function startDrawing(event) {

    drawing = true;

    const position = getPosition(event);

    ctx.beginPath();

    ctx.moveTo(
        position.x,
        position.y
    );
}


function draw(event) {

    if (!drawing) {
        return;
    }

    event.preventDefault();

    const position = getPosition(event);

    ctx.lineTo(
        position.x,
        position.y
    );

    ctx.stroke();
}


function stopDrawing() {
    drawing = false;
}


canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);


clearBtn.addEventListener("click", function () {

    ctx.fillStyle = "black";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvasResult.classList.add("hidden");
});


imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        previewImage.style.display = "block";
    };

    reader.readAsDataURL(file);
});


predictCanvasBtn.addEventListener("click", async function () {

    const imageData = canvas.toDataURL("image/png");

    const response = await fetch(
        "/predict-canvas",
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                image: imageData
            })
        }
    );

    const data = await response.json();


    if (data.error) {

        alert(data.error);

        return;
    }


    canvasPrediction.textContent = data.prediction;

    canvasConfidence.textContent =
        `Confidence: ${data.confidence}%`;

    canvasConfidenceFill.style.width =
        `${data.confidence}%`;

    canvasResult.classList.remove("hidden");
});
