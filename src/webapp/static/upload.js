var timerInterval;

function startTimer() {
    clearInterval(timerInterval);
    var duration = 60 * 5;
    display = document.querySelector('#time');
    var timer = duration, minutes, seconds;
    timerInterval = setInterval(function () {
        minutes = parseInt(timer / 60, 10)
        seconds = parseInt(timer % 60, 10);

        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        display.textContent = minutes + ":" + seconds;

        if (--timer < 0) {
            //clear file input and file name
            document.getElementById("uploadFileInput").value = "";
            document.getElementById("uploadFileName").value = "";
            // Animate "Waveform section" with FadeOut effect
            var waveform = $("#waveformSection");
            var currentOpacity = waveform.css("opacity");
            if (currentOpacity == 1.0) {
                waveform
                    .css({ opacity: currentOpacity, visibility: "visible" })
                    .animate({ opacity: 0.0 }, "slow");
            }

            // Animate "Result section" with FadeOut effect
            var results = $("#resultsSection");
            currentOpacity = results.css("opacity");
            if (currentOpacity == 1.0) {
                results
                    .css({ opacity: currentOpacity, visibility: "visible" })
                    .animate({ opacity: 0.0 }, "slow");
                    
            }
            clearInterval(timerInterval);
        }
    }, 1000);
}
// Called from waveplot.js - function initGetInstrumentButton
function sendFileToServer(data) {
    $.ajax({
        type: "POST",
        url: "/upload",
        data: data,
        contentType: false,
        cache: false,
        processData: false,
        async: true,
        xhr: function() {
            var xhr = new window.XMLHttpRequest();
            return xhr;
        },
        error: function(xhr, status, error) {
            alert("Unexpected file upload error");
            console.log("File upload error. Please try again later.");
        },
        success: function(response) {
            console.log("File was successfuly uploaded");
            if (response['success']) {
                startTimer()
                window.localStorage.setItem('SavedFilePath', response['path']);
                console.log("File path saved to localStorage");
            } else {
                console.log("Server responed with no success status")
            }
        }
    });
}

function sendRegionsToServer(data) {
    $.ajax({
        type: "POST",
        url: "/classify",
        data: data,
        contentType: false,
        cache: false,
        processData: false,
        async: true,
        xhr: function() {
            var xhr = new window.XMLHttpRequest();
            return xhr;
        },
        error: function(xhr, status, error) {
            alert("Unexpected region processing error. Please try again later.");
            console.log("Regions upload error");
        },
        success: function(response) {
            startTimer()
            console.log("Regions were successfully uploaded");
            $("#waitingForResultsProgress").hide();
            $("#results")
                .prepend(response)
                .slideDown();
            $("#resultsSection")
                .css({ opacity: 0.0, visibility: "visible" })
                .animate({ opacity: 1.0 }, "slow");
            componentHandler.upgradeDom();
        }
    });
}
