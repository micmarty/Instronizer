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