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
            // TODO progress bar
            return xhr;
        },
        error: function(xhr, status, error) {
            // TODO when file is too big, or in wrong format display getmdl popup
            console.log("File upload error");
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
            // TODO progress bar
            return xhr;
        },
        error: function(xhr, status, error) {
            // TODO when file is too big, or in wrong format display getmdl popup
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