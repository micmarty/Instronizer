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
            console.log(response);

            // TODO display box and put labels into regions

            // $("#place_for_upload_dialog").html(response);
            // componentHandler.upgradeDom();
        }
    });
}