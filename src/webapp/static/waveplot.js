// Allow to execute when page is fully loaded
$(function() {
    var wavesurfer = initWavesurfer();
    bindOnUploadChange(wavesurfer);
    initWaveformControls(wavesurfer);
    initZoomSlider(wavesurfer);
    initGetInstrumentButton(wavesurfer);
});


/**
 * Bind action when file input element has changed its value
 */
function bindOnUploadChange(wavesurfer) {
    $('#uploadFileInput').on('change', function() {
        $('#processingProgress').show();

        // Put filename int readonly textfield, when file is chosen
        $('#uploadFileName').val(this.files[0].name);
        fileUrl = URL.createObjectURL(this.files[0]);
        wavesurfer.load(fileUrl);
        wavesurfer.clearRegions();
    });
}

/**
 * Assign actions to player buttons
 */
function initWaveformControls(player) {
    $('#backward').click(function() {
        player.skipBackward();
    });
    $('#togglePlay').click(function() {
        player.playPause();
    });
    $('#forward').click(function() {
        player.skipForward();
    });
    $('#toggleMute').click(function() {
        player.toggleMute();
    });
}

/**
 * Build wavesurfer object with all necessary options
 * Add timeline and regions plugins
 */
function initWavesurfer() {
    var wavesurfer = WaveSurfer.create({
        container: '#waveform',
        barWidth: 3,
        height: 300,
        progressColor: '#512da8',
        skipLength: 30,
        fillParent: true
    });

    wavesurfer.on('ready', function() {
        // Hide Progress bar
        $('#processingProgress').hide();

        // Animate with FadeIn effect
        $('#waveformSection')
            .css({ opacity: 0.0, visibility: 'visible' })
            .animate({ opacity: 1.0 }, 'slow');

        // Add timeline
        var timeline = Object.create(WaveSurfer.Timeline);
        timeline.init({
            wavesurfer: wavesurfer,
            container: '#waveform-timeline',
            timeInterval: 5
        });

        // Enable creating regions by dragging
        wavesurfer.addRegion({
            id: 'startend',
            start: 0, // in seconds
            end: 6, // in seconds
            drag: true,
            resize: false,
            color: 'hsla(262, 52%, 47%, 0.48)'
        });
    });
    return wavesurfer;
}

function initZoomSlider(player) {
    var slider = document.querySelector("#zoomSlider");

    slider.oninput = function() {
        var zoomLevel = Number(slider.value);
        player.zoom(zoomLevel);
    };
}

function initGetInstrumentButton(wavesurfer) {
    $("#getInstrumentNameButton").click(function() {
        var form_data = new FormData();
        file = $("#uploadFileInput")[0].files[0];
        var start = wavesurfer.regions.list["startend"].start;
        var end = wavesurfer.regions.list["startend"].end;

        form_data.append("file", file);
        form_data.append("start", start);
        form_data.append("end", end);

        // Defined in upload.js
        sendFileToServer(form_data);
    });
}