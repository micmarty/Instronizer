// Build components
var fileSound;
var wavesurfer = WaveSurfer.create({
    container: "#waveform",
    barWidth: 3,
    height: 300,
    progressColor: '#512da8',
    skipLength: 30,
    fillParent: true
});

// Play uploaded audio
document.getElementById("uploadedFileInput").onchange = function(e) {

    $("#processingProgress").show();

    // Put filename int readonly textfield, when file is chosen
    document.getElementById("uploadFileName").value = this.files[0].name;

    // Build player
    var sound = document.getElementById("waveform");
    fileSound = this.files[0];
    sound.src = URL.createObjectURL(fileSound);

    // Build zoom slider
    var slider = document.querySelector("#zoomSlider");
    slider.oninput = function() {
        var zoomLevel = Number(slider.value);
        wavesurfer.zoom(zoomLevel);
    };

    // Cleanup old regions
    wavesurfer.clearRegions();

    // Load file
    wavesurfer.load(URL.createObjectURL(fileSound));

    // Not really needed in this exact case, but since it is really important in other cases,
    // don't forget to revoke the blobURI when you don't need it
    sound.onend = function(e) {
        URL.revokeObjectURL(this.src);
    };
};
$(function() {
    $("#backward").click(function() {
        wavesurfer.skipBackward();
    });
    $("#togglePlay").click(function() {
        wavesurfer.playPause();
    });
    $("#forward").click(function() {
        wavesurfer.skipForward();
    });
    $("#toggleMute").click(function() {
        wavesurfer.toggleMute();
    });

    wavesurfer.on("ready", function() {
        $("#processingProgress").hide();
        $("#waveformSection").css({ opacity: 0.0, visibility: "visible" }).animate({ opacity: 1.0 }, "slow"); //.fadeToggle(); //css("visibility", "visible");
        // Add timeline
        var timeline = Object.create(WaveSurfer.Timeline);
        timeline.init({
            wavesurfer: wavesurfer,
            container: "#waveform-timeline",
            timeInterval: 5
        });

        // Enable creating regions by dragging
        wavesurfer.addRegion({
            id: "startend",
            start: 0, // time in seconds
            end: 6, // time in seconds
            drag: true,
            resize: false,
            color: "hsla(262, 52%, 47%, 0.48)"
        });
    });
});