var fileSound;
// Stworzenie instancji
var wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "violet",
    fillParent: true
});
// Play uploaded audio
input.onchange = function(e) {
    var sound = document.getElementById("waveform");
    fileSound = this.files[0];
    sound.src = URL.createObjectURL(fileSound);

    wavesurfer.load(URL.createObjectURL(fileSound));
    // not really needed in this exact case, but since it is really important in other cases,
    // don't forget to revoke the blobURI when you don't need it
    sound.onend = function(e) {
        URL.revokeObjectURL(this.src);
    };
};
wavesurfer.on("ready", function() {
    // Enable creating regions by dragging
    wavesurfer.addRegion({
        id: "startend",
        start: 0, // time in seconds
        end: 6, // time in seconds
        drag: true,
        resize: false,
        color: "hsla(100, 100%, 30%, 0.1)"
    });
});