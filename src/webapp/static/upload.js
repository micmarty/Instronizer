$(function() {
    $("#upload-file-btn").click(function() {
        var form_data = new FormData($("#upload-form")[0]);
        var start = wavesurfer.regions.list["startend"].start;
        var end = wavesurfer.regions.list["startend"].end;
        form_data.append("start", start);
        form_data.append("end", end);
        $.ajax({
            type: "POST",
            url: "/upload",
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            xhr: function() {
                var xhr = new window.XMLHttpRequest();

                // Upload progress
                xhr.upload.addEventListener(
                    "progress",
                    function(evt) {
                        if (evt.lengthComputable) {
                            var percentComplete = evt.loaded / evt.total;
                            percentComplete = parseInt(percentComplete * 100);
                            //Do something with upload progress
                            document
                                .querySelector("#p1")
                                .MaterialProgress.setProgress(percentComplete);
                        } else {
                            document.querySelector("#p1").MaterialProgress.setProgress(10);
                        }
                    },
                    false
                );

                return xhr;
            },
            success: function(response) {
                console.log("success");
                $("#place_for_upload_dialog").html(response);
                componentHandler.upgradeDom();
            }
        });
    });
});