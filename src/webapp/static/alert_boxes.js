var close = document.getElementsByClassName("closebtn");

// For all alert boxes, assign onclick event to close button
for (var i = 0; i < close.length; i++) {
    close[i].onclick = function() {
        // Fade out effect
        var div = this.parentElement;
        div.style.opacity = "0";
        setTimeout(function() { div.style.display = "none"; }, 600);
    }
}
