//* ======================== Slide Control ===================== */
var contents = document.getElementsByClassName("slide-content");

var slideMenu = document.getElementById("slide-menu");
if (slideMenu) {
  slideMenu.addEventListener("click", function(e) {
    const idx = [...this.children]
      .filter(el => el.className.indexOf('dot') > -1)
      .indexOf(e.target);
      
    if (idx >= 0) {
      var prev = document.querySelector(".dot.active");
      if (prev) prev.classList.remove("active");
      e.target.classList.add("active");
      
      for (var i = 0; i < contents.length; i++) {
        if (i == idx) {
          contents[i].style.display = "block";
        } else {
          contents[i].style.display = "none";
        }
      }  
    }
  });
}

//* ======================== Video Control ===================== */
function ToggleVideo(x) {
  var videos = document.getElementsByClassName(x + '-video');
  for (var i = 0; i < videos.length; i++) {
      if (videos[i].paused) {
          videos[i].play();
      } else {
          videos[i].pause();
      }
  }
};


function SlowVideo(x) {
  var videos = document.getElementsByClassName(x + '-video');
  for (var i = 0; i < videos.length; i++) {
    videos[i].playbackRate = videos[i].playbackRate * 0.9;
    videos[i].play();
  }
  
  var msg = document.getElementById(x + '-msg');
  msg.innerHTML = 'Speed: ' + '×' + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = 'none';
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null; };


function FastVideo(x) {
  var videos = document.getElementsByClassName(x + '-video');
  for (var i = 0; i < videos.length; i++) {
    videos[i].playbackRate = videos[i].playbackRate / 0.9;
    videos[i].play();
  }

  var msg = document.getElementById(x + '-msg');
  msg.innerHTML = 'Speed: ' + '×' + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = 'none';
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null; 
};

function RestartVideo(x) {
  var videos = document.getElementsByClassName(x + '-video');
  for (var i = 0; i < videos.length; i++) {
    videos[i].pause();
    videos[i].playbackRate = 1.0;
    videos[i].currentTime = 0;
    videos[i].play();
  }
  
  var msg = document.getElementById(x + '-msg');
  msg.innerHTML = 'Speed: ' + '×' + videos[0].playbackRate.toFixed(2);

  msg.classList.add("fade-in-out");
  msg.style.animation = 'none';
  msg.offsetHeight; /* trigger reflow */
  msg.style.animation = null; 
};

//* ======================== Slide Show Control ===================== */
const slider = document.querySelector('.container .slider');
const [btnLeft, btnRight] = ['prev_btn', 'next_btn'].map(id => document.getElementById(id));
let interval;

// Set positions
const setPositions = () => {
    if (slider) {
        [...slider.children].forEach((item, i) => 
            item.style.left = `${(i-1) * 440}px`);
    }
};

// Initial setup
if (slider) {
    setPositions();
}

// Set transition speed
const setTransitionSpeed = (speed) => {
    if (slider) {
        [...slider.children].forEach(item => 
            item.style.transitionDuration = speed);
    }
};

// Slide functions
const next = (isAuto = false) => {
    if (slider) {
        setTransitionSpeed(isAuto ? '1.5s' : '0.2s');
        slider.appendChild(slider.firstElementChild); 
        setPositions();
    }
};

const prev = () => {
    if (slider) {
        setTransitionSpeed('0.2s');
        slider.prepend(slider.lastElementChild); 
        setPositions();
    }
};

// Auto slide
const startAuto = () => interval = interval || setInterval(() => next(true), 2000);
const stopAuto = () => { clearInterval(interval); interval = null; };

// Event listeners
if (btnRight) btnRight.addEventListener('click', () => next(false));
if (btnLeft) btnLeft.addEventListener('click', prev);

// Mouse hover controls
[slider, btnLeft, btnRight].forEach(el => {
    if (el) {
        el.addEventListener('mouseover', stopAuto);
        el.addEventListener('mouseout', startAuto);
    }
});

// Start auto slide
if (slider) startAuto();

//* ======================== Copy Button in Code ===================== */
// add copy button to code blocks
var codeBlocks = document.querySelectorAll('pre');
codeBlocks.forEach(function(pre) {
  console.log('Processing pre block');
  var button = document.createElement('button');
  button.className = 'code-copy-btn';
  button.innerHTML = '<i class="far fa-copy"></i><span class="copy-text"></span>';
  pre.appendChild(button);
  
  // Add click handler for copy functionality
  button.addEventListener('click', function(e) {
    e.preventDefault();
    
    // Get the code text from the code element
    var code = pre.querySelector('code');
    if (code) {
      var text = code.textContent;
      
      // Copy to clipboard
      navigator.clipboard.writeText(text).then(function() {
          // Add copied class to show text (kept for animation)
          button.classList.add('copied');
          // Change icon to check and show text
          var icon = button.querySelector('i');
          var span = button.querySelector('.copy-text');
          icon.className = 'fa-solid fa-check';
          if (span) span.textContent = 'Copied';

          // Reset icon, text and class after 2 seconds
          setTimeout(function() {
            icon.className = 'far fa-copy';
            if (span) span.textContent = '';
            button.classList.remove('copied');
          }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy:', err);
      });
    }
  });
});

