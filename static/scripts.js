// ==============================
// scripts.js
// ==============================

async function selectSongAndPlay() {
  try {
    const response = await fetch('/next_song');
    const data = await response.json();
    if(data.song && data.song.audio_url) {
      updatePlayer(data.song, true);
    } else {
      console.error("Failed to get enriched song:", data);
    }
  } catch (error) {
    console.error("Error in selecting song:", error);
  }
}

// ------------------------------
// Check if the browser supports the Web Speech API
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;
const synth = window.speechSynthesis;

// Ensure voices are loaded (voices load asynchronously in some browsers)
window.speechSynthesis.onvoiceschanged = function () {
  console.log("Voices loaded:", synth.getVoices());
};

// ------------------------------
// Global Elements
const audioElement = document.createElement('audio');
const micButton = document.getElementById('mic-button');
const sendButton = document.getElementById('send-button');
const messageInput = document.getElementById('message-input');
const chatMessages = document.getElementById('chat-messages');

// Remove/hide the voice toggle button from UI since we no longer need it.
const voiceToggleBtn = document.getElementById('voice-toggle-btn');
if (voiceToggleBtn) {
  voiceToggleBtn.style.display = 'none';
}

// ------------------------------
// Global State Variables for Playback and TTS
// We remove any dependency on a voice toggle; TTS is enabled ONLY when mic is pressed.
let isSpeechOutputEnabled = false; // Will be true when user uses the Speak (mic) button
let currentSong = null;

// Global variables for sequential song playback from suggestions.
let globalSuggestions = [];      // Array of song objects from backend
let currentSuggestionIndex = 0;  // Index of the current song

// ========================================
// Configure Speech Recognition
// ========================================
if (recognition) {
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    // When speech recognition is used, enable TTS.
    isSpeechOutputEnabled = true;
    sendMessage(transcript);
  };

  recognition.onerror = (event) => {
    appendMessage('Speech recognition error: ' + event.error, 'error');
  };

  recognition.onend = () => {
    appendMessage('Stopped listening.', 'system');
  };

  // When mic (speak) button is clicked, always start recognition and enable TTS.
  micButton.addEventListener('click', () => {
    // Enable speech output so that later, TTS is executed.
    isSpeechOutputEnabled = true;
    recognition.start();
    appendMessage('Listening...', 'system');
    setTimeout(() => {
      recognition.stop();
    }, 5000);
  });
} else {
  micButton.disabled = true;
  micButton.title = 'Speech recognition not supported in this browser.';
}

// ========================================
// Adjust UI Based on User Login Status
// ========================================
fetch('/get_user')
  .then(response => response.json())
  .then(data => {
    const isGuest = !data.username;
    if (isGuest) {
      micButton.style.display = 'none';
      // voiceToggleBtn is already hidden.
    } else {
      micButton.style.display = 'inline-block';
    }
  })
  .catch(error => console.error("Error fetching user:", error));
  function appendMessage(message, sender, isSystem = false) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message ${isSystem ? 'system-message' : ''}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

// ========================================
// updatePlayer function with sequential song handling
// ========================================
function updatePlayer(song, autoPlay = false) {
  console.log("updatePlayer called with song:", song);

  // Get chatMessages element
  const chatMessages = document.getElementById('chat-messages');
  if (!chatMessages) {
    console.error("chat-messages element not found");
    return;
  }

  // Remove any existing player
  const existingPlayer = document.querySelector('.custom-player');
  if (existingPlayer) existingPlayer.remove();

  // Create the player container
  const playerDiv = document.createElement('div');
  playerDiv.className = 'custom-player received-message';

  // Create background, controls, etc.
  const backgroundDiv = document.createElement('div');
  backgroundDiv.className = 'player-background';

  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'player-controls';

  const playPauseBtn = document.createElement('button');
  playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';

  const nextBtn = document.createElement('button');
  nextBtn.innerHTML = '<i class="fas fa-step-forward"></i>';

  const progressSlider = document.createElement('input');
  progressSlider.type = 'range';
  progressSlider.id = 'progress-slider';
  progressSlider.min = 0;
  progressSlider.max = 100;
  progressSlider.value = 0;

  const songInfo = document.createElement('div');
  songInfo.id = 'song-info';

  // Audio element & info setup
  const audio = document.createElement('audio');
  audio.src = `/proxy_audio?url=${encodeURIComponent(song.audio_url)}`;
  songInfo.textContent = `${song.song_name} by ${song.artist_name}`;

  // Set thumbnail with fallback
  const defaultThumbnail = '/static/default_thumbnail.jpg';
  const img = new Image();
  img.onload = () => { backgroundDiv.style.backgroundImage = `url(${img.src})`; };
  img.onerror = () => { backgroundDiv.style.backgroundImage = `url(${defaultThumbnail})`; };
  img.src = song.thumbnail_url || defaultThumbnail;

  // Assemble the player UI
  playerDiv.appendChild(backgroundDiv);
  controlsDiv.appendChild(playPauseBtn);
  controlsDiv.appendChild(progressSlider);
  controlsDiv.appendChild(nextBtn);
  controlsDiv.appendChild(songInfo);
  playerDiv.appendChild(controlsDiv);
  playerDiv.appendChild(audio);

  // Add "Now Playing..." message
  appendMessage(`Now Playing... ${song.song_name}`, 'system', true);

  // Append the constructed player to chatMessages container
  chatMessages.appendChild(playerDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Add event listeners
  playPauseBtn.addEventListener('click', () => {
    if (audio.paused) {
      audio.play().catch(err => console.error('Error playing audio:', err));
      playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    } else {
      audio.pause();
      playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
  });

  nextBtn.addEventListener('click', () => { nextSongSequence(); });

  audio.addEventListener('timeupdate', () => {
    if (isFinite(audio.duration) && audio.duration > 0) {
      progressSlider.value = (audio.currentTime / audio.duration) * 100;
    }
  });

  progressSlider.addEventListener('input', () => {
    if (isFinite(audio.duration)) {
      audio.currentTime = (progressSlider.value / 100) * audio.duration;
    }
  });

  // Inside updatePlayer() after setting up the audio element:
  audio.onended = () => {
    setTimeout(() => { nextSongSequence(); }, 4000);  // 4 second delay after song ends
  };


  if (autoPlay) {
    audio.oncanplay = () => {
      audio.play()
        .then(() => { playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>'; })
        .catch(err => console.error('Error during autoplay:', err));
      audio.oncanplay = null;
    };
  }

  console.log("Player UI created for song:", song.song_name);
}


// ========================================
// Sequential Song Playback Functions
// ========================================
function startSongSequence() {
  if (globalSuggestions.length > 0) {
    const song = globalSuggestions[currentSuggestionIndex];
    // Display a system message with the song name.
    appendMessage(`Now playing: ${song.song_name} by ${song.artist_name}`, 'system', true);
    setTimeout(() => { updatePlayer(song, true); }, 500);
  }
}

async function nextSongSequence() {
  let nextIndex = currentSuggestionIndex + 1;
  if (nextIndex >= globalSuggestions.length) {
    appendMessage("You've reached the end of the song list. Want more recommendations?", "system", true);
    return;
  }
  let nextSong = globalSuggestions[nextIndex];
  
  // Pre-fetch the song if audio_url is missing or invalid
  if (!nextSong.audio_url || nextSong.audio_url === '' || nextSong.audio_url === 'undefined') {
    nextSong = await prefetchSong(nextSong);
    globalSuggestions[nextIndex] = nextSong;
  }
  currentSuggestionIndex = nextIndex;
  updatePlayer(nextSong, true);
}


// ========================================
// Consolidated sendMessage function
// ========================================
async function sendMessage(message) {
  // Remove this line if voice input should trigger TTS:
  // isSpeechOutputEnabled = false; // <-- COMMENT OR REMOVE
  
  appendMessage(message, 'sent');
  
  const typingDiv = document.createElement('div');
  typingDiv.className = 'received-message typing';
  typingDiv.textContent = '....';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  await new Promise(resolve => setTimeout(resolve, 1500));
  chatMessages.removeChild(typingDiv);
  
  try {
    const affirmations = ['yes', 'yeah', 'yep', 'sure', 'okay'];
    const lowerMessage = message.toLowerCase().trim();
    const isAffirmation = affirmations.some(affirm => lowerMessage.includes(affirm));
    const requestBody = { user_input: message };
    if (isAffirmation) requestBody.is_affirmation = true;
    
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });
    const data = await response.json();
    
    if (data.error) {
      appendMessage(data.error, 'error', true);
      if (isSpeechOutputEnabled) speak(data.error);
    } else {
      // Append the chatbot response with animation
      appendMessage(data.response, 'received', true);
      
      // Trigger the TTS output for the chatbot response.
      if (isSpeechOutputEnabled) {
        speak(data.response);
      }
      
      if (data.suggestions && data.suggestions.length > 0) {
        globalSuggestions = data.suggestions;
        currentSuggestionIndex = 0;
        
        const wordCount = data.response.split(" ").length;
        const animationDelay = (wordCount * 200) + 300;
        
        setTimeout(() => {
          prefetchSong(globalSuggestions[0]).then(prefetchedSong => {
            globalSuggestions[0] = prefetchedSong;
            updatePlayer(prefetchedSong, true);
          });
        }, animationDelay);
      }
    }
  } catch (err) {
    appendMessage('Failed to connect to the server', 'error', true);
    console.error("Fetch error:", err);
  }
}


async function prefetchSong(song) {
  try {
    // Send song metadata for enrichment (adjust the endpoint and payload as per your server implementation)
    const response = await fetch('/prefetch_song', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ song })
    });
    const data = await response.json();
    // Expecting data.song to contain audio_url and thumbnail_url at minimum.
    return data.song;
  } catch (err) {
    console.error("Error prefetching song:", err);
    return song; // fallback: return the original song metadata
  }
}

// ========================================
// appendMessage function (with optional animation)
function appendMessage(content, type, animate = false) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `${type}-message`;
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  if (animate) {
    const words = content.split(' ');
    msgDiv.textContent = '';
    let i = 0;
    const interval = setInterval(() => {
      if (i < words.length) {
        msgDiv.textContent += (i === 0 ? '' : ' ') + words[i];
        chatMessages.scrollTop = chatMessages.scrollHeight;
        i++;
      } else {
        clearInterval(interval);
      }
    }, 200);
  } else {
    msgDiv.textContent = content;
  }
}

// ========================================
// speak function for Text-to-Speech (TTS)
function speak(text) {
  if (synth && isSpeechOutputEnabled) {
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();
    const femaleVoiceKeywords = ['female', 'neerja', 'heera', 'samantha', 'tessa', 'zira', 'google us english', 'google uk english female'];
    const isLikelyFemale = (voice) => femaleVoiceKeywords.some(keyword => voice.name.toLowerCase().includes(keyword));
    const enINVoices = voices.filter(voice => voice.lang === 'en-IN');
    if (enINVoices.length > 0) {
      const femaleEnINVoice = enINVoices.find(isLikelyFemale);
      utterance.voice = femaleEnINVoice || enINVoices[0];
      utterance.lang = 'en-IN';
    } else {
      const englishVoices = voices.filter(voice => voice.lang.startsWith('en-'));
      const femaleEnglishVoice = englishVoices.find(isLikelyFemale);
      utterance.voice = femaleEnglishVoice || null;
      utterance.lang = femaleEnglishVoice ? femaleEnglishVoice.lang : 'en-US';
    }
    synth.speak(utterance);
    appendMessage('Speaking...', 'system');
    utterance.onend = () => appendMessage('Speech ended.', 'system');
    utterance.onerror = (event) => appendMessage('Speech synthesis error: ' + event.error, 'error');
    return utterance;
  } else {
    appendMessage('Text-to-speech is not enabled.', 'error');
    return null;
  }
}


// ========================================
// Menu Toggle Functionality
function toggleMenu() {
  const menu = document.getElementById("guest-menu");
  menu.style.display = menu.style.display === "block" ? "none" : "block";
}

window.onclick = function(event) {
  if (!event.target.matches(".dropdown img") && !event.target.matches(".dropdown span")) {
    const dropdowns = document.getElementsByClassName("dropdown-content");
    for (let i = 0; i < dropdowns.length; i++) {
      const openDropdown = dropdowns[i];
      if (openDropdown.style.display === "block") {
        openDropdown.style.display = "none";
      }
    }
  }
};

// ========================================
// DOMContentLoaded: Attach Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  if (!sendButton || !messageInput) {
    console.error("send-button or message-input not found in the DOM!");
    return;
  }
  
  // Attach send button click event.
  sendButton.addEventListener("click", () => {
    const message = messageInput.value.trim();
    if (!message) return;
    messageInput.value = '';
    // When send is used, disable TTS.
    isSpeechOutputEnabled = false;
    sendMessage(message);
  });
  
  // Also, send message on Enter key.
  messageInput.addEventListener("keydown", (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendButton.click();
    }
  });
  
  // Setup user info in UI.
  fetch("/get_user")
    .then(response => response.json())
    .then(data => {
      const usernameDisplay = document.getElementById("username-display");
      const dropdownMenu = document.getElementById("dropdown-menu");
      if (data.username) {
        usernameDisplay.textContent = data.username;
        dropdownMenu.innerHTML = `
          <a href="/">Home</a>
          <a href="#">Upgrade to Premium</a>
          <a href="#">Account Details</a>
          <a href="/logout">Sign Out</a>
        `;
        micButton.style.display = 'inline-block';
      } else {
        usernameDisplay.textContent = "Guest";
        dropdownMenu.innerHTML = `
          <a href="/">Home</a>
          <a href="/login">Sign In</a>
        `;
        micButton.style.display = 'none';
      }
    })
    .catch(error => console.error("Error fetching user:", error));
});
