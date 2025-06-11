// ==============================
// scripts.js
// ==============================

let currentUserId = null;
if (typeof chatHistory === 'undefined' || !Array.isArray(chatHistory)) {
  var chatHistory = [];
}

function startNewChat() {
  chatId = null;
  chatHistory = [];
  localStorage.removeItem("chatId");
  window.location.reload();  // Or re-render the chat UI appropriately
}

fetch('/get_user')
  .then(response => response.json())
  .then(data => {
    if (data.user_id) {
      currentUserId = data.user_id;
    } else if (data.username) {
      currentUserId = data.username;
    } else {
      currentUserId = "guest";
    }
    console.log("Current user id:", currentUserId);
  })
  .catch(err => {
    console.error("Error fetching user details:", err);
    currentUserId = "guest";
  });

// --- Modified saveChatHistory() ---
function saveChatHistory() {
  if (!chatHistory || chatHistory.length === 0) {
    console.log("Chat history is empty. Skipping save.");
    return;
  }
  let effectiveChatId = chatId;
  if (typeof effectiveChatId !== 'string' || effectiveChatId.trim() === "" || effectiveChatId === "None") {
    effectiveChatId = null;
  }
  const payload = {
    user_id: currentUserId || "guest",
    session_name: "Default Chat",
    messages: chatHistory,
    chat_id: effectiveChatId
  };
  fetch('/save_chat', {
    method: 'POST',
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    credentials: 'include'
  })
  .then(response => response.json())
  .then(result => {
    console.log("Chat history saved:", result);
    if (result.id && !effectiveChatId) {
      chatId = result.id;
      localStorage.setItem("chatId", result.id);
    }
  })
  .catch(err => {
    console.error("Error saving chat history:", err);
  });
}

function saveChatHistoryBeacon() {
  if (typeof chatHistory === 'undefined' || !Array.isArray(chatHistory)) return;
  if (!chatHistory || chatHistory.length === 0) return;
  let storedChatId = chatId || localStorage.getItem("chatId") || null;
  const payload = JSON.stringify({
    user_id: currentUserId || "guest",
    session_name: "Default Chat",
    messages: chatHistory,
    chat_id: storedChatId
  });
  navigator.sendBeacon('/save_chat', payload);
}

window.addEventListener('beforeunload', () => {
  saveChatHistoryBeacon();
});

// Web Speech API Setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;
const synth = window.speechSynthesis;

window.speechSynthesis.onvoiceschanged = function () {
  console.log("Voices loaded:", synth.getVoices());
};

// Global Elements
const audioElement = document.createElement('audio');
const micButton = document.getElementById('mic-button');
const sendButton = document.getElementById('send-button');
const messageInput = document.getElementById('message-input');
const chatMessages = document.getElementById('chat-messages');

const voiceToggleBtn = document.getElementById('voice-toggle-btn');
if (voiceToggleBtn) {
  voiceToggleBtn.style.display = 'none';
}

// Global State Variables
let isSpeechOutputEnabled = false;
let currentSong = null;
let globalSuggestions = [];
let currentSuggestionIndex = 0;

if (recognition) {
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    isSpeechOutputEnabled = true;
    sendMessage(transcript);
  };

  recognition.onerror = (event) => {
    appendMessage('Speech recognition error: ' + event.error, 'error');
  };

  recognition.onend = () => {
    appendMessage('Stopped listening.', 'system');
  };

  if (micButton) {
    micButton.addEventListener('click', () => {
      isSpeechOutputEnabled = true;
      recognition.start();
      appendMessage('Listening...', 'system');
      setTimeout(() => recognition.stop(), 5000);
    });
  } else {
    console.log("micButton element not found in the DOM.");
  }
} else {
  if (micButton) {
    micButton.disabled = true;
    micButton.title = 'Speech recognition not supported in this browser.';
  }
}

function debounce(fn, delay) {
  let timerId;
  return function(...args) {
    if (timerId) clearTimeout(timerId);
    timerId = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  }
}
const debouncedSaveChatHistory = debounce(saveChatHistory, 2000);

fetch('/get_user')
  .then(response => response.json())
  .then(data => {
    const isGuest = !data.username;
    if (micButton) {
      micButton.style.display = isGuest ? 'none' : 'inline-block';
    }
  })
  .catch(error => console.error("Error fetching user:", error));

function appendMessage(content, sender, animate = false, includeFeedback = false) {
  if (typeof content !== 'string') {
    content = String(content);
  }

  const msgDiv = document.createElement('div');
  msgDiv.className = `${sender}-message`;
  msgDiv.textContent = content;

  if (includeFeedback && sender === 'received') {
    const feedbackContainer = document.createElement('div');
    feedbackContainer.className = 'feedback-container';

    const likeBtn = document.createElement('button');
    likeBtn.className = 'feedback-btn like';
    likeBtn.setAttribute('data-feedback', 'like');
    likeBtn.innerHTML = '👍';

    const dislikeBtn = document.createElement('button');
    dislikeBtn.className = 'feedback-btn dislike';
    dislikeBtn.setAttribute('data-feedback', 'dislike');
    dislikeBtn.innerHTML = '👎';

    likeBtn.addEventListener('click', () => sendFeedback('like'));
    dislikeBtn.addEventListener('click', () => sendFeedback('dislike'));

    feedbackContainer.appendChild(likeBtn);
    feedbackContainer.appendChild(dislikeBtn);
    msgDiv.appendChild(feedbackContainer);
  }

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
    }, 150);
  }

  chatHistory.push({
    sender: sender,
    text: content,
    timestamp: new Date().toISOString()
  });
  debouncedSaveChatHistory();
}

function updatePlayer(song, autoPlay = false) {
  console.log("updatePlayer called with song:", song);
  const existingPlayer = document.querySelector('.custom-player');
  if (existingPlayer) existingPlayer.remove();

  const playerDiv = document.createElement('div');
  playerDiv.className = 'custom-player received-message';

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

  const feedbackContainer = document.createElement('div');
  feedbackContainer.className = 'feedback-container';

  const likeBtn = document.createElement('button');
  likeBtn.className = 'feedback-btn like';
  likeBtn.setAttribute('data-feedback', 'like');
  likeBtn.innerHTML = '👍';

  const dislikeBtn = document.createElement('button');
  dislikeBtn.className = 'feedback-btn dislike';
  dislikeBtn.setAttribute('data-feedback', 'dislike');
  dislikeBtn.innerHTML = '👎';

  likeBtn.addEventListener('click', () => sendFeedback('like', song.track_id));
  dislikeBtn.addEventListener('click', () => sendFeedback('dislike', song.track_id));

  feedbackContainer.appendChild(likeBtn);
  feedbackContainer.appendChild(dislikeBtn);

  const songInfo = document.createElement('div');
  songInfo.id = 'song-info';

  const audio = document.createElement('audio');
  audio.src = `/proxy_audio?url=${encodeURIComponent(song.audio_url)}`;
  songInfo.textContent = `${song.song_name} by ${song.artist_name}`;

  const defaultThumbnail = '/static/default_thumbnail.jpg';
  const img = new Image();
  img.onload = () => { backgroundDiv.style.backgroundImage = `url(${img.src})`; };
  img.onerror = () => { backgroundDiv.style.backgroundImage = `url(${defaultThumbnail})`; };
  img.src = song.thumbnail_url || defaultThumbnail;

  playerDiv.appendChild(backgroundDiv);
  controlsDiv.appendChild(playPauseBtn);
  controlsDiv.appendChild(progressSlider);
  controlsDiv.appendChild(nextBtn);
  controlsDiv.appendChild(songInfo);
  controlsDiv.appendChild(feedbackContainer);
  playerDiv.appendChild(controlsDiv);
  playerDiv.appendChild(audio);

  appendMessage(`Now Playing... ${song.song_name}`, 'system', true);
  chatMessages.appendChild(playerDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Retry function for playing audio
  async function playAudioWithRetry() {
    const maxRetries = 1; // Total of 2 attempts (initial + 1 retry)
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        await audio.play();
        return true; // Success
      } catch (err) {
        if (err.name === 'NotSupportedError' && attempt < maxRetries) {
          console.log(`Attempt ${attempt + 1} failed. Retrying in 1 second...`);
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
        } else {
          console.error('Error playing audio after retries:', err);
          return false; // Failure
        }
      }
    }
    return false; // Shouldn't reach here, but added for completeness
  }

  // Play/Pause button with retry
  playPauseBtn.addEventListener('click', async () => {
    if (audio.paused) {
      const success = await playAudioWithRetry();
      if (success) {
        playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
      } else {
        appendMessage("Failed to play the song after retries.", 'error');
        nextSongSequence(); // Move to next song
      }
    } else {
      audio.pause();
      playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
  });

  // Next button event listener
  nextBtn.addEventListener('click', () => {
    if (globalPrefetchedSong && globalPrefetchedSong.audio_url) {
      currentSuggestionIndex++;
      updatePlayer(globalPrefetchedSong, true);
      prefetchNextSong().then(prefetchedSong => {
        globalPrefetchedSong = prefetchedSong;
        console.log("New next song prefetched:", globalPrefetchedSong);
      });
    } else {
      nextSongSequence();
    }
  });

  // Progress slider update
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

  // On audio end, move to next song
  audio.onended = () => {
    setTimeout(() => {
      if (globalPrefetchedSong) {
        currentSuggestionIndex++;
        updatePlayer(globalPrefetchedSong, true);
        prefetchNextSong().then(prefetchedSong => {
          globalPrefetchedSong = prefetchedSong;
        });
      } else {
        nextSongSequence();
      }
    }, 4000);
  };

  // Autoplay with retry
  if (autoPlay) {
    audio.oncanplay = async () => {
      const success = await playAudioWithRetry();
      if (success) {
        playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
      } else {
        appendMessage("Failed to play the song after retries.", 'error');
        nextSongSequence(); // Move to next song
      }
      audio.oncanplay = null;
    };
  }

  // Prefetch next song
  prefetchNextSong().then(prefetchedSong => {
    globalPrefetchedSong = prefetchedSong;
    console.log("Next song prefetched:", globalPrefetchedSong);
  });
}

function startSongSequence() {
  if (globalSuggestions.length > 0) {
    const song = globalSuggestions[currentSuggestionIndex];
    appendMessage(`Now playing: ${song.song_name} by ${song.artist_name}`, 'system', true);
    setTimeout(() => { updatePlayer(song, true); }, 500);
  }
}

async function nextSongSequence() {
  let nextIndex = currentSuggestionIndex + 1;

  if (nextIndex >= globalSuggestions.length) {
    const endMessage = "You've reached the end of the song list. Want more recommendations?";
    appendMessage(endMessage, 'system', true);
    return;
  }

  const nextSong = globalSuggestions[nextIndex];

  if (!nextSong.audio_url || nextSong.audio_url === '' || nextSong.audio_url === 'undefined') {
    let prefetched = await prefetchSong(nextSong);
    globalSuggestions[nextIndex] = prefetched;
  }
  
  currentSuggestionIndex = nextIndex;
  updatePlayer(globalSuggestions[nextIndex], true);
}

function handleAffirmation() {
  nextSongSequence();
}

function isNegative(text) {
  const negativeWords = ["no", "nope", "nah", "not really", "negative"];
  const lowerText = text.toLowerCase();
  return negativeWords.some(word => lowerText === word || lowerText.includes(word));
}


async function sendMessage(message) {
  // Append the user's message to the chat UI
  appendMessage(message, 'sent');

  // Show temporary typing indicator
  const typingDiv = document.createElement('div');
  typingDiv.className = 'received-message typing';
  typingDiv.textContent = '....';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  await new Promise(resolve => setTimeout(resolve, 1500));
  chatMessages.removeChild(typingDiv);

  try {
    const affirmativeWords = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'of course'];
    const lowerMessage = message.toLowerCase().trim();
    const isAffirmative = affirmativeWords.some(word => lowerMessage.includes(word));
    const isNegativeResponse = isNegative(lowerMessage); // our new negative helper
    
    // Build the request payload. Notice we pass the chat_id from a global variable if available.
    const requestBody = {
      user_input: message,
      // Pass the current chatId (might be null or an empty string)
      chat_id: (typeof chatId !== 'undefined' && chatId) ? chatId : ""
    };
    if (isAffirmative) {
      requestBody.is_affirmation = true;
    } else if (isNegativeResponse) {
      requestBody.is_affirmation = false;
    }
    
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      credentials: 'include'
    });
    
    const data = await response.json();
    // If data.response isn't a string, convert it to one.
    if (typeof data.response !== 'string') {
      data.response = JSON.stringify(data.response);
    }

    // If an error occurred, show it
    if (data.error) {
      appendMessage(data.error, 'error', true);
      if (isSpeechOutputEnabled) speak(data.error);
    } else {
      // Append the response from the server
      appendMessage(data.response, 'received', true);
      if (isSpeechOutputEnabled) speak(data.response);

      // If the server created a new chat, update your global chatId.
      if (data.chat_id) {
        chatId = data.chat_id;
        localStorage.setItem("chatId", data.chat_id);
      }

      // Process any suggestions
      if (data.suggestions && data.suggestions.length > 0) {
        globalSuggestions = data.suggestions;
        currentSuggestionIndex = 0;

        const wordCount = data.response.split(" ").length;
        const animationDelay = (wordCount * 180) + 300;

        setTimeout(() => {
          prefetchSong(globalSuggestions[0]).then(prefetchedSong => {
            globalSuggestions[0] = prefetchedSong;
            updatePlayer(prefetchedSong, true);
          });
        }, animationDelay);
      }
    }
  } catch (err) {
    appendMessage(`Error: ${err.message}`, 'error', true);
    console.error("Error in sendMessage:", err);
  }
  
  // After processing the message, save the chat history
  await saveChatHistory();
}

async function initChat() {
  // Check if a chatId is already stored (from localStorage, for example)
  if (!chatId) {
    // Fire a POST request with the auto_greeting flag
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_greeting: true }),  // minimal payload
        credentials: 'include'
      });
      const data = await response.json();
      console.log("Auto greeting response:", data);
      // Update the UI with the greeting message
      if (data.response) {
        appendMessage(data.response, 'received', true);
      }
      // Save the new chat id globally & to localStorage for later use
      if (data.chat_id) {
        chatId = data.chat_id;
        localStorage.setItem("chatId", data.chat_id);
      }
    } catch (err) {
      console.error("Error initializing chat (auto greeting):", err);
    }
  }
}



async function prefetchSong(song) {
  try {
    const response = await fetch('/prefetch_song', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ song })
    });
    const data = await response.json();
    if (data.song && data.song.audio_url) {
      return data.song;
    } else {
      console.warn("Prefetched song missing audio_url.", data);
      return song;
    }
  } catch (err) {
    console.error("Error prefetching song:", err);
    return song;
  }
}

let globalPrefetchedSong = null;

async function prefetchNextSong() {
  if (globalSuggestions && globalSuggestions[currentSuggestionIndex + 1]) {
    const nextSong = globalSuggestions[currentSuggestionIndex + 1];
    try {
      const prefetchedSong = await prefetchSong(nextSong);
      return prefetchedSong;
    } catch (error) {
      console.error("Error prefetching next song:", error);
      return nextSong;
    }
  }
  return null;
}

async function sendFeedback(feedbackType, songId) {
  const userId = currentUserId || "guest";
  const payload = {
    user_id: userId,
    song_id: songId,
    feedback: feedbackType
  };

  try {
    const response = await fetch('/submit_feedback', {
      method: 'POST',
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    console.log('Feedback submitted:', result);
  } catch (err) {
    console.error("Error submitting feedback:", err);
  }
}

function speak(text) {
  if (synth && isSpeechOutputEnabled) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.8;

    const voices = synth.getVoices();
    const femaleVoiceKeywords = ['female', 'neerja', 'heera', 'samantha', 'tessa', 'zira', 'google us english'];
    const isLikelyFemale = (voice) => femaleVoiceKeywords.some(keyword => voice.name.toLowerCase().includes(keyword));
    const enINVoices = voices.filter(voice => voice.lang === 'en-IN');
    if (enINVoices.length > 0) {
      const femaleEnINVoice = enINVoices.find(isLikelyFemale);
      utterance.voice = femaleEnINVoice || enINVoices[0];
      utterance.lang = 'hi-IN';
    } else {
      const englishVoices = voices.filter(voice => voice.lang.startsWith('en-'));
      const femaleEnglishVoice = englishVoices.find(isLikelyFemale);
      utterance.voice = femaleEnglishVoice || null;
      utterance.lang = femaleEnglishVoice ? femaleEnglishVoice.lang : 'en-IN';
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

function toggleMenu() {
  const dropdown = document.getElementById('dropdown-menu');
  if (dropdown.style.display === 'block') {
    dropdown.style.display = 'none';
  } else {
    dropdown.style.display = 'block';
  }
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

document.addEventListener("DOMContentLoaded", () => {
  initChat();
  if (window.location.pathname === "/") {
    const showChatHistoryBtn = document.getElementById("show-chat-history");
    const chatHistoryDiv = document.getElementById("chat-history");
    if (showChatHistoryBtn && chatHistoryDiv) {
      showChatHistoryBtn.addEventListener("click", () => {
        const computedStyle = window.getComputedStyle(chatHistoryDiv).display;
        console.log("Chat history button clicked. Current display is:", computedStyle);
        chatHistoryDiv.style.display = computedStyle === "none" ? "block" : "none";
      });
    }
  }
 
  const sendButton = document.getElementById("send-button");
  const messageInput = document.getElementById("message-input");
  if (sendButton && messageInput) {
    sendButton.addEventListener("click", () => {
      const message = messageInput.value.trim();
      if (!message) return;
      messageInput.value = '';
      isSpeechOutputEnabled = false;
      sendMessage(message);
    });

    messageInput.addEventListener("keydown", (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        sendButton.click();
      }
    });
  } else {
    console.log("Send button or message input not found; likely on index page.");
  }
});

function showToast(message, type) {
  const colors = {
    success: "#28a745",
    error: "#dc3545",
    warning: "#ffc107",
    message: "#17a2b8"
  };

  Swal.fire({
    toast: true,
    position: "top",
    icon: type,
    title: message,
    showConfirmButton: false,
    timer: 1500,
    background: colors[type] || "#333",
    color: "#fff",
    customClass: {
      popup: "colored-toast"
    },
    didOpen: () => {
      document.body.style.overflow = "hidden";
    },
    didClose: () => {
      document.body.style.overflow = "";
    }
  });
}

function upgradeToPremium() {
  fetch("/upgrade")
    .then((res) => {
      if (res.ok) {
        showToast("Congrats! You've been upgraded successfully", "success");
        setTimeout(() => {
          window.location.href = "/";
        }, 1100);
      } else {
        showToast("Failed to send upgrade email.", "error");
      }
    })
    .catch(() => {
      showToast("An unexpected error occurred.", "error");
    });
}