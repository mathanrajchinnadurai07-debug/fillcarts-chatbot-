// static/chat.js - Customer Chat Client Logic
document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const modeBadge = document.getElementById('mode-badge');

  // Random user ID for session
  const userId = 'web_user_' + Math.random().toString(36).substring(2, 9);

  // Check system mode (offline vs gemini)
  fetch('/mode')
    .then(res => res.json())
    .then(data => {
      if (modeBadge) {
        if (data.mode === 'offline') {
          modeBadge.textContent = '100% Offline Mode';
          modeBadge.className = 'badge badge-offline';
        } else {
          modeBadge.textContent = 'Gemini 2.5 Flash Connected';
          modeBadge.className = 'badge';
        }
      }
    })
    .catch(() => {});

  function appendMessage(text, sender, meta = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerText = text;

    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.innerHTML = meta;

    msgDiv.appendChild(bubble);
    msgDiv.appendChild(metaDiv);
    chatMessages.appendChild(msgDiv);

    // Auto scroll
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function sendMessage() {
    const messageText = userInput.value.trim();
    if (!messageText) return;

    // Render user message
    appendMessage(messageText, 'user', 'You');
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE'
        },
        body: JSON.stringify({
          user_id: userId,
          message: messageText
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      let metaInfo = `Intent: <span class="source-tag">${data.intent}</span> (${Math.round(data.confidence * 100)}% conf)`;
      if (data.sources && data.sources.length > 0) {
        metaInfo += ` • Source: ${data.sources[0].intent || 'Knowledge Base'}`;
      }
      metaInfo += ` • ${data.latency_ms}ms`;

      appendMessage(data.response, 'bot', metaInfo);

    } catch (err) {
      appendMessage(`Sorry, something went wrong: ${err.message}`, 'bot', 'Error');
    } finally {
      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  sendBtn.addEventListener('click', sendMessage);

  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });
});
