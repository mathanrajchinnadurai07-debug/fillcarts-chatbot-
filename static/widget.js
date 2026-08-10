/**
 * Fillcarts AI Chatbot — Floating Embeddable Widget
 * Copy-paste this script into any website (e.g. www.fillcarts.co.in)
 */
(function () {
  // Config: Detect current script URL or use user override
  const scriptTag = document.currentScript;
  const backendUrl = (window.FILLCARTS_CHATBOT_URL || (scriptTag ? new URL(scriptTag.src).origin : '') || 'http://localhost:8000').replace(/\/$/, '');
  const userId = 'fc_' + Math.random().toString(36).substring(2, 9);

  // Inject Styles
  const style = document.createElement('style');
  style.innerHTML = `
    #fc-chat-widget-button {
      position: fixed;
      bottom: 20px;
      right: 20px;
      width: 60px;
      height: 60px;
      border-radius: 50%;
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      border: none;
      cursor: pointer;
      z-index: 999999;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      transition: transform 0.2s, background 0.2s;
    }
    #fc-chat-widget-button:hover {
      transform: scale(1.05);
      background: #0369a1;
    }
    #fc-chat-widget-box {
      position: fixed;
      bottom: 90px;
      right: 20px;
      width: 360px;
      max-width: calc(100vw - 40px);
      height: 520px;
      max-height: calc(100vh - 120px);
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 16px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
      z-index: 999999;
      display: none;
      flex-direction: column;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #fc-chat-widget-box.open {
      display: flex;
    }
    .fc-widget-header {
      background: #1e293b;
      padding: 14px 16px;
      color: #f8fafc;
      font-weight: 600;
      font-size: 15px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
    }
    .fc-widget-close {
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 20px;
      cursor: pointer;
    }
    .fc-widget-close:hover { color: #fff; }
    .fc-widget-messages {
      flex: 1;
      padding: 14px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .fc-msg {
      max-width: 82%;
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.45;
      word-wrap: break-word;
    }
    .fc-msg.user {
      align-self: flex-end;
      background: #0284c7;
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .fc-msg.bot {
      align-self: flex-start;
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid #334155;
      border-bottom-left-radius: 2px;
    }
    .fc-widget-input-area {
      padding: 10px;
      background: #0f172a;
      border-top: 1px solid #334155;
      display: flex;
      gap: 8px;
    }
    .fc-widget-input {
      flex: 1;
      background: #1e293b;
      border: 1px solid #334155;
      color: #f8fafc;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }
    .fc-widget-input:focus { border-color: #38bdf8; }
    .fc-widget-send {
      background: #0284c7;
      color: white;
      border: none;
      padding: 0 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
  `;
  document.head.appendChild(style);

  // Inject HTML Elements
  const button = document.createElement('button');
  button.id = 'fc-chat-widget-button';
  button.innerHTML = '💬';
  button.title = 'Chat with AI Support';

  const box = document.createElement('div');
  box.id = 'fc-chat-widget-box';
  box.innerHTML = `
    <div class="fc-widget-header">
      <span>💬 Fillcarts Customer Support</span>
      <button class="fc-widget-close" id="fc-widget-close-btn">&times;</button>
    </div>
    <div class="fc-widget-messages" id="fc-widget-messages-list">
      <div class="fc-msg bot">Hello! Welcome to <b>Fillcarts</b>. How can I help you today?</div>
    </div>
    <div class="fc-widget-input-area">
      <input type="text" class="fc-widget-input" id="fc-widget-input-field" placeholder="Ask a question..." />
      <button class="fc-widget-send" id="fc-widget-send-btn">Send</button>
    </div>
  `;

  document.body.appendChild(button);
  document.body.appendChild(box);

  // Logic
  const closeBtn = document.getElementById('fc-widget-close-btn');
  const messagesList = document.getElementById('fc-widget-messages-list');
  const inputField = document.getElementById('fc-widget-input-field');
  const sendBtn = document.getElementById('fc-widget-send-btn');

  button.addEventListener('click', () => box.classList.toggle('open'));
  closeBtn.addEventListener('click', () => box.classList.remove('open'));

  async function sendMsg() {
    const text = inputField.value.trim();
    if (!text) return;

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'fc-msg user';
    userMsg.innerText = text;
    messagesList.appendChild(userMsg);

    inputField.value = '';
    inputField.disabled = true;
    sendBtn.disabled = true;
    messagesList.scrollTop = messagesList.scrollHeight;

    try {
      const res = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': window.FILLCARTS_API_KEY || window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE'
        },
        body: JSON.stringify({ user_id: userId, message: text })
      });
      const data = await res.json();

      const botMsg = document.createElement('div');
      botMsg.className = 'fc-msg bot';
      botMsg.innerText = data.response || 'Sorry, I could not process your request.';
      messagesList.appendChild(botMsg);
    } catch (err) {
      const errorMsg = document.createElement('div');
      errorMsg.className = 'fc-msg bot';
      errorMsg.innerText = 'Unable to connect to support server.';
      messagesList.appendChild(errorMsg);
    } finally {
      inputField.disabled = false;
      sendBtn.disabled = false;
      inputField.focus();
      messagesList.scrollTop = messagesList.scrollHeight;
    }
  }

  sendBtn.addEventListener('click', sendMsg);
  inputField.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMsg(); });
})();
