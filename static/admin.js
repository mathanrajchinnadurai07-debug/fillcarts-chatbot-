// static/admin.js - Admin Panel Knowledge Base Management
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('kb-form');
  const questionInput = document.getElementById('kb-question');
  const answerInput = document.getElementById('kb-answer');
  const intentSelect = document.getElementById('kb-intent');
  const tableBody = document.getElementById('kb-table-body');
  const countSpan = document.getElementById('kb-count');
  const syncBtn = document.getElementById('sync-btn');

  async function loadKnowledge() {
    try {
      const res = await fetch('/knowledge?limit=100', {
        headers: { 'X-API-Key': window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE' }
      });
      const data = await res.json();
      renderTable(data.entries || []);
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="4" style="color:var(--danger-color);">Failed to load entries: ${err.message}</td></tr>`;
    }
  }

  function renderTable(entries) {
    countSpan.textContent = entries.length;
    if (entries.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-secondary);">No custom knowledge entries found. Add one on the left!</td></tr>`;
      return;
    }

    tableBody.innerHTML = entries.map(item => `
      <tr>
        <td><strong>${escapeHtml(item.question)}</strong></td>
        <td>${escapeHtml(item.answer)}</td>
        <td><span class="badge" style="font-size:0.7rem;">${item.intent}</span></td>
        <td>
          <button class="btn-sm btn-danger" onclick="deleteEntry('${item.id}')">Delete</button>
        </td>
      </tr>
    `).join('');
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.deleteEntry = async function(id) {
    if (!confirm('Are you sure you want to delete this knowledge entry?')) return;
    try {
      const res = await fetch(`/knowledge/${id}`, {
        method: 'DELETE',
        headers: { 'X-API-Key': window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE' }
      });
      if (res.ok) {
        loadKnowledge();
      } else {
        alert('Failed to delete entry');
      }
    } catch (err) {
      alert('Error deleting entry: ' + err.message);
    }
  };

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    const answer = answerInput.value.trim();
    const intent = intentSelect.value;

    if (!question || !answer) return;

    try {
      const res = await fetch('/knowledge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE'
        },
        body: JSON.stringify({ question, answer, intent })
      });

      if (res.ok) {
        questionInput.value = '';
        answerInput.value = '';
        loadKnowledge();
        alert('✅ Saved to Chatbot Memory successfully!');
      } else {
        const errData = await res.json();
        alert('Error saving: ' + (errData.detail || 'Unknown error'));
      }
    } catch (err) {
      alert('Network error: ' + err.message);
    }
  });

  syncBtn.addEventListener('click', async () => {
    syncBtn.disabled = true;
    syncBtn.textContent = 'Syncing...';
    try {
      const res = await fetch('/knowledge/sync', {
        method: 'POST',
        headers: { 'X-API-Key': window.CHATBOT_API_KEY || '9C0BB558CB737B91D9A52C5845956D969D8B4B680321C60594CB91A98B5F05FE' }
      });
      const data = await res.json();
      alert(`✅ Re-synced ${data.synced || 0} entries to ChromaDB memory!`);
    } catch (err) {
      alert('Sync failed: ' + err.message);
    } finally {
      syncBtn.disabled = false;
      syncBtn.textContent = '🔄 Re-sync ChromaDB';
    }
  });

  loadKnowledge();
});
