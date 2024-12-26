document.getElementById('chatForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = document.getElementById('question').value;
    const chatWindow = document.getElementById('chatWindow');

    const userDiv = document.createElement('div');
    userDiv.classList.add('user-message');
    userDiv.textContent = "You: " + question;
    chatWindow.appendChild(userDiv);

    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });
    const result = await response.json();

    const botDiv = document.createElement('div');
    botDiv.classList.add('bot-message');
    botDiv.textContent = "Bot: " + (result.answer || result.error || "No response from server");

    if (textContent.includes("I'm sorry, the provided context does not contain sufficient information")) {
        botDiv.style.color = "red"; 
    }

    botDiv.textContent = "Bot: " + textContent;
    chatWindow.appendChild(botDiv);

    chatWindow.scrollTop = chatWindow.scrollHeight;
});

document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById("fileInput");
    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append("file", file);
    }

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error('Failed to upload');
        
        const result = await response.json();
        alert(result.message || result.error);
    } catch (error) {
        console.error(error);
        alert("Failed to upload the files.");
    }
});

document.addEventListener('DOMContentLoaded', async () => {
    const chatWindow = document.getElementById('chatWindow');

    try {
        const response = await fetch('/chat_history', { method: 'GET' });
        if (!response.ok) throw new Error('Failed to fetch chat history');
        
        const result = await response.json();
        if (result.chats && Array.isArray(result.chats)) {
            result.chats.forEach(chat => {
                const userDiv = document.createElement('div');
                userDiv.classList.add('user-message');
                userDiv.textContent = "You: " + chat.question;
                chatWindow.appendChild(userDiv);

                const botDiv = document.createElement('div');
                botDiv.classList.add('bot-message');
                botDiv.textContent = "Bot: " + chat.answer;
                chatWindow.appendChild(botDiv);
            });
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
});

