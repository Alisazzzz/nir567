// js-file with all frontend logic about chat



document.addEventListener("DOMContentLoaded", () => {
    //bind some tricky events to buttons and elements
    bindButtons();
});


function bindButtons() {
    //two tabs in chat section
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));            
            this.classList.add('active');
            
            const tabName = this.getAttribute('data-tab');
            const tabContent = document.getElementById(`${tabName}-tab`);
            tabContent.classList.add('active');

            const rightSection = document.getElementById('right-section');
            if (tabName === 'graph') {
                rightSection.style.display = 'flex';
                if (!isGraphLoaded) { loadAndRenderGraph(); }
            } else { 
                rightSection.style.display = 'none'; 
            }
        });
    });
    
    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab && activeTab.getAttribute('data-tab') === 'graph') { loadAndRenderGraph(); }
}

async function sendMessage() {
    const text = document.getElementById("chat-input").value.trim();
    if (!text) return;
    addChatMessage("user", text);
    document.getElementById("chat-input").value = "";
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        addChatMessage("assistant", data.answer);
    } catch (error) {
        console.error("Chat error:", error);
        addChatMessage("assistant", "Error communicating with server");
    }
}

function addChatMessage(role, text) {
    const chatWindow = document.getElementById("chat-window");
    let templateId;
    if (role === "user" || role === "Human") {
        templateId = "human-question-template";
    } else {
        templateId = "chat-answer-template";
    }
    const template = document.getElementById(templateId);
    const clone = template.content.cloneNode(true);
    const paragraphs = clone.querySelectorAll("p");
    
    if (role === "user" || role === "Human") {
        if (paragraphs.length > 0) {
            paragraphs[0].textContent = text;
        }
    } else {
        if (paragraphs.length > 0) {
            paragraphs[0].textContent = `Chat model (${role}):`;
        }
        if (paragraphs.length > 1) {
            paragraphs[1].textContent = text;
        }
    }
    chatWindow.appendChild(clone);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}