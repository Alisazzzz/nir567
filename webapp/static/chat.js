/* =========================================
   CHAT LOGIC
========================================= */

const chatWindow = document.getElementById("chat-window");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("btn-send-message");

const checkboxAddHistory = document.getElementById("checkbox-add-history");
const checkboxUseLLM = document.getElementById("checkbox-use-llm-filter");

const userTemplate = document.getElementById("template-user-message");
const assistantTemplate = document.getElementById("template-assistant-message");

let isSending = false;


/* =========================================
   AUTO-RESIZE TEXTAREA
========================================= */

function autoResizeTextarea() {
    chatInput.style.height = "auto";
    chatInput.style.height = chatInput.scrollHeight + "px";
}

chatInput.addEventListener("input", autoResizeTextarea);


/* =========================================
   ADD MESSAGE TO CHAT
========================================= */

function appendUserMessage(text) {
    const clone = userTemplate.content.cloneNode(true);
    clone.querySelector(".message-text").textContent = text;
    chatWindow.appendChild(clone);
    scrollToBottom();
}

function appendAssistantMessage(text, modelName) {
    const clone = assistantTemplate.content.cloneNode(true);

    clone.querySelector(".message-text").textContent = text;
    clone.querySelector(".model-name").textContent = modelName;

    const updateBtn = clone.querySelector(".btn-update-graph");

    updateBtn.addEventListener("click", () => {
        updateGraphWithLastAnswer(text);
    });

    chatWindow.appendChild(clone);
    scrollToBottom();
}

function appendSystemMessage(text) {
    const div = document.createElement("div");
    div.className = "chat-message assistant-message";
    div.textContent = text;
    chatWindow.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
}


/* =========================================
   SEND MESSAGE
========================================= */

async function sendMessage() {
    if (isSending) return;

    const text = chatInput.value.trim();
    if (!text) return;

    isSending = true;
    sendButton.disabled = true;

    appendUserMessage(text);

    chatInput.value = "";
    autoResizeTextarea();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                use_timestamps: checkboxUseLLM.checked,
                add_history: checkboxAddHistory.checked
            })
        });

        const data = await response.json();

        appendAssistantMessage(data.answer, data.model);

    } catch (error) {
        appendSystemMessage("Error connecting to server.");
        console.error(error);
    }

    isSending = false;
    sendButton.disabled = false;
}


/* =========================================
   UPDATE GRAPH
========================================= */

async function updateGraphWithLastAnswer(answerText) {
    try {
        await fetch("/api/graph/update", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: answerText
            })
        });

        // если у тебя есть функция перерисовки графа:
        if (typeof reloadGraphVisualization === "function") {
            reloadGraphVisualization();
        }

    } catch (error) {
        console.error("Error updating graph:", error);
    }
}


/* =========================================
   LOAD CHAT HISTORY
   (вызывается извне)
========================================= */

/*
Ожидаемый формат history:

[
  { role: "user", content: "Hello" },
  { role: "assistant", content: "Hi there!" }
]
*/

function loadChatHistory(historyArray) {
    clearChat();

    historyArray.forEach(message => {
        if (message.role === "user") {
            appendUserMessage(message.content);
        } else if (message.role === "assistant") {
            appendAssistantMessage(message.content, "previous model");
        }
    });
}

function clearChat() {
    chatWindow.innerHTML = "";
}


/* =========================================
   KEYBOARD HANDLING
========================================= */

chatInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener("click", sendMessage);


/* =========================================
   EXPORT FUNCTIONS (для других файлов)
========================================= */

window.loadChatHistory = loadChatHistory;
window.clearChat = clearChat;