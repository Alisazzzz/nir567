/* =========================================
   MODELS.JS
   Chat + Instruct models logic
========================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================
       ELEMENTS
    ===================================== */

    const btnSelectChatModel = document.getElementById("btn-select-chat-model");
    const btnCreateChatModel = document.getElementById("btn-create-chat-model");
    const btnSelectInstructModel = document.getElementById("btn-select-instruct-model");
    const btnCreateInstructModel = document.getElementById("btn-create-instruct-model");

    const modalSelectModel = document.getElementById("modal-select-model");
    const modalCreateModel = document.getElementById("modal-create-model");

    const modelListContainer = document.getElementById("model-list-container");

    const currentChatInfo = document.getElementById("current-chat-model");
    const currentInstructInfo = document.getElementById("current-instruct-model");

    const createModelBtn = document.getElementById("confirm-create-model");

    const connectionTypeSelect = document.getElementById("input-model-option");
    const apiKeyDiv = document.getElementById("api-key-group");
    const apiKeyInput = document.getElementById("input-api-key");

    /* =====================================
       STATE
    ===================================== */

    let currentMode = "chat"; // chat | instruct
    let selectedModelName = null;

    /* =====================================
       MODAL HELPERS
    ===================================== */

    function openModal(modal) {
        modal?.classList.remove("hidden");
    }

    function closeModal(modal) {
        modal?.classList.add("hidden");
    }

    document.querySelectorAll(".modal-close, .modal-cancel").forEach(btn => {
        btn.addEventListener("click", () => {
            btn.closest(".modal")?.classList.add("hidden");
        });
    });

    /* =====================================
       OPEN SELECT MODEL
    ===================================== */

    btnSelectChatModel?.addEventListener("click", () => {
        currentMode = "chat";
        openSelectModelModal();
    });

    btnSelectInstructModel?.addEventListener("click", () => {
        currentMode = "instruct";
        openSelectModelModal();
    });

    function openSelectModelModal() {
        loadModelsFromServer();
        openModal(modalSelectModel);
        document.getElementById("select-model-title").textContent =
            currentMode === "chat" ? "Select Chat Model" : "Select Graph Model";
    }

    /* =====================================
       LOAD MODELS
    ===================================== */

    async function loadModelsFromServer() {
        if (!modelListContainer) return;
        modelListContainer.innerHTML = "";

        try {
            const response = await fetch("/api/models/load-all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_type: currentMode })
            });
            const models = await response.json();

            if (!models.length) {
                modelListContainer.innerHTML = "<p>No models available</p>";
                return;
            }

            models.forEach(model => {
                const div = document.createElement("div");
                div.classList.add("model-item");
                div.dataset.modelName = model.name;

                div.innerHTML = `
                    <div><strong>${model.name}</strong> (${model.option})</div>
                    <div>Model: ${model.model_name}</div>
                    <div>Temp: ${model.temperature}, Max tokens: ${model.max_tokens}</div>
                    <button class="btn btn-small btn-primary select-model-btn">Select</button>
                `;

                div.querySelector(".select-model-btn").addEventListener("click", () => {
                    selectedModelName = model.name;
                    confirmModelSelection();
                });

                modelListContainer.appendChild(div);
            });

        } catch (error) {
            console.error("Error loading models:", error);
        }
    }

    /* =====================================
       CONFIRM MODEL SELECTION
    ===================================== */

    async function confirmModelSelection() {
        if (!selectedModelName) return;

        try {
            const response = await fetch("/api/models/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: selectedModelName,
                    model_type: currentMode
                })
            });
            const result = await response.json();

            if (currentMode === "chat") {
                currentChatInfo.textContent = result.name;
            } else {
                currentInstructInfo.textContent = result.name;
            }

            closeModal(modalSelectModel);

        } catch (error) {
            console.error("Error selecting model:", error);
        }
    }

    /* =====================================
       OPEN CREATE MODEL
    ===================================== */

    btnCreateChatModel?.addEventListener("click", () => {
        currentMode = "chat";
        openModal(modalCreateModel);
    });

    btnCreateInstructModel?.addEventListener("click", () => {
        currentMode = "instruct";
        openModal(modalCreateModel);
    });

    /* =====================================
       SHOW API KEY IF NEEDED
    ===================================== */

    connectionTypeSelect?.addEventListener("change", () => {
        if (connectionTypeSelect.value === "openai" || connectionTypeSelect.value === "hf_api") {
            apiKeyDiv?.classList.remove("hidden");
        } else {
            apiKeyDiv?.classList.add("hidden");
        }
    });

    /* =====================================
       CREATE MODEL
    ===================================== */

    createModelBtn?.addEventListener("click", async () => {
        const payload = {
            name: document.getElementById("input-display-name").value,
            option: connectionTypeSelect.value,
            model_name: document.getElementById("input-model-name").value,
            temperature: parseFloat(document.getElementById("input-temperature").value),
            max_tokens: parseInt(document.getElementById("input-max-tokens").value),
            api: apiKeyInput?.value || "",
            model_type: currentMode
        };

        try {
            const response = await fetch("/api/models/create-and-select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const result = await response.json();

            if (currentMode === "chat") {
                currentChatInfo.textContent = result.name;
            } else {
                currentInstructInfo.textContent = result.name;
            }

            closeModal(modalCreateModel);

        } catch (error) {
            console.error("Error creating model:", error);
        }
    });

});