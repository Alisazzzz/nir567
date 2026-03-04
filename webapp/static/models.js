/* =========================================
   MODELS.JS (rewritten)
   Template-based selection
========================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       ELEMENTS
    ===================================================== */

    const btnSelectChat = document.getElementById("btn-select-chat-model");
    const btnSelectInstruct = document.getElementById("btn-select-instruct-model");

    const btnCreateChat = document.getElementById("btn-create-chat-model");
    const btnCreateInstruct = document.getElementById("btn-create-instruct-model");

    const modalSelect = document.getElementById("modal-select-model");
    const modalCreate = document.getElementById("modal-create-model");

    const modalOverlay = document.getElementById("modal-overlay");

    const modelListContainer = document.getElementById("model-list-container");
    const selectTemplate = modalSelect.querySelector("template");

    const confirmSelectBtn = document.getElementById("confirm-select-model");
    const confirmCreateBtn = document.getElementById("confirm-create-model");

    const selectTitle = document.getElementById("select-model-title");
    const createTitle = document.getElementById("create-model-title");

    const currentChatLabel = document.getElementById("current-chat-model");
    const currentInstructLabel = document.getElementById("current-instruct-model");

    const inputOption = document.getElementById("input-model-option");
    const inputModelName = document.getElementById("input-model-name");
    const inputTemp = document.getElementById("input-temperature");
    const inputMaxTokens = document.getElementById("input-max-tokens");
    const inputDisplayName = document.getElementById("input-display-name");
    const inputApiKey = document.getElementById("input-api-key");
    const apiKeyGroup = document.getElementById("api-key-group");

    /* =====================================================
       STATE
    ===================================================== */

    let currentMode = "chat";            // chat | instruct
    let backendCurrentModel = null;      // то, что реально выбрано
    let tempSelectedModel = null;        // временный выбор в модалке
    let loadedModels = [];               // список моделей

    /* =====================================================
    LOAD CURRENT MODELS ON PAGE LOAD
    ===================================================== */

    async function loadCurrentModelFromBackend(type) {

        const response = await fetch("/api/models/get-current", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_type: type })
        });

        const result = await response.json();

        if (type === "chat") {
            currentChatLabel.textContent = result.name;
        } else {
            currentInstructLabel.textContent = result.name;
        }
    }

    // вызываем при старте страницы
    loadCurrentModelFromBackend("chat");
    loadCurrentModelFromBackend("instruct");


    /* =====================================================
       MODAL HELPERS
    ===================================================== */

    function openModal(modal) {
        modal.classList.remove("hidden");
        modalOverlay.classList.remove("hidden");
    }

    function closeModal(modal) {
        modal.classList.add("hidden");
        modalOverlay.classList.add("hidden");
    }

    document.querySelectorAll(".modal-close, .modal-cancel")
        .forEach(btn => {
            btn.addEventListener("click", () => {
                closeModal(btn.closest(".modal"));
                resetTempSelection();
            });
        });

    modalOverlay.addEventListener("click", () => {
        document.querySelectorAll(".modal")
            .forEach(m => m.classList.add("hidden"));
        modalOverlay.classList.add("hidden");
        resetTempSelection();
    });


    /* =====================================================
       OPEN SELECT MODAL
    ===================================================== */

    btnSelectChat.addEventListener("click", () => {
        currentMode = "chat";
        openSelectModal();
    });

    btnSelectInstruct.addEventListener("click", () => {
        currentMode = "instruct";
        openSelectModal();
    });

    async function openSelectModal() {
        selectTitle.textContent =
            currentMode === "chat"
                ? "Select model for chat"
                : "Select model for graph construction";

        backendCurrentModel =
            currentMode === "chat"
                ? currentChatLabel.textContent
                : currentInstructLabel.textContent;

        tempSelectedModel = backendCurrentModel;

        await loadModels();
        renderModelList();
        openModal(modalSelect);
    }


    /* =====================================================
       LOAD MODELS
    ===================================================== */

    async function loadModels() {

        const response = await fetch("/api/models/load-all", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_type: currentMode })
        });

        loadedModels = await response.json();

        // ИСПОЛЬЗУЕМ is_current
        const currentFromBackend =
            loadedModels.find(m => m.is_current);

        if (currentFromBackend) {
            backendCurrentModel = currentFromBackend.name;
            tempSelectedModel = currentFromBackend.name;
        }
    }

    /* =====================================================
       RENDER TEMPLATE LIST
    ===================================================== */

    function renderModelList() {

        modelListContainer.innerHTML = "";

        if (!loadedModels.length) {
            modelListContainer.innerHTML = "<p>No models available</p>";
            return;
        }

        loadedModels.forEach(model => {

            const clone = selectTemplate.content.cloneNode(true);
            const root = clone.querySelector(".select-item");

            root.dataset.name = model.name;

            clone.querySelector(".select-chat-name").textContent = model.name;

            clone.querySelector(".info-chat-model-name").textContent =
                `model: ${model.model_name}`;

            clone.querySelector(".info-chat-model-temp").textContent =
                `temp: ${model.temperature}`;

            clone.querySelector(".info-chat-model-tokens").textContent =
                `max tokens: ${model.max_tokens}`;

            const selectBtn = clone.querySelector(".select-chat-btn");
            const currentBadge = clone.querySelector(".selected-chat-btn");

            // фронтенд логика выбора
            selectBtn.addEventListener("click", () => {
                tempSelectedModel = model.name;
                updateFrontendSelection();
            });

            modelListContainer.appendChild(clone);
        });

        updateFrontendSelection();
    }


    /* =====================================================
       FRONTEND CURRENT SWITCHING
    ===================================================== */

    function updateFrontendSelection() {

        document.querySelectorAll("#model-list-container .select-item")
            .forEach(item => {

                const name = item.dataset.name;
                const selectBtn = item.querySelector(".select-chat-btn");
                const currentBadge = item.querySelector(".selected-chat-btn");

                if (name === tempSelectedModel) {
                    selectBtn.classList.add("hidden");
                    currentBadge.classList.remove("hidden");
                } else {
                    selectBtn.classList.remove("hidden");
                    currentBadge.classList.add("hidden");
                }
            });
    }

    function resetTempSelection() {
        tempSelectedModel = backendCurrentModel;
    }


    /* =====================================================
       CONFIRM SELECTION (ONLY HERE WE CALL BACKEND)
    ===================================================== */

    confirmSelectBtn.addEventListener("click", async () => {

        if (!tempSelectedModel ||
            tempSelectedModel === backendCurrentModel) {
            closeModal(modalSelect);
            return;
        }

        const response = await fetch("/api/models/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: tempSelectedModel,
                model_type: currentMode
            })
        });

        const result = await response.json();

        if (currentMode === "chat") {
            currentChatLabel.textContent = result.name;
        } else {
            currentInstructLabel.textContent = result.name;
        }

        backendCurrentModel = result.name;
        closeModal(modalSelect);
    });


    /* =====================================================
       OPEN CREATE MODAL
    ===================================================== */

    btnCreateChat.addEventListener("click", () => {
        currentMode = "chat";
        createTitle.textContent = "Create Chat Model";
        openModal(modalCreate);
    });

    btnCreateInstruct.addEventListener("click", () => {
        currentMode = "instruct";
        createTitle.textContent = "Create Graph Model";
        openModal(modalCreate);
    });


    /* =====================================================
       SHOW API FIELD
    ===================================================== */

    inputOption.addEventListener("change", () => {
        if (inputOption.value === "openai" ||
            inputOption.value === "hf_api") {
            apiKeyGroup.classList.remove("hidden");
        } else {
            apiKeyGroup.classList.add("hidden");
        }
    });


    /* =====================================================
       CREATE MODEL
    ===================================================== */

    confirmCreateBtn.addEventListener("click", async () => {

        const payload = {
            name: inputDisplayName.value,
            option: inputOption.value,
            model_name: inputModelName.value,
            temperature: parseFloat(inputTemp.value),
            max_tokens: parseInt(inputMaxTokens.value),
            api: inputApiKey.value,
            model_type: currentMode
        };

        const response = await fetch("/api/models/create-and-select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (currentMode === "chat") {
            currentChatLabel.textContent = result.name;
        } else {
            currentInstructLabel.textContent = result.name;
        }

        closeModal(modalCreate);
    });

});