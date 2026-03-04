/* =========================================================
   TAB SWITCHING (CHAT <-> GRAPH)
========================================================= */

const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");
const infoPanel = document.getElementById("info-panel");

tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {

        tabButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        tabContents.forEach(c => c.classList.remove("active"));
        document.getElementById("tab-" + btn.dataset.tab)
            .classList.add("active");

        if (btn.dataset.tab === "graph") {
            infoPanel.classList.remove("hidden");
        } else {
            infoPanel.classList.add("hidden");
        }
    });
});


/* =========================================================
   MODAL STACK SYSTEM
========================================================= */

const modalOverlay = document.getElementById("modal-overlay");
let modalStack = [];

function openModal(modal) {
    modal.classList.remove("hidden");
    modalStack.push(modal);
    modalOverlay.classList.remove("hidden");
}

function closeTopModal() {
    const modal = modalStack.pop();
    if (modal) modal.classList.add("hidden");

    if (modalStack.length === 0) {
        modalOverlay.classList.add("hidden");
    }
}

document.querySelectorAll(".modal-close, .modal-cancel")
    .forEach(btn => btn.addEventListener("click", closeTopModal));


/* =========================================================
   GRAPH VARIABLES
========================================================= */

const btnSelectGraph = document.getElementById("btn-select-graph");
const btnCreateGraph = document.getElementById("btn-create-graph");

const modalSelectGraph = document.getElementById("modal-select-graph");
const modalCreateGraph = document.getElementById("modal-create-graph");
const modalLoading = document.getElementById("modal-loading");

const graphListContainer = document.getElementById("graph-list-container");
const confirmSelectGraphBtn = document.getElementById("confirm-select-graph");
const confirmCreateGraphBtn = document.getElementById("btn-confirm-create-graph");

const inputGraphDocument = document.getElementById("input-graph-document");
const inputGraphName = document.getElementById("input-graph-name");

const currentGraphName = document.getElementById("current-graph-name");
const currentGraphDocument = document.getElementById("current-graph-document");

let selectedGraphTemp = null;
let selectedEmbeddingModelName = null;


/* =========================================================
   LOAD CURRENT GRAPH
========================================================= */

async function loadCurrentGraph() {
    const res = await fetch("/api/graph/get-current");
    const data = await res.json();

    currentGraphName.textContent = data.filename;
    currentGraphDocument.textContent = data.document;

    if (data.is_current) {
        reloadGraphVisualization();
    }
}

document.addEventListener("DOMContentLoaded", loadCurrentGraph);


/* =========================================================
   SELECT GRAPH (FRONTEND SELECTION ONLY)
========================================================= */

btnSelectGraph.addEventListener("click", async () => {
    await loadAllGraphs();
    openModal(modalSelectGraph);
});

async function loadAllGraphs() {

    graphListContainer.innerHTML = "";
    selectedGraphTemp = null;

    const res = await fetch("/api/graph/load-all");
    const graphs = await res.json();

    const template = document.getElementById("graph-model-info-template");

    graphs.forEach(graph => {

        const clone = template.content.cloneNode(true);
        const item = clone.querySelector(".select-item");

        const nameEl = clone.querySelector(".select-graph-name");
        const docEl = clone.querySelector(".select-graph-document");
        const selectBtn = clone.querySelector(".select-graph-btn");
        const currentLabel = clone.querySelector(".selected-graph-btn");

        nameEl.textContent = graph.filename;
        docEl.textContent = graph.document;

        // начальное состояние из backend
        if (graph.is_current) {
            selectBtn.classList.add("hidden");
            currentLabel.classList.remove("hidden");
        }

        selectBtn.addEventListener("click", () => {

            // сброс состояния у всех
            document.querySelectorAll(
                "#graph-list-container .select-item"
            ).forEach(card => {

                card.classList.remove("selected");

                const btn = card.querySelector(".select-graph-btn");
                const lbl = card.querySelector(".selected-graph-btn");

                btn.classList.remove("hidden");
                lbl.classList.add("hidden");
            });

            // активируем текущую карточку
            item.classList.add("selected");
            selectBtn.classList.add("hidden");
            currentLabel.classList.remove("hidden");

            selectedGraphTemp = graph.filename;
        });

        graphListContainer.appendChild(clone);
    });
}


/* ===== CONFIRM GRAPH SELECTION ===== */

confirmSelectGraphBtn.addEventListener("click", async () => {

    if (!selectedGraphTemp) return;

    const res = await fetch("/api/graph/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: selectedGraphTemp })
    });

    const data = await res.json();

    currentGraphName.textContent = data.existing_graph.filename;
    currentGraphDocument.textContent = data.existing_graph.document;

    if (data.chat_history?.history && typeof loadChatHistory === "function") {
        loadChatHistory(data.chat_history.history);
    }

    reloadGraphVisualization();
    closeTopModal();
});


/* =========================================================
   CREATE GRAPH
========================================================= */

btnCreateGraph.addEventListener("click", () => {
    openModal(modalCreateGraph);
});

confirmCreateGraphBtn.addEventListener("click", async () => {

    const file = inputGraphDocument.files[0];
    const graphName = inputGraphName.value.trim();

    if (!file || !graphName || !selectedEmbeddingModelName) {
        alert("Fill all fields.");
        return;
    }

    openModal(modalLoading);

    const res = await fetch("/api/graph/create-and-select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            document_filepath: file.name,
            graph_filename: graphName,
            embedding_model_name: selectedEmbeddingModelName
        })
    });

    const data = await res.json();

    currentGraphName.textContent = data.filename;
    currentGraphDocument.textContent = data.document;

    reloadGraphVisualization();

    closeTopModal(); // loading
    closeTopModal(); // create modal
});


/* =========================================================
   GRAPH VISUALIZATION
========================================================= */

let network = null;

async function reloadGraphVisualization() {

    if (!currentGraphName.textContent ||
        currentGraphName.textContent === "no graph selected") return;

    const res = await fetch("/assets/graphs/" + currentGraphName.textContent);
    const graphData = await res.json();

    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const container = document.getElementById("graph-container");

    network = new vis.Network(container, { nodes, edges }, {
        physics: { stabilization: false }
    });

    network.on("click", params =>
        handleGraphClick(params, graphData));
}


/* =========================================================
   EMBEDDING LOGIC
========================================================= */

const btnSelectEmbedding = document.getElementById("btn-select-embedding");
const btnOpenCreateEmbedding = document.getElementById("btn-open-create-embedding");

const modalSelectEmbedding = document.getElementById("modal-select-embedding");
const modalCreateEmbedding = document.getElementById("modal-create-embedding");

const embeddingListContainer = document.getElementById("embedding-list-container");

const inputEmbeddingName = document.getElementById("input-embedding-name");
const inputEmbeddingOption = document.getElementById("input-embedding-option");
const inputEmbeddingModelName = document.getElementById("input-embedding-model-name");
const inputEmbeddingApi = document.getElementById("input-embedding-api");

let selectedEmbeddingTemp = null;


/* ===== OPEN EMBEDDING SELECT ===== */

btnSelectEmbedding.addEventListener("click", async () => {
    await loadAllEmbeddings();
    openModal(modalSelectEmbedding);
});


/* ===== LOAD EMBEDDINGS ===== */

async function loadAllEmbeddings() {

    embeddingListContainer.innerHTML = "";
    selectedEmbeddingTemp = null;

    const res = await fetch("/api/embedding/load-all");
    const embeddings = await res.json();

    const template = document.getElementById("embedding-model-info-template");

    embeddings.forEach(model => {

        const clone = template.content.cloneNode(true);
        const item = clone.querySelector(".select-item");

        const nameEl = clone.querySelector(".select-embedding-name");
        const infoEl = clone.querySelector(".select-embedding-info");
        const selectBtn = clone.querySelector(".select-embedding-btn");
        const currentLabel = clone.querySelector(".selected-embedding-btn");

        nameEl.textContent = model.name;
        infoEl.textContent = `${model.option} | ${model.model_name || ""}`;

        if (model.name === selectedEmbeddingModelName) {
            selectBtn.classList.add("hidden");
            currentLabel.classList.remove("hidden");
        } else {

            selectBtn.addEventListener("click", () => {

                document.querySelectorAll(
                    "#embedding-list-container .select-item"
                ).forEach(i => i.classList.remove("selected"));

                item.classList.add("selected");
                selectedEmbeddingTemp = model.name;
            });
        }

        embeddingListContainer.appendChild(clone);
    });
}


/* ===== CONFIRM EMBEDDING ===== */

document.getElementById("confirm-select-embedding")
    .addEventListener("click", async () => {

        if (!selectedEmbeddingTemp) return;

        await fetch("/api/embedding/select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: selectedEmbeddingTemp,
                model_type: "embedding"
            })
        });

        selectedEmbeddingModelName = selectedEmbeddingTemp;

        document.getElementById("current-embedding-name")
            .textContent = selectedEmbeddingTemp;

        document.getElementById("selected-embedding-model")
            .textContent = selectedEmbeddingTemp;

        closeTopModal();
    });


/* ===== CREATE EMBEDDING ===== */

btnOpenCreateEmbedding.addEventListener("click", () => {
    openModal(modalCreateEmbedding);
});

document.getElementById("confirm-create-embedding")
    .addEventListener("click", async () => {

        const name = inputEmbeddingName.value.trim();
        const option = inputEmbeddingOption.value;
        const modelName = inputEmbeddingModelName.value.trim();
        const api = inputEmbeddingApi.value.trim() || null;

        if (!name || !option || !modelName) {
            alert("Fill all required fields.");
            return;
        }

        await fetch("/api/embedding/create-and-select", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name,
                option,
                model_name: modelName,
                api_info: api
            })
        });

        selectedEmbeddingModelName = name;

        document.getElementById("current-embedding-name").textContent = name;
        document.getElementById("selected-embedding-model").textContent = name;

        closeTopModal();
        closeTopModal();
    });


/* ===== API FIELD VISIBILITY ===== */

const embeddingApiGroup = document.getElementById("embedding-api-group");

inputEmbeddingOption.addEventListener("change", () => {
    if (inputEmbeddingOption.value === "openai") {
        embeddingApiGroup.classList.remove("hidden");
    } else {
        embeddingApiGroup.classList.add("hidden");
    }
});