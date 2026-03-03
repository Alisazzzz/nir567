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

        tabContents.forEach(content => content.classList.remove("active"));

        const target = btn.dataset.tab;
        document.getElementById("tab-" + target).classList.add("active");

        // показываем правую панель только на graph вкладке
        if (target === "graph") {
            infoPanel.classList.remove("hidden");
        } else {
            infoPanel.classList.add("hidden");
        }
    });
});


/* =========================================================
   MODAL HELPERS
========================================================= */

const modalOverlay = document.getElementById("modal-overlay");

function openModal(modal) {
    modal.classList.remove("hidden");
    modalOverlay.classList.remove("hidden");
}

function closeAllModals() {
    document.querySelectorAll(".modal").forEach(m => m.classList.add("hidden"));
    modalOverlay.classList.add("hidden");
}

document.querySelectorAll(".modal-close, .modal-cancel")
    .forEach(btn => btn.addEventListener("click", closeAllModals));


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
    try {
        const response = await fetch("/api/graph/get-current");
        const data = await response.json();

        if (data.is_current) {
            currentGraphName.textContent = data.filename;
            currentGraphDocument.textContent = data.document;
            reloadGraphVisualization();
        }

    } catch (err) {
        console.error(err);
    }
}

document.addEventListener("DOMContentLoaded", loadCurrentGraph);


/* =========================================================
   SELECT GRAPH
========================================================= */

btnSelectGraph.addEventListener("click", async () => {
    await loadAllGraphs();
    openModal(modalSelectGraph);
});

async function loadAllGraphs() {
    graphListContainer.innerHTML = "";

    const response = await fetch("/api/graph/load-all");
    const graphs = await response.json();

    graphs.forEach(graph => {
        const item = document.createElement("div");
        item.className = "select-item";
        item.textContent = graph.filename;

        item.addEventListener("click", () => {
            document.querySelectorAll(".select-item")
                .forEach(i => i.classList.remove("selected"));

            item.classList.add("selected");
            selectedGraphTemp = graph.filename;
        });

        graphListContainer.appendChild(item);
    });
}

confirmSelectGraphBtn.addEventListener("click", async () => {
    if (!selectedGraphTemp) return;

    const response = await fetch("/api/graph/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filepath: selectedGraphTemp })
    });

    const data = await response.json();

    currentGraphName.textContent = data.existing_graph.filename;
    currentGraphDocument.textContent = data.existing_graph.document;

    if (data.chat_history && data.chat_history.history) {
        if (typeof loadChatHistory === "function") {
            loadChatHistory(data.chat_history.history);
        }
    }

    reloadGraphVisualization();
    closeAllModals();
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
        alert("Please fill all fields.");
        return;
    }

    closeAllModals();
    openModal(modalLoading);

    const response = await fetch("/api/graph/create-and-select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            document_filepath: file.name,
            graph_filename: graphName,
            embedding_model_name: selectedEmbeddingModelName
        })
    });

    const data = await response.json();

    currentGraphName.textContent = data.filename;
    currentGraphDocument.textContent = data.document;

    reloadGraphVisualization();
    closeAllModals();
});


/* =========================================================
   GRAPH VISUALIZATION
========================================================= */

let network = null;

async function reloadGraphVisualization() {

    if (!currentGraphName.textContent ||
        currentGraphName.textContent === "no graph selected") return;

    const response = await fetch("/assets/graphs/" + currentGraphName.textContent);
    const graphData = await response.json();

    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const container = document.getElementById("graph-container");

    network = new vis.Network(container, { nodes, edges }, {
        physics: { stabilization: false }
    });

    network.on("click", params => handleGraphClick(params, graphData));
}


/* =========================================================
   GRAPH CLICK HANDLING
========================================================= */

function handleGraphClick(params, graphData) {

    const container = document.getElementById("element-info-container");
    container.innerHTML = "";

    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = graphData.nodes.find(n => n.id === nodeId);

        container.innerHTML = `
            <div><strong>${node.label}</strong></div>
            <div>Type: ${node.type}</div>
            <div>${node.description || ""}</div>
        `;
    }

    if (params.edges.length > 0) {
        const edgeId = params.edges[0];
        const edge = graphData.edges.find(e => e.id === edgeId);

        container.innerHTML = `
            <div><strong>${edge.label}</strong></div>
            <div>${edge.description || ""}</div>
        `;
    }
}


/* =========================================================
   EMBEDDING LOGIC
========================================================= */

const btnSelectEmbedding = document.getElementById("btn-select-embedding");
const modalSelectEmbedding = document.getElementById("modal-select-embedding");
const modalCreateEmbedding = document.getElementById("modal-create-embedding");

const embeddingListContainer = document.getElementById("embedding-list-container");

const inputEmbeddingName = document.getElementById("input-embedding-name");
const inputEmbeddingOption = document.getElementById("input-embedding-option");
const inputEmbeddingModelName = document.getElementById("input-embedding-model-name");
const inputEmbeddingApi = document.getElementById("input-embedding-api");

let selectedEmbeddingTemp = null;


/* ===== OPEN SELECT EMBEDDING ===== */

btnSelectEmbedding.addEventListener("click", async () => {
    await loadAllEmbeddings();
    openModal(modalSelectEmbedding);
});


/* ===== LOAD EMBEDDINGS ===== */

async function loadAllEmbeddings() {

    embeddingListContainer.innerHTML = "";

    const response = await fetch("/api/embedding/load-all");
    const embeddings = await response.json();

    embeddings.forEach(model => {
        const item = document.createElement("div");
        item.className = "select-item";
        item.textContent = model.name;

        item.addEventListener("click", () => {
            document.querySelectorAll("#embedding-list-container .select-item")
                .forEach(i => i.classList.remove("selected"));

            item.classList.add("selected");
            selectedEmbeddingTemp = model.name;
        });

        embeddingListContainer.appendChild(item);
    });
}


/* ===== CONFIRM SELECT EMBEDDING ===== */

document.getElementById("confirm-select-embedding")
    ?.addEventListener("click", async () => {

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
        document.getElementById("selected-embedding-model").textContent =
            selectedEmbeddingModelName;

        closeAllModals();
    });


/* ===== CREATE EMBEDDING ===== */

document.getElementById("btn-create-embedding")
    ?.addEventListener("click", () => {
        openModal(modalCreateEmbedding);
    });


// открыть создание embedding из выбора
document.getElementById("btn-open-create-embedding")
    ?.addEventListener("click", () => {
        document.getElementById("modal-select-embedding").classList.add("hidden");
        openModal(document.getElementById("modal-create-embedding"));
});

document.getElementById("confirm-create-embedding")
    ?.addEventListener("click", async () => {

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
                name: name,
                option: option,
                model_name: modelName,
                api_info: api
            })
        });

        selectedEmbeddingModelName = name;
        document.getElementById("selected-embedding-model").textContent = name;

        closeAllModals();
    });


/* ===== SHOW/HIDE API FIELD ===== */

const embeddingApiGroup = document.getElementById("embedding-api-group");

inputEmbeddingOption?.addEventListener("change", () => {

    if (inputEmbeddingOption.value === "openai") {
        embeddingApiGroup.classList.remove("hidden");
    } else {
        embeddingApiGroup.classList.add("hidden");
    }
});