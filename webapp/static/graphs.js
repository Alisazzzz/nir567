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

    // Создаем FormData для отправки файла
    const formData = new FormData();
    formData.append("document", file);
    formData.append("graph_filename", graphName);
    formData.append("embedding_model_name", selectedEmbeddingModelName);

    console.log(selectedEmbeddingModelName);
    // Важно: не устанавливаем Content-Type вручную для FormData, 
    // браузер сам добавит boundary
    const res = await fetch("/api/graph/create-and-select", {
        method: "POST",
        body: formData 
    });

    const data = await res.json();

    currentGraphName.textContent = data.filename;
    currentGraphDocument.textContent = data.document;

    reloadGraphVisualization();

    closeTopModal(); // loading
    closeTopModal(); // create modal
});

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

/* =========================================================
   GRAPH VISUALIZATION
========================================================= */

let network = null;
let selectedElement = null;
let selectedElementType = null; // 'node' or 'edge'

// Цвета для разных типов вершин
const nodeTypeColors = {
    location: '#FF6B6B',      // красный
    character: '#4ECDC4',     // бирюзовый
    item: '#FFE66D',          // желтый
    event: '#95E1D3',         // мятный
    organization: '#A8E6CF',  // светло-зеленый
    person: '#FF8B94',        // розовый
    default: '#74B9FF'        // голубой
};

// Функция получения цвета для типа вершины
function getNodeColor(nodeType) {
    const type = nodeType ? nodeType.toLowerCase() : 'default';
    return nodeTypeColors[type] || nodeTypeColors.default;
}

async function reloadGraphVisualization() {
    if (!currentGraphName.textContent ||
        currentGraphName.textContent === "no graph selected") return;

    try {
        // Получаем данные графа с бэкенда
        const res = await fetch("/api/graph/visualize");
        const graphData = await res.json();

        // Преобразуем узлы для Vis.js
        const visNodes = new vis.DataSet(
            graphData.nodes.map(node => ({
                id: node.id,
                label: node.name,
                title: node.type, // tooltip
                color: {
                    background: getNodeColor(node.type),
                    border: '#2C3E50',
                    highlight: {
                        background: getNodeColor(node.type),
                        border: '#E74C3C'
                    }
                },
                shape: 'dot',
                size: 25,
                font: {
                    size: 14,
                    color: '#2C3E50',
                    face: 'Arial'
                },
                borderWidth: 2,
                shadow: true
            }))
        );

        // Преобразуем рёбра для Vis.js
        const visEdges = new vis.DataSet(
            graphData.edges.map(edge => ({
                id: edge.id,
                from: edge.source,
                to: edge.target,
                label: edge.relation,
                title: edge.relation, // tooltip
                width: Math.max(1, edge.weight),
                color: {
                    color: '#95A5A6',
                    highlight: '#E74C3C'
                },
                font: {
                    size: 11,
                    color: '#7F8C8D',
                    align: 'middle'
                },
                smooth: {
                    type: 'continuous',
                    roundness: 0.2
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.5,
                        type: 'arrow'
                    }
                }
            }))
        );

        const container = document.getElementById("graph-container");

        // Опции для сети
        const options = {
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 200
                },
                barnesHut: {
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            nodes: {
                shape: 'dot',
                size: 25,
                font: {
                    size: 14,
                    face: 'Arial'
                },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1,
                font: {
                    size: 11,
                    align: 'middle'
                },
                smooth: {
                    type: 'continuous'
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.5
                    }
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true,
                dragNodes: true,
                hideEdgesOnDrag: false,
                selectConnectedEdges: false
            }
        };

        // Создаём сеть
        if (network) {
            network.destroy();
        }
        
        network = new vis.Network(container, { nodes: visNodes, edges: visEdges }, options);

        // Обработчик клика
        network.on("click", (params) => {
            handleGraphClick(params);
        });

        // Обновляем легенду
        updateLegend(graphData.nodes);

    } catch (error) {
        console.error("Error loading graph visualization:", error);
    }
}

// Обработка клика по графу
async function handleGraphClick(params) {
    // Сбрасываем предыдущее выделение
    if (network && selectedElement) {
        network.unselectAll();
    }

    // Если кликнули на узел
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        selectedElement = nodeId;
        selectedElementType = 'node';
        
        // Выделяем узел
        network.selectNodes([nodeId]);
        
        // Получаем подробную информацию
        await loadNodeInfo(nodeId);
        return;
    }

    // Если кликнули на ребро
    if (params.edges.length > 0) {
        const edgeId = params.edges[0];
        selectedElement = edgeId;
        selectedElementType = 'edge';
        
        // Выделяем ребро
        network.selectEdges([edgeId]);
        
        // Получаем подробную информацию
        await loadEdgeInfo(edgeId);
        return;
    }

    // Если кликнули на пустое место - снимаем выделение
    selectedElement = null;
    selectedElementType = null;
    hideElementInfo();
}

// Загрузка информации о вершине
async function loadNodeInfo(nodeId) {
    try {
        const response = await fetch("/api/graph/get-node-info", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ id: nodeId })
        });

        if (!response.ok) {
            throw new Error("Failed to load node info");
        }

        const nodeInfo = await response.json();
        displayNodeInfo(nodeInfo);
        
    } catch (error) {
        console.error("Error loading node info:", error);
    }
}

// Загрузка информации о ребре
async function loadEdgeInfo(edgeId) {
    try {
        console.log(edgeId);
        const response = await fetch("/api/graph/get-edge-info", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ id: edgeId })
        });

        if (!response.ok) {
            throw new Error("Failed to load edge info");
        }

        const edgeInfo = await response.json();
        displayEdgeInfo(edgeInfo);
        
    } catch (error) {
        console.error("Error loading edge info:", error);
    }
}

// Отображение информации о вершине
function displayNodeInfo(nodeInfo) {
    const container = document.getElementById("element-info-container");
    const template = document.getElementById("template-node-info");
    
    // Очищаем контейнер
    container.innerHTML = "";
    
    // Клонируем шаблон
    const clone = template.content.cloneNode(true);
    
    // Заполняем данные
    const entityType = clone.querySelector(".entity-type");
    const entityName = clone.querySelector(".entity-name");
    const entityDescription = clone.querySelector(".entity-description");
    const entityAttributes = clone.querySelector(".entity-attributes");
    const entityStates = clone.querySelector(".entity-states");
    
    // Тип и имя
    entityType.textContent = nodeInfo.type || "Unknown Type";
    entityType.style.color = getNodeColor(nodeInfo.type);
    entityName.textContent = nodeInfo.name || "Unnamed";
    
    // Описание
    if (nodeInfo.base_description) {
        entityDescription.innerHTML = `<strong>Description:</strong><p>${nodeInfo.base_description}</p>`;
    } else {
        entityDescription.innerHTML = "";
    }
    
    // Атрибуты
    if (nodeInfo.base_attributes && Object.keys(nodeInfo.base_attributes).length > 0) {
        let attrsHtml = "<strong>Attributes:</strong><ul>";
        for (const [key, value] of Object.entries(nodeInfo.base_attributes)) {
            attrsHtml += `<li><strong>${key}:</strong> ${value}</li>`;
        }
        attrsHtml += "</ul>";
        entityAttributes.innerHTML = attrsHtml;
    } else {
        entityAttributes.innerHTML = "";
    }
    
    // Состояния
    if (nodeInfo.states && nodeInfo.states.length > 0) {
        let statesHtml = "<strong>States:</strong><div class='states-list'>";
        nodeInfo.states.forEach((state, index) => {
            statesHtml += `
                <div class='state-item'>
                    <div class='state-header'>State ${index + 1}</div>
                    ${state.current_description ? `<p>${state.current_description}</p>` : ''}
                    ${state.time_start_event ? `<div><small>Start: ${state.time_start_event}</small></div>` : ''}
                    ${state.time_end_event ? `<div><small>End: ${state.time_end_event}</small></div>` : ''}
                </div>
            `;
        });
        statesHtml += "</div>";
        entityStates.innerHTML = statesHtml;
    } else {
        entityStates.innerHTML = "";
    }
    
    // Добавляем в контейнер
    container.appendChild(clone);
    
    // Показываем панель
    document.getElementById("element-info-card").classList.remove("hidden");
}

// Отображение информации о ребре
function displayEdgeInfo(edgeInfo) {
    const container = document.getElementById("element-info-container");
    const template = document.getElementById("template-edge-info");
    
    // Очищаем контейнер
    container.innerHTML = "";
    
    // Клонируем шаблон
    const clone = template.content.cloneNode(true);
    
    // Заполняем данные
    const relationName = clone.querySelector(".relation-name");
    const relationTimestamps = clone.querySelector(".relation-timestamps");
    const relationDescription = clone.querySelector(".relation-description");
    const relationWeight = clone.querySelector(".relation-weight");
    
    // Название отношения
    relationName.innerHTML = `<strong>Relation:</strong> ${edgeInfo.relation || "Unknown"}`;
    
    // Временные метки
    if (edgeInfo.time_start_event || edgeInfo.time_end_event) {
        let timeHtml = "<strong>Timestamps:</strong><ul>";
        if (edgeInfo.time_start_event) {
            timeHtml += `<li>Start: ${edgeInfo.time_start_event}</li>`;
        }
        if (edgeInfo.time_end_event) {
            timeHtml += `<li>End: ${edgeInfo.time_end_event}</li>`;
        }
        timeHtml += "</ul>";
        relationTimestamps.innerHTML = timeHtml;
    } else {
        relationTimestamps.innerHTML = "";
    }
    
    // Описание
    if (edgeInfo.description) {
        relationDescription.innerHTML = `<strong>Description:</strong><p>${edgeInfo.description}</p>`;
    } else {
        relationDescription.innerHTML = "";
    }
    
    // Вес
    relationWeight.innerHTML = `<strong>Weight:</strong> ${edgeInfo.weight || 1.0}`;
    
    // Добавляем в контейнер
    container.appendChild(clone);
    
    // Показываем панель
    document.getElementById("element-info-card").classList.remove("hidden");
}

// Скрыть информацию об элементе
function hideElementInfo() {
    const container = document.getElementById("element-info-container");
    container.innerHTML = "";
    document.getElementById("element-info-card").classList.add("hidden");
}

// Обновление легенды
function updateLegend(nodes) {
    const legendContainer = document.querySelector("#legend-card .legend");
    if (!legendContainer) return;
    
    // Получаем уникальные типы
    const types = [...new Set(nodes.map(node => node.type).filter(Boolean))];
    
    // Очищаем легенду (кроме статических элементов, если есть)
    legendContainer.innerHTML = "";
    
    // Добавляем элементы легенды
    types.forEach(type => {
        const legendItem = document.createElement("div");
        legendItem.className = "legend-item";
        
        const colorDiv = document.createElement("div");
        colorDiv.className = "legend-color";
        colorDiv.style.backgroundColor = getNodeColor(type);
        
        const span = document.createElement("span");
        span.textContent = type.charAt(0).toUpperCase() + type.slice(1) + "s";
        
        legendItem.appendChild(colorDiv);
        legendItem.appendChild(span);
        legendContainer.appendChild(legendItem);
    });
}

// Инициализация при загрузке страницы
document.addEventListener("DOMContentLoaded", () => {
    // Дополнительные обработчики для переключения вкладок
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.dataset.tab === "graph" && network) {
                // Перерисовываем сеть при переключении на вкладку графа
                setTimeout(() => {
                    if (network) {
                        network.fit();
                        network.redraw();
                    }
                }, 100);
            }
        });
    });
});