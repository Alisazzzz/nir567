// js-file with all frontend logic



document.addEventListener("DOMContentLoaded", () => {

    //get current active graph if server saved it
    getCurrentGraph();

    //bind some tricky events to buttons and elements
    bindButtons();
});

async function getCurrentGraph() {
    try {
        const response = await fetch("/api/graph/get_current");
        const selected = await response.json();
        document.getElementById("current-graph-btn").textContent = selected.filename;
        document.getElementById("graph-document-name").textContent = selected.document;
    } catch (error) {
        document.getElementById("current-graph-btn").textContent = "error loading graph";
        document.getElementById("graph-document-name").textContent = "no document selected";
    }
}

function bindButtons() {
    document.getElementById("graph-list").addEventListener("click", (event) => {
        const button = event.target.closest(".graph-select-btn");
        if (!button) return;
        const graphItem = button.closest(".graph-item");
        const filepath = graphItem.dataset.filepath;
        selectGraphDraft(filepath);
    });
    
    //chat section
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


    document.getElementById("model-list").addEventListener("click", (event) => {
        const button = event.target.closest(".model-select-btn");
        if (!button) return;
        const modelItem = button.closest(".model-item");
        const modelName = modelItem.dataset.model_name;
        selectModel(modelName);
    });

    //in model selection modal window
    const modelTabButtons = document.querySelectorAll('#model-select-modal .tab-btn');
    modelTabButtons.forEach(button => {
        button.addEventListener('click', function() {
            modelTabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('#model-select-modal .tab-content').forEach(content => content.classList.remove('active'));           
            this.classList.add('active');
            
            const tabName = this.getAttribute('data-tab');
            const tabContent = document.getElementById(`${tabName}-tab`);
            tabContent.classList.add('active');
            
            if (tabName === 'create') {
                draftModelData = {
                    name: '',
                    option: 'ollama',
                    model_name: '',
                    max_tokens: 1024,
                    temperature: 0.7,
                    api: ''
                };
                updateCreateForm();
            }
        });
    });
    
    document.getElementById('connection-type').addEventListener('change', function() {
        if (!draftModelData) return; 
        draftModelData.option = this.value;
        updateCreateForm();
    });
    
    document.getElementById('model-name').addEventListener('input', function() {
        if (!draftModelData) return;
        draftModelData.model_name = this.value;
    });

    document.getElementById('temperature').addEventListener('input', function() {
        if (!draftModelData) return;
        draftModelData.temperature = parseFloat(this.value);
    });

    document.getElementById('max-tokens').addEventListener('input', function() {
        if (!draftModelData) return;
        draftModelData.max_tokens = parseInt(this.value);
    });

    document.getElementById('model-display-name').addEventListener('input', function() {
        if (!draftModelData) return;
        draftModelData.name = this.value;
    });

    document.getElementById('api-key').addEventListener('input', function() {
        if (!draftModelData) return;
        draftModelData.api = this.value;
    });
}

// =====================================================
// CHAT
// =====================================================

async function sendMessage() {
    const text = document.getElementById("chat-input").value.trim();
    if (!text) return;

    addChatMessage("user", text);
    document.getElementById("chat-input").value = "";

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        addChatMessage("assistant", data.answer);

    } catch (error) {
        console.error("[FRONT] Chat error:", error);
        addChatMessage("assistant", "Error communicating with server");
    }
}

function addChatMessage(role, text) {
    const chatWindow = document.getElementById("chat-window");
    const messageDiv = document.createElement("div");

    messageDiv.style.border = "1px solid gray";
    messageDiv.style.margin = "5px";
    messageDiv.style.padding = "5px";

    messageDiv.innerHTML = `<strong>${role}:</strong> ${text}`;
    chatWindow.appendChild(messageDiv);

    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// =====================================================
// GRAPH
// =====================================================

async function updateGraphFromChat() {
    try {
        const response = await fetch("/api/graph/update", { method: "POST" });
        const data = await response.json();
        
        alert(
            `Graph updated!\nNodes: ${data.nodes_added}\nEdges: ${data.edges_added}`
        );
        
        // Обновляем визуализацию, если вкладка активна
        if (document.getElementById('graph-tab').classList.contains('active')) {
            loadAndRenderGraph();
        }
    } catch (error) {
        console.error("[FRONT] Graph update error:", error);
        alert("Error updating graph");
    }
}

// =====================================================
// MODALS
// =====================================================

function openGraphCreateModal() {
    document.getElementById("graph-create-modal").style.display = "block";
    document.getElementById("modal-overlay").style.display = "block";
}

function closeGraphCreateModal() {
    document.getElementById("graph-create-modal").style.display = "none";
    document.getElementById("modal-overlay").style.display = "none";
}

function closeModelSelectModal() {
    document.getElementById("model-select-modal").style.display = "none";
    document.getElementById("modal-overlay").style.display = "none";
}

// choosing graph modal window
let confirmedGraphPath = null;
let draftGraphPath = null;

function openGraphSelectModal() {
    loadGraphs();
    document.getElementById("modal-overlay").style.display = "block";
    document.getElementById("graph-select-modal").style.display = "block";
}

async function loadGraphs() {
    try {
        const response = await fetch("/api/graph/load-all");
        const graphs = await response.json();      
        if (!Array.isArray(graphs) || graphs.length === 0) {
            document.getElementById("if-no-graph").style.display = "block";
            document.getElementById("if-no-graph").textContent = "No graphs available. Create a new one";
            return;
        } else { 
            document.getElementById("if-no-graph").style.display = "none"; 
        }
        renderGraphList(graphs);
    } catch (error) {
        console.error("Error loading graphs:", error);
        document.getElementById("if-no-graph").style.display = "block";
        document.getElementById("if-no-graph").textContent = "Error loading graphs";
    }
}

function renderGraphList(graphs) {
    const container = document.getElementById("graph-list");
    const template = document.getElementById("graph-item-template");
    container.innerHTML = "";

    graphs.forEach(graph => {
        const clone = template.content.cloneNode(true);
        const graphItem = clone.querySelector(".graph-item");
        graphItem.dataset.filepath = graph.filename;
        clone.querySelector(".graph-name").textContent = graph.filename;
        clone.querySelector(".graph-document").textContent = `Document: ${graph.document}`;

        const button = clone.querySelector(".graph-select-btn");
        if (graph.is_current) {
            button.textContent = "Current graph";
            button.disabled = true;
            button.setAttribute("data-current", "true");
            confirmedGraphPath = graph.filename;
            draftGraphPath = confirmedGraphPath;
        } else { 
            button.textContent = "Select";
        }
        container.appendChild(clone);
    });
}

function selectGraphDraft(filepath) {
    if (draftGraphPath === filepath) return;
    updateGraphSelectionUI(draftGraphPath, filepath);
    draftGraphPath = filepath;
}

function updateGraphSelectionUI(prevPath, nextPath) {
    if (prevPath) {
        const prevItem = document.querySelector(`.graph-item[data-filepath="${CSS.escape(prevPath)}"]`);
        const prevBtn = prevItem?.querySelector(".graph-select-btn");
        if (prevBtn) {
            prevBtn.textContent = "Select";
            prevBtn.disabled = false;
            prevBtn.removeAttribute("data-current");
        }
    }
    if (nextPath) {
        const nextItem = document.querySelector(`.graph-item[data-filepath="${CSS.escape(nextPath)}"]`);
        const nextBtn = nextItem?.querySelector(".graph-select-btn");
        if (nextBtn) {
            nextBtn.textContent = "Current graph";
            nextBtn.disabled = true;
            nextBtn.setAttribute("data-current", "true");
        }
    }
}

async function saveGraphSelection() {
    if (!draftGraphPath || draftGraphPath === confirmedGraphPath) {
        closeGraphSelectModal();       
        return;
    }
    try {
        const response = await fetch("/api/graph/select_graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filepath: draftGraphPath })
        });
        const selected = await response.json();
        document.getElementById("current-graph-btn").textContent = selected.filename;
        document.getElementById("graph-document-name").textContent = selected.document;

        if (document.getElementById('graph-tab').classList.contains('active')) {
            loadAndRenderGraph();
        }        
        confirmedGraphPath = draftGraphPath;
        closeGraphSelectModal();
    } catch (error) {
        console.error("Error selecting graph:", error);
    }
}

function cancelGraphSelection() {
    if (draftGraphPath !== confirmedGraphPath) {
        updateGraphSelectionUI(draftGraphPath, confirmedGraphPath);
        draftGraphPath = confirmedGraphPath;
    }
    closeGraphSelectModal();
}

function closeGraphSelectModal() {
    document.getElementById("graph-select-modal").style.display = "none";
    document.getElementById("modal-overlay").style.display = "none";
}

// visualizing graph 
let network = null;
let isGraphLoaded = false;

function showGraphMessage(message) {
    const container = document.getElementById("graph-container");
    if (!container) return;
    
    // Сохраняем оригинальное содержимое
    container.dataset.originalContent = container.innerHTML;
    
    container.innerHTML = `
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: #fff;
            font-size: 16px;
            text-align: center;
            padding: 20px;
        ">
            ${message}
        </div>
    `;
}

function hideGraphMessage() {
    const container = document.getElementById("graph-container");
    if (!container) return;
    
    // Восстанавливаем оригинальное содержимое (если было)
    if (container.dataset.originalContent) {
        container.innerHTML = container.dataset.originalContent;
        delete container.dataset.originalContent;
    }
}

function showPanel(title, data) {
    const vertexDetails = document.getElementById("vertex-details");
    if (!vertexDetails) return;
    
    let content = `<h3>${title}</h3>`;
    
    if (Object.keys(data).length > 0) {
        content += '<div style="margin-top: 10px;">';
        for (const [key, value] of Object.entries(data)) {
            content += `<div><strong>${key}:</strong> ${JSON.stringify(value, null, 2)}</div>`;
        }
        content += '</div>';
    } else {
        content += '<div style="margin-top: 10px; color: #aaa;">No additional data</div>';
    }
    
    vertexDetails.innerHTML = content;
    // Убрано display: block, так как панель теперь всегда видима
}

function hidePanel() {
    // Удалена строка, так как панель информации теперь всегда видима
}

function attachGraphHandlers(network) {
    network.on("click", function (params) {
        if (params.nodes.length === 0 && params.edges.length === 0) {
            hidePanel();
            return;
        }

        if (params.nodes.length === 1) {
            const nodeId = params.nodes[0];
            const node = network.body.data.nodes.get(nodeId);
            showPanel("Node: " + nodeId, node.data || {});
            return;
        }

        if (params.edges.length === 1) {
            const edgeId = params.edges[0];
            const edge = network.body.data.edges.get(edgeId);
            showPanel(
                "Edge: " + edge.from + " → " + edge.to,
                edge.data || {}
            );
        }
    });
}

async function loadAndRenderGraph() {
    try {
        // Проверка наличия текущего графа
        if (!confirmedGraphPath) {
            showGraphMessage("No graph selected. Please select a graph first.");
            hidePanel();
            return;
        }

        showGraphMessage("Loading graph visualization...");
        
        const response = await fetch("/api/graph/visualize_graph");
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const graph = await response.json();
        
        // Проверяем, что получили валидные данные
        if (!graph.nodes || !graph.edges) {
            throw new Error("Invalid graph data received");
        }

        // Очищаем предыдущую визуализацию
        if (network) {
            network.destroy();
            network = null;
        }

        // Подготовка данных
        const nodes = new vis.DataSet(
            graph.nodes.map(n => ({
                id: n.id,
                label: n.label,
                color: n.color || '#666',
                size: n.size || 20,
                shape: 'dot',
                data: n.data || {}
            }))
        );
        
        const edges = new vis.DataSet(
            graph.edges.map(e => ({
                from: e.from_,
                to: e.to,
                label: e.label || '',
                arrows: graph.directed ? "to" : undefined,
                data: e.data || {}
            }))
        );

        // Настройки визуализации
        const options = {
            physics: {
                barnesHut: {
                    gravitationalConstant: -30000,
                    centralGravity: 0.3,
                    springLength: 60,
                    springConstant: 0.02,
                    damping: 0.9,
                    avoidOverlap: 0.1
                },
                stabilization: {
                    iterations: 100
                }
            },
            interaction: {
                hover: true,
                zoomView: true,
                dragView: true,
                selectable: true
            },
            edges: {
                smooth: {
                    type: 'continuous'
                },
                color: '#888',
                width: 1,
                font: {
                    size: 12,
                    color: '#333'
                }
            },
            nodes: {
                shape: 'dot',
                size: 20,
                font: {
                    size: 14,
                    face: 'Arial',
                    color: '#fff'
                }
            }
        };

        // Создание сети
        const container = document.getElementById("graph-container");
        if (!container) throw new Error("Graph container not found");
        
        container.innerHTML = '';
        network = new vis.Network(container, { nodes, edges }, options);

        // Обработчики
        network.on("stabilizationIterationsDone", function() {
            hideGraphMessage();
            network.setOptions({ physics: false });
        });
        
        network.on("click", function(params) {
            if (params.nodes.length === 0 && params.edges.length === 0) {
                hidePanel();
            }
        });
        
        attachGraphHandlers(network);
        isGraphLoaded = true;
        
    } catch (error) {
        console.error("Error loading graph:", error);
        showGraphMessage(`Error loading graph: ${error.message}`);
        isGraphLoaded = false;
        hidePanel();
    }
}

// choosing or creating chat and instruct model
let selectedModelName = null;
let draftSelecledModelName = null;
let draftModelData = null;
let currentModelType = null;

function openModelSelectModal(type) {
    currentModelType = type;
    document.getElementById("model-type-placeholder").textContent = type === "chat" ? "chat" : "graph construction";
    draftSelecledModelName = selectedModelName;
    draftModelData = null;
    
    loadModels(type);
    document.getElementById("model-select-modal").style.display = "block";
    document.getElementById("modal-overlay").style.display = "block";
    
    document.querySelectorAll('#model-select-modal .tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('#model-select-modal .tab-content').forEach(content => content.classList.remove('active'));
    document.querySelector('#model-select-modal .tab-btn[data-tab="select"]').classList.add('active');
    document.getElementById('select-tab').classList.add('active');
    document.getElementById('create-tab').classList.remove('active');
    
    draftModelData = {
        name: '',
        option: 'ollama',
        model_name: 'model_name:latest',
        max_tokens: 1024,
        temperature: 0.7,  
        api: ''
    };
    updateCreateForm();
}

async function loadModels(type) {
    try {
        const response = await fetch("/api/models/load-all-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json"},
            body: JSON.stringify({ model_type: type })
        });
        console.log(JSON.stringify({ model_type: type }));
        const models = await response.json();
        if (!Array.isArray(models) || models.length === 0) {
            document.getElementById("if-no-models").style.display = "block";
            document.getElementById("if-no-models").textContent = "No models are available. Create a new one";
            return;
        } else { 
            document.getElementById("if-no-models").style.display = "none"; 
        }
        renderModelList(models);
    } catch (error) {
        console.error("Error loading models:", error);
        document.getElementById("if-no-models").style.display = "block";
        document.getElementById("if-no-models").textContent = "Error loading models";
    }
}

function renderModelList(models) {
    const container = document.getElementById("model-list");
    const template = document.getElementById("model-item-template");

    models.forEach(model => {
        const clone = template.content.cloneNode(true);
        const modelItem = clone.querySelector(".model-item");
        modelItem.dataset.modelId = model.id;
        
        clone.querySelector(".model-name").textContent = model.display_name || model.name;
        clone.querySelector(".model-option").textContent = 
            model.connection_type === "ollama" ? "ollama local" :
            model.connection_type === "openai" ? "openai api" :
            model.connection_type === "hf_local" ? "hugging face local" : "hugging face api";
        clone.querySelector(".model-item-name").textContent = model.name;
        clone.querySelector(".model-item-temperature").textContent = model.temperature;
        clone.querySelector(".model-item-tokens").textContent = model.max_tokens;
        
        const button = clone.querySelector(".model-select-btn");
        if (model.is_current) {
            button.textContent = "Current model";
            button.disabled = true;
            button.setAttribute("data-current", "true");
            selectedModelName = model.name;
            draftSelecledModelName = selectedModelName;
        }
        container.appendChild(clone);
    });
}

function selectModel(name) {
    selectedModelName = name;
    
    document.querySelectorAll('.model-item').forEach(item => {
        item.style.borderColor = '#eee';
        item.style.backgroundColor = '#f9f9f9';
    });
    
    const selectedItem = document.querySelector(`.model-item[data-model-id="${modelId}"]`);
    if (selectedItem) {
        selectedItem.style.borderColor = '#1890ff';
        selectedItem.style.backgroundColor = '#f0f7ff';
    }
}

function updateCreateForm() {
    if (!draftModelData) return;
    document.getElementById('connection-type').value = draftModelData.connection_type;
    document.getElementById('model-name').value = draftModelData.model_name;
    document.getElementById('temperature').value = draftModelData.temperature;
    document.getElementById('max-tokens').value = draftModelData.max_tokens;
    document.getElementById('model-display-name').value = draftModelData.display_name;
    document.getElementById('api-key').value = draftModelData.api_key;
    
    const apiKeyField = document.getElementById('api-key-div');
    apiKeyField.style.display = draftModelData.connection_type === 'openai' || draftModelData.connection_type === 'hf_api' ? 'flex' : 'none';
}

async function saveModelSelection() {
    try {
        let result = null;
        if (document.querySelector('#model-select-modal .tab-btn.active').getAttribute('data-tab') === 'select') {
            if (!selectedModelName) {
                alert("No model is selected!");
                return;
            }     
            const response = await fetch("/api/models/select-chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    model_name: selectedModelName,
                    model_type: currentModelType
                })
            });
            result = await response.json();
        } else {
            console.log(JSON.stringify({ 
                    ...draftModelData,
                    model_type: currentModelType
                }));
            if (!draftModelData) {
                return;
            }
            const response = await fetch("/api/models/create-and-select-chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    ...draftModelData,
                    model_type: currentModelType
                })
            });
            result = await response.json();
            selectedModelName = result.id;
        }
        if (result) {
            if (currentModelType === "chat") {
                document.getElementById("chat-model-btn").textContent = result.display_name;
            } else if (currentModelType === "graph") {
                document.getElementById("graph-model-btn").textContent = result.display_name;
            } else if (currentModelType === "embedding") {
                document.getElementById("embedding-model-btn").textContent = result.display_name;
            }
        }
        closeModelSelectModal();
    } catch (error) {
        console.error("Error saving model:", error);
        alert("Error saving model");
    }
}

function closeModelSelectModal() {
    document.getElementById("model-select-modal").style.display = "none";
    document.getElementById("modal-overlay").style.display = "none";
    selectedModelName = null;
    draftModelData = null;
    currentModelType = null;
}