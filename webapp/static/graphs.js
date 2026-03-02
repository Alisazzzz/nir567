// js-file with all frontend logic about graph



document.addEventListener("DOMContentLoaded", () => {
    //get current active graph if server saved it
    getCurrentGraph();
    //bind some tricky events to buttons and elements
    bindButtons();
});

async function getCurrentGraph() {
    try {
        const response = await fetch("/api/graph/get-current");
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
}

async function updateGraphFromChat() {
    try {
        const response = await fetch("/api/graph/update", { method: "POST" });
        const data = await response.json();
        if (document.getElementById('graph-tab').classList.contains('active')) {
            loadAndRenderGraph();
        }
    } catch (error) {
        console.error("Graph update error:", error);
    }
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
        const response = await fetch("/api/graph/select-graph", {
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

// ========== Graph Modal Functions ==========

function openCreateGraphModal() {
    document.getElementById('graph-create-modal').style.display = 'block';
    loadEmbeddingModels();
}

function cancelGraphCreation() {
    document.getElementById('graph-create-modal').style.display = 'none';
    resetGraphForm();
}

function handleGraphFileSelect(input) {
    const fileName = input.files[0] ? input.files[0].name : 'No file selected';
    document.getElementById('graph-doc-name').textContent = fileName;
}

function loadEmbeddingModels() {
    fetch('/api/models/embedding/list', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(models => {
        const select = document.getElementById('embedding-model-select');
        select.innerHTML = '<option value="">Select embedding model</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            select.appendChild(option);
        });
    })
    .catch(error => {
        console.error('Error loading embedding models:', error);
    });
}

function createGraph() {
    const docInput = document.getElementById('graph-doc-input');
    const graphName = document.getElementById('graph-name-input').value.trim();
    const modelId = document.getElementById('embedding-model-select').value;
    
    // Validation
    if (!docInput.files[0]) {
        alert('Please select a document');
        return;
    }
    if (!graphName) {
        alert('Please enter a graph name');
        return;
    }
    if (!modelId) {
        alert('Please select an embedding model');
        return;
    }
    
    const formData = new FormData();
    formData.append('document', docInput.files[0]);
    formData.append('graph_name', graphName);
    formData.append('embedding_model_id', modelId);
    
    fetch('/api/graphs/create', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to create graph');
        }
        return response.json();
    })
    .then(data => {
        console.log('Graph created:', data);
        cancelGraphCreation();
        
        // Callback or event
        if (typeof onGraphCreated === 'function') {
            onGraphCreated(data);
        }
    })
    .catch(error => {
        console.error('Error creating graph:', error);
        alert('Failed to create graph: ' + error.message);
    });
}

function resetGraphForm() {
    document.getElementById('graph-doc-input').value = '';
    document.getElementById('graph-doc-name').textContent = 'No file selected';
    document.getElementById('graph-name-input').value = '';
    document.getElementById('embedding-model-select').value = '';
}

// ========== Model Modal Functions ==========

function openCreateModelModal() {
    document.getElementById('create-embedding-modal').style.display = 'block';
}

function cancelModelCreation() {
    document.getElementById('create-embedding-modal').style.display = 'none';
    resetModelForm();
}

function handleConnectionTypeChange() {
    const connectionType = document.getElementById('connection-type-select').value;
    const modelInput = document.getElementById('model-path-input');
    
    switch(connectionType) {
        case 'ollama':
            modelInput.placeholder = 'e.g., llama2, mistral, nomic-embed-text';
            break;
        case 'openai':
            modelInput.placeholder = 'e.g., text-embedding-ada-002';
            break;
        case 'hf_local':
            modelInput.placeholder = 'e.g., sentence-transformers/all-MiniLM-L6-v2';
            break;
        default:
            modelInput.placeholder = 'Model name or path';
    }
}

function createModel() {
    const connectionType = document.getElementById('connection-type-select').value;
    const modelPath = document.getElementById('model-path-input').value.trim();
    const modelName = document.getElementById('model-name-input').value.trim();
    
    // Validation
    if (!connectionType) {
        alert('Please select a connection type');
        return;
    }
    if (!modelPath) {
        alert('Please enter a model');
        return;
    }
    if (!modelName) {
        alert('Please enter a model name');
        return;
    }
    
    fetch('/api/models/embedding/create', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            connection_type: connectionType,
            model: modelPath,
            name: modelName
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to create model');
        }
        return response.json();
    })
    .then(data => {
        console.log('Model created:', data);
        cancelModelCreation();
        
        // Reload models in graph modal
        loadEmbeddingModels();
        
        // Auto-select the created model
        setTimeout(() => {
            document.getElementById('embedding-model-select').value = data.id;
        }, 100);
        
        // Callback
        if (typeof onModelCreated === 'function') {
            onModelCreated(data);
        }
    })
    .catch(error => {
        console.error('Error creating model:', error);
        alert('Failed to create model: ' + error.message);
    });
}

function resetModelForm() {
    document.getElementById('connection-type-select').value = '';
    document.getElementById('model-path-input').value = '';
    document.getElementById('model-name-input').value = '';
}

// ========== Callback Functions (override these as needed) ==========

function onGraphCreated(graph) {
    console.log('Graph created successfully:', graph);
    // Add your custom logic here
}

function onModelCreated(model) {
    console.log('Model created successfully:', model);
    // Add your custom logic here
}

// ========== Close modals on outside click ==========

window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
}