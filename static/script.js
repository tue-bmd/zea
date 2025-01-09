document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("canvas");
    const operationsList = document.getElementById("operations-list");
    const blocks = {};
    const connections = [];
    const jsPlumbInstance = jsPlumb.getInstance();

    jsPlumbInstance.setContainer(canvas);

    // Fetch operations from the backend
    fetch('/operations')
        .then(response => response.json())
        .then(operations => {
            operations.forEach(op => {
                const button = document.createElement("button");
                button.innerText = op.name;
                button.onclick = () => addBlock(op);
                operationsList.appendChild(button);
            });
        })
        .catch(error => console.error('Error fetching operations:', error));

    function addBlock(op) {
        const id = `block-${Object.keys(blocks).length}`;
        const x = 50 + Object.keys(blocks).length * 120;
        const y = 100;

        const block = document.createElement("div");
        block.classList.add("block");
        block.id = id;
        block.innerHTML = `
            <span>${op.name}</span>
            <button class="toggle-params-btn">Show/Hide Parameters</button>
            <div class="parameters" style="display: none;">
                ${op.init_keys.map(key => `<label>${key}: <input type="text" data-key="${key}" /></label>`).join("")}
            </div>
            <button class="delete-btn" onclick="deleteBlock('${id}')" style="position: absolute; top: 0px; right: 0px; border-radius: 30%; width: 18px; height: 18px;">x</button>
        `;
        block.style.left = `${x}px`;
        block.style.top = `${y}px`;

        canvas.appendChild(block);

        jsPlumbInstance.draggable(block);

        // Toggle parameters visibility
        const toggleParamsBtn = block.querySelector(".toggle-params-btn");
        const parametersDiv = block.querySelector(".parameters");
        toggleParamsBtn.addEventListener("click", () => {
            parametersDiv.style.display = parametersDiv.style.display === "none" ? "block" : "none";
        });

        // Add input endpoints for multiple inputs
        if (op.allow_multiple_inputs) {
            jsPlumbInstance.addEndpoint(block, {
                isSource: false,
                isTarget: true,
                endpoint: "Dot",
                anchor: "TopLeft",
                maxConnections: 1
            });

            jsPlumbInstance.addEndpoint(block, {
                isSource: false,
                isTarget: true,
                endpoint: "Dot",
                anchor: "BottomLeft",
                maxConnections: 1
            });
        } else {
            // Single input endpoint
            jsPlumbInstance.addEndpoint(block, {
                isSource: false,
                isTarget: true,
                endpoint: "Dot",
                anchor: "LeftMiddle",
                maxConnections: 1
            });
        }

        // Add output endpoint
        jsPlumbInstance.addEndpoint(block, {
            isSource: true,
            isTarget: false,
            endpoint: "Dot",
            anchor: "RightMiddle",
            maxConnections: -1 // Unlimited outputs
        });

        blocks[id] = { id, name: op.name, parameters: op.init_keys.reduce((acc, key) => ({ ...acc, [key]: "" }), {}), x, y };
    }

    // Delete a block and its connections
    window.deleteBlock = function (blockId) {
        const block = document.getElementById(blockId);

        // Remove all connections associated with the block
        jsPlumbInstance.removeAllEndpoints(blockId);

        // Remove the block from the canvas
        block.remove();

        // Remove the block from the blocks object
        delete blocks[blockId];

        // Update connections array to exclude those related to the deleted block
        const updatedConnections = connections.filter(
            conn => conn.source !== blockId && conn.target !== blockId
        );
        connections.length = 0;
        connections.push(...updatedConnections);
    };

    // Listen for connections
    jsPlumbInstance.bind("connection", function (info) {
        const sourceId = info.sourceId;
        const targetId = info.targetId;

        connections.push({
            source: sourceId,
            target: targetId
        });
    });

    // Prevent multiple connections to a single input unless allowed
    jsPlumbInstance.bind("beforeDrop", function (info) {
        const targetEndpoint = info.targetEndpoint;
        if (targetEndpoint && targetEndpoint.maxConnections !== -1 && targetEndpoint.connections.length >= targetEndpoint.maxConnections) {
            alert("This input already has a connection.");
            return false;
        }
        return true;
    });

    function exportPipeline() {
        const pipelineConfig = {
            blocks: Object.values(blocks).map(block => ({
                id: block.id,
                name: block.name,
                parameters: block.parameters,
                x: block.x,
                y: block.y
            })),
            connections: connections
        };

        // Export as YAML
        const yaml = jsyaml.dump(pipelineConfig);
        console.log("Pipeline YAML:\n", yaml);

        // POST the pipeline config to the backend
        fetch('/save_pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(pipelineConfig)
        })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Error saving pipeline:', error));
    }

    // Expose the export function globally
    window.exportPipeline = exportPipeline;

    // Track changes to parameter inputs
    canvas.addEventListener("input", function (event) {
        const blockElement = event.target.closest(".block");
        if (blockElement) {
            const blockId = blockElement.id;
            const key = event.target.dataset.key;
            blocks[blockId].parameters[key] = event.target.value;
        }
    });
});