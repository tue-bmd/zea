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
                button.innerText = `${op}`;
                button.onclick = () => addBlock(op);
                operationsList.appendChild(button);
            });
        })
        .catch(error => console.error('Error fetching operations:', error));

    function addBlock(name) {
        const id = `block-${Object.keys(blocks).length}`;
        const x = 50 + Object.keys(blocks).length * 120;
        const y = 100;

        const block = document.createElement("div");
        block.classList.add("block");
        block.id = id;
        block.innerHTML = `
            <span>${name}</span>
            <button class="delete-btn" onclick="deleteBlock('${id}')">X</button>
        `;
        block.style.left = `${x}px`;
        block.style.top = `${y}px`;

        canvas.appendChild(block);

        jsPlumbInstance.draggable(block);

        // Add input endpoint
        jsPlumbInstance.addEndpoint(block, {
            isSource: false,
            isTarget: true,
            endpoint: "Dot",
            anchor: "LeftMiddle",
            maxConnections: 1
        });

        // Add output endpoint
        jsPlumbInstance.addEndpoint(block, {
            isSource: true,
            isTarget: false,
            endpoint: "Dot",
            anchor: "RightMiddle",
            maxConnections: -1 // Unlimited connections
        });

        blocks[id] = { id, name, x, y };
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

    // Prevent multiple connections to a single input
    jsPlumbInstance.bind("beforeDrop", function (info) {
        const targetEndpoint = info.targetEndpoint;
        if (targetEndpoint.connections.length >= targetEndpoint.maxConnections) {
            alert("This input already has a connection.");
            return false;
        }
        return true;
    });

    function exportPipeline() {
        const pipelineConfig = {
            blocks: Object.values(blocks),
            connections: connections
        };

        // Export as YAML (optional: you can also just log it or send it to the backend)
        const yaml = jsyaml.dump(pipelineConfig);
        console.log("Pipeline YAML:\n", yaml);

        // Example: POST the pipeline config to the backend for saving
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
});
