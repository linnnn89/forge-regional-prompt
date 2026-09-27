class ForgeCoupleMaskHandler {
	/** @type {HTMLDivElement} */ #group = undefined;
	/** @type {HTMLDivElement} */ #gallery = undefined;
	/** @type {HTMLDivElement} */ #preview = undefined;
	/** @type {HTMLInputElement} */ #separatorField = undefined;
	/** @type {HTMLInputElement} */ #background = undefined;
	/** @type {HTMLTextAreaElement} */ #promptField = undefined;
	/** @type {HTMLTextAreaElement} */ #weightField = undefined;
	/** @type {HTMLTextAreaElement} */ #operationField = undefined;
	/** @type {HTMLButtonElement} */ #operationButton = undefined;
	/** @type {HTMLButtonElement} */ #loadButton = undefined;
    #pendingOperation = null;
    #revision = 0;

    constructor(group, gallery, preview, separatorField, background, promptField, weightField, operationField, operationButton, loadButton) {
        this.#group = group;
        this.#gallery = gallery;
        this.#preview = preview;
        this.#separatorField = separatorField;
        this.#background = background;
        this.#promptField = promptField;
        this.#weightField = weightField;
        this.#operationField = operationField;
        this.#operationButton = operationButton;
        this.#loadButton = loadButton;

        this.#separatorField.addEventListener("blur", () => { this.syncPrompts(); });
        this.#group.addEventListener("fc-operation-done", (event) => {
            if (event.detail.success) this.#finishOperation();
            this.#allRows.forEach((row) => row.querySelectorAll("button, input").forEach((el) => { el.disabled = false; }));
            this.#pendingOperation = null;
        });
    }

    /** @returns {string} */
    get #separator() {
        const sep = this.#separatorField.value.trim();
        return !sep ? "\n" : sep.replace(/\\n/g, "\n").split("\n").map((c) => c.trim()).join("\n");
    }

    /** @returns {boolean} */
    get #selectionAvailable() {
        return !this.#loadButton.disabled;
    }

    /** @returns {HTMLDivElement[]} */
    get #allRows() {
        return Array.from(this.#preview.querySelectorAll(".fc_mask_row"));
    }

    hideButtons() {
        const undo = this.#group.querySelector("button[aria-label='Undo']");
        if (undo == null || undo.style.display === "none") return;

        undo.style.display = "none";

        const clear = this.#group.querySelector("button[aria-label='Clear']");
        clear.style.display = "none";

        const remove = this.#group.querySelector("button[aria-label='Remove Image']");
        remove.style.display = "none";

        const brush = this.#group.querySelector("button[aria-label='Use brush']");
        brush.firstElementChild.style.width = "20px";
        brush.firstElementChild.style.height = "20px";

        const color = this.#group.querySelector("button[aria-label='Select brush color']");
        color.firstElementChild.style.width = "20px";
        color.firstElementChild.style.height = "20px";

        brush.parentElement.parentElement.style.top = "var(--size-2)";
        brush.parentElement.parentElement.style.right = "var(--size-10)";
    }

    generatePreview() {
        const imgs = this.#gallery.querySelectorAll("img");
        const maskCount = imgs.length;

        // Clear Excess Rows
        while (this.#preview.children.length > maskCount) this.#preview.lastElementChild.remove();

        // Append Insufficient Rows
        while (this.#preview.children.length < maskCount) {
            const row = document.createElement("div");
            row.classList.add("fc_mask_row");
            this.#preview.appendChild(row);
        }

        this.#populateRows(this.#allRows, imgs);
        const payload = this.#weightField.value ? JSON.parse(this.#weightField.value) : {revision: 0, weights: []};
        this.#revision = payload.revision;
        const weights = payload.weights;
        this.#allRows.forEach((row, i) => {
            row.weight.value = Number(weights[i] ?? 1).toFixed(2);
        });
        this.syncPrompts();

        if (!this.#selectionAvailable) {
            const lastSelected = this.#preview.querySelector(".selected");
            if (lastSelected) lastSelected.classList.remove("selected");
        }
    }

    /** @param {HTMLDivElement} row */
    #constructRow(row) {
        if (row.hasAttribute("setup")) return;

        const img = document.createElement("img");
        img.title = "Select this Mask";
        img.setAttribute("style", "width: 96px !important; height: 96px !important; object-fit: contain;");
        img.addEventListener("click", () => {
            this.#onSelectRow(row);
        });

        row.appendChild(img);
        row.img = img;

        const txt = document.createElement("input");
        txt.value = "";
        txt.setAttribute("style", "width: 80%;");
        txt.setAttribute("type", "text");
        txt.addEventListener("blur", () => { this.#onSubmitPrompt(); });

        row.appendChild(txt);
        row.txt = txt;

        const weight = document.createElement("input");
        weight.title = "Weight";
        weight.value = Number(1.0).toFixed(2);
        weight.setAttribute("style", "width: 10%;");
        weight.setAttribute("type", "number");
        weight.addEventListener("blur", () => {
            this.#onSubmitWeight(weight);
        });

        row.appendChild(weight);
        row.weight = weight;

        const del = document.createElement("button");
        del.classList.add("del");
        del.textContent = "❌";
        del.title = "删除此图层及对应区域提示词";
        del.addEventListener("click", () => {
            this.#onDeleteRow(row);
        });

        row.appendChild(del);

        const up = document.createElement("button");
        up.classList.add("up");
        up.textContent = "^";
        up.title = "上移图层（蒙版、提示词和权重一起移动）";
        up.addEventListener("click", () => {
            this.#onShiftRow(row, true);
        });

        row.appendChild(up);

        const down = document.createElement("button");
        down.classList.add("down");
        down.textContent = "^";
        down.title = "下移图层（蒙版、提示词和权重一起移动）";
        down.addEventListener("click", () => {
            this.#onShiftRow(row, false);
        });

        row.appendChild(down);

        row.setAttribute("setup", true);
    }

    /** @param {HTMLDivElement[]} rows @param {HTMLImageElement[]} imgs */
    #populateRows(rows, imgs) {
        rows.forEach((row, i) => {
            this.#constructRow(row);
            row.img.src = imgs[i].src;
            row.img.title = `区域 ${i + 1}：点击选择，再载入画布修正`;
            row.img.alt = `区域 ${i + 1}`;
            row.txt.placeholder = `区域 ${i + 1} 的提示词`;
            row.txt.setAttribute("aria-label", `区域 ${i + 1} 的提示词`);
            row.weight.setAttribute("aria-label", `区域 ${i + 1} 的权重`);
        });
    }

    #onSubmitPrompt() {
        const prompts = this.#allRows.map((row) => row.txt.value);

        const radio = this.#background.querySelector("div.wrap>label.selected>span");
        const background = radio.textContent;

        const existingPrompts = this.#promptField.value.split(this.#separator).map((line) => line.trim());

        const global = background === "First Line" ? existingPrompts.shift() :
            background === "Last Line" ? existingPrompts.pop() : undefined;
        const merged = [...prompts, ...existingPrompts.slice(prompts.length)];
        if (background === "First Line") merged.unshift(global ?? "");
        else if (background === "Last Line") merged.push(global ?? "");
        this.#promptField.value = merged.join(this.#separator);
        updateInput(this.#promptField);
    }

    #submitOperation(operation) {
        if (this.#pendingOperation !== null) return;
        // Include the current values even if a blur callback is still queued.
        this.parseWeights();
        this.#pendingOperation = operation;
        this.#allRows.forEach((row) => row.querySelectorAll("button, input").forEach((el) => { el.disabled = true; }));
        this.#operationField.value = operation;
        updateInput(this.#operationField);
        this.#operationButton.click();
    }

    #finishOperation() {
        const operation = this.#pendingOperation;
        if (!operation || (!operation.includes("=") && !operation.startsWith("-"))) return;
        const background = this.#background.querySelector("div.wrap>label.selected>span").textContent;
        const prompts = this.#promptField.value.split(this.#separator);
        const global = background === "First Line" ? prompts.shift() :
            background === "Last Line" ? prompts.pop() : undefined;
        if (operation.includes("=")) {
            const [from, to] = operation.split("=").map(Number);
            while (prompts.length <= Math.max(from, to)) prompts.push("");
            [prompts[from], prompts[to]] = [prompts[to], prompts[from]];
        } else {
            prompts.splice(Number(operation.slice(1)), 1);
        }
        if (background === "First Line") prompts.unshift(global ?? "");
        else if (background === "Last Line") prompts.push(global ?? "");
        this.#promptField.value = prompts.join(this.#separator);
        updateInput(this.#promptField);
    }

    /** @param {HTMLInputElement} field */
    #onSubmitWeight(field) {
        const w = this.#clamp05(field.value);
        field.value = w.toFixed(2);
        this.parseWeights();
    }

    /** @param {HTMLDivElement} row */
    #onSelectRow(row) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);

        const lastSelected = this.#preview.querySelector(".selected");
        if (lastSelected) lastSelected.classList.remove("selected");
        row.classList.add("selected");

        this.#submitOperation(`${index}`);
    }

    /** @param {HTMLDivElement} row */
    #onDeleteRow(row) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);

        this.#submitOperation(`-${index}`);
    }

    /** @param {HTMLDivElement} row @param {boolean} isUp */
    #onShiftRow(row, isUp) {
        const rows = Array.from(this.#allRows);
        const index = rows.indexOf(row);
        const target = isUp ? index - 1 : index + 1;

        if (target < 0 || target >= rows.length) return;

        this.#submitOperation(`${index}=${target}`);
    }

    syncPrompts() {
        const prompt = this.#promptField.value;
        let prompts = prompt.split(this.#separator).map((line) => line.trim());

        const radio = this.#background.querySelector("div.wrap>label.selected>span");
        const background = radio.textContent;

        if (background === "First Line") prompts = prompts.slice(1);
        else if (background === "Last Line") prompts = prompts.slice(0, -1);

        const active = document.activeElement;
        this.#allRows.forEach((row, i) => {
            const promptCell = row.txt;

            // Skip the Cell being Edited
            if (promptCell === active) return;

            promptCell.value = i < prompts.length ? prompts[i].replace(/\n+/g, ", ").replace(/,+/g, ",") : "";
        });
        const status = this.#group.querySelector(".fc_mask_status");
        if (status) {
            const layers = this.#allRows.length;
            const empty = prompts.filter((part) => !part).length;
            const matches = layers > 0 && prompts.length === layers && empty === 0;
            status.textContent = `区域提示词 ${prompts.length} 段 · 已保存蒙版 ${layers} 层` +
                (matches ? " · 数量匹配" : layers === 0 ? " · 请先应用模板或保存图层" :
                 ` · 请让提示词段数与蒙版层数一致${empty ? `，有 ${empty} 段为空` : ""}`) +
                (background !== "None" ? `（${background === "First Line" ? "首" : "末"}段作为全局层，不计入区域数）` : "");
            status.dataset.ready = String(matches);
        }
    }

    parseWeights() {
        const weights = this.#allRows.map((row) => this.#clamp05(row.weight.value));
        this.#weightField.value = JSON.stringify({revision: this.#revision, weights});
        updateInput(this.#weightField);
    }

    /** @param {number} v @returns {number} */
    #clamp05(v) {
        const val = parseFloat(v);
        if (Number.isNaN(val)) return 0.0;
        return Math.min(Math.max(val, 0.0), 5.0);
    }
}
