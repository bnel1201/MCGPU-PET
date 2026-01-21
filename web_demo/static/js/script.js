document.addEventListener('DOMContentLoaded', () => {

    // --- State ---
    let sinogramData = null; // { data: Int32Array, shape: [slices, angles, rad] }
    let imageData = null;    // { data: Int32Array, shape: [z, y, x] }

    // Zoom control state
    let sinoZoom = 1.0;
    let reconZoom = 1.0;
    let isDragging = false;
    let dragStartY = 0;
    let dragTarget = null; // 'sino' or 'recon'

    let currentSinoSlice = 0;
    let currentReconSlice = 0;

    // --- Elements ---
    const runBtn = document.getElementById('run-btn');
    const progressBar = document.getElementById('progress-bar');
    const statusBar = document.getElementById('status-bar');

    const sinoCanvas = document.getElementById('sino-canvas');
    const reconCanvas = document.getElementById('recon-canvas');

    const sinoInfo = document.getElementById('sino-info');
    const reconInfo = document.getElementById('recon-info');

    // --- Params ---
    const getParams = () => {
        return {
            time_sec: document.getElementById('time_sec').value,
            activity_phantom: document.getElementById('activity_phantom').value,
            activity_bg: document.getElementById('activity_bg').value,
            fov_z: document.getElementById('fov_z').value,
            detector_radius: document.getElementById('detector_radius').value,
            image_res: document.getElementById('image_res').value
        };
    };

    // --- Helper: Decode Base64 Numpy ---
    const decodeArray = (b64Data, dtype = 'int32') => {
        const binaryString = window.atob(b64Data);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        if (dtype === 'float32') {
            return new Float32Array(bytes.buffer);
        }
        return new Int32Array(bytes.buffer);
    };

    // --- Rendering ---

    // Auto-contrast normalization
    const getNormalizationFactor = (arr) => {
        // Simple percentile-ish max (skip actual sort for speed, just max)
        // Or finding global max
        let max = 0;
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] > max) max = arr[i];
        }
        return max > 0 ? 255 / max : 1;
    };

    const renderSlice = (canvas, flatData, width, height, sliceIndex, sliceStride, globalMax) => {
        const ctx = canvas.getContext('2d');

        // Resize canvas if needed
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;

        const imgData = ctx.createImageData(width, height);
        const data = imgData.data;

        const start = sliceIndex * sliceStride;
        const scale = 255 / (globalMax || 1); // Normalize locally or globally? Globally is better for comparison

        for (let i = 0; i < sliceStride; i++) {
            const val = flatData[start + i];
            const pixelVal = Math.floor(val * scale);

            // Grayscale (Inferno-ish mapping could be done here manually if we want color)
            // Let's do simple grayscale for now
            data[i * 4] = pixelVal;     // R
            data[i * 4 + 1] = pixelVal; // G
            data[i * 4 + 2] = pixelVal; // B
            data[i * 4 + 3] = 255;      // A
        }

        ctx.putImageData(imgData, 0, 0);
    };

    const getSliceMax = (flatData, start, length) => {
        let max = 0;
        for (let i = 0; i < length; i++) {
            if (flatData[start + i] > max) max = flatData[start + i];
        }
        return max;
    };

    const updateSinoView = () => {
        if (!sinogramData) return;
        const [slices, angles, rads] = sinogramData.shape;
        const width = rads;
        const height = angles;

        sinoInfo.textContent = `Slice: ${currentSinoSlice + 1}/${slices}`;

        // Slice stride = angles * rads
        const stride = width * height;
        const start = currentSinoSlice * stride;
        const localMax = getSliceMax(sinogramData.data, start, stride);

        renderSlice(sinoCanvas, sinogramData.data, width, height, currentSinoSlice, stride, localMax);
    };

    const updateReconView = () => {
        if (!imageData) return;
        const [d, h, w] = imageData.shape; // Z, Y, X

        reconInfo.textContent = `Slice: ${currentReconSlice + 1}/${d}`;

        const stride = w * h;
        const start = currentReconSlice * stride;
        const localMax = getSliceMax(imageData.data, start, stride);

        renderSlice(reconCanvas, imageData.data, w, h, currentReconSlice, stride, localMax);
    };


    // --- Reconstruction UI Logic ---
    const reconMethodSelect = document.getElementById('recon_method');
    const paramGroups = {
        'FBP': document.getElementById('params-FBP'),
        'OSEM': document.getElementById('params-OSEM'),
        'BSREM': document.getElementById('params-BSREM')
    };

    reconMethodSelect.addEventListener('change', () => {
        const method = reconMethodSelect.value;
        // Hide all
        Object.values(paramGroups).forEach(el => el.classList.add('hidden'));
        // Show selected
        if (paramGroups[method]) paramGroups[method].classList.remove('hidden');
    });

    const getReconParams = () => {
        const method = reconMethodSelect.value;
        const params = { method: method, slice_index: currentReconSlice }; // Recon current view slice

        if (method === 'OSEM') {
            params.iterations = document.getElementById('osem_iters').value;
            params.subsets = document.getElementById('osem_subsets').value;
        } else if (method === 'BSREM') {
            params.iterations = document.getElementById('bsrem_iters').value;
            params.subsets = document.getElementById('bsrem_subsets').value;
            params.beta = document.getElementById('bsrem_beta').value;
        }
        return params;
    };

    const reconBtn = document.getElementById('recon-btn');

    reconBtn.addEventListener('click', async () => {
        reconBtn.disabled = true;
        statusBar.textContent = "Reconstructing...";
        statusBar.style.color = "#fbbf24";

        const params = getReconParams();

        try {
            const response = await fetch('/api/reconstruct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            const result = await response.json();

            if (result.status === 'success') {
                // Update specific slice in the volume
                // Note: We only get one slice back.
                // We need to update our imageData volume locally.
                const sliceRes = result.results.image_slice;
                const sliceData = decodeArray(sliceRes.data, 'float32'); // Float32 from recon
                const sliceIdx = result.results.slice_index;

                // Update logic: In a real app we might recon whole volume.
                // Here we just replace the ONE slice in memory and re-render.
                // BUT: decodeArray returns new TypedArray.
                // We need to copy into the big buffer.
                // Problem: Big buffer is Int32 or Float32? 
                // Script.js line 42: new Int32Array(bytes.buffer).
                // Backend line 147: astype(np.float32).
                // WAIT.
                // When we simulated, we sent Float32?
                // app.py: encode_array -> astype(np.int32) originally?
                // Let's check app.py simulation output.
                // It was: encode_array(arr) { base64... }
                // Wrapper returns Int32.
                // JS decodeArray creates Int32Array.

                // Recon returns Float32.
                // Mismatch!
                // We should handle this. Let's assume visualization can handle float.
                // If imageData.data is Int32, we can't easily put Floats in it if we want precision.
                // But for display, we just need values.
                // Ideally, we convert everything to Float32 on JS side.

                // Hack: Convert slice to Int32 for compatibility or upgrade whole volume to Float32?
                // Let's upgrade `decodeArray` to handle types or auto-detect?

                // For now, let's just render this SINGLE SLICE directly to canvas without updating 3D volume?
                // Or update 3D volume.
                // Let's try to write into the volume.
                // If volume is Int32, floats will be cast.

                const volWidth = imageData.shape[2];
                const volHeight = imageData.shape[1];
                const start = sliceIdx * volWidth * volHeight;

                // Check if we need to upgrade storage to Float32 (if sim was Int32)
                if (!(imageData.data instanceof Float32Array)) {
                    const newBuffer = new Float32Array(imageData.data);
                    imageData.data = newBuffer;
                }

                // Copy
                for (let i = 0; i < sliceData.length; i++) {
                    imageData.data[start + i] = sliceData[i];
                }

                updateReconView(); // Re-render current slice

                statusBar.textContent = "Reconstruction Complete (" + params.method + ")";
                statusBar.style.color = "#10b981";

            } else {
                statusBar.textContent = "Error: " + result.message;
                statusBar.style.color = "#ef4444";
            }
        } catch (e) {
            console.error(e);
            statusBar.textContent = "Recon Error";
        } finally {
            reconBtn.disabled = false;
        }
    });

    // --- Event Listeners ---

    runBtn.addEventListener('click', async () => {
        const params = getParams();

        // UI Updates
        runBtn.disabled = true;
        statusBar.textContent = "Running Simulation...";
        statusBar.style.color = "#fbbf24"; // warning constant
        progressBar.style.width = "30%"; // Fake progress start
        document.getElementById('sino-overlay').classList.add('hidden');
        document.getElementById('recon-overlay').classList.add('hidden');

        // Reset zoom
        sinoZoom = 1.0;
        reconZoom = 1.0;
        sinoCanvas.style.transform = `scale(1)`;
        reconCanvas.style.transform = `scale(1)`;

        try {
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            progressBar.style.width = "80%";

            const result = await response.json();

            if (result.status === 'success') {
                const results = result.results;

                // Process Sinogram
                if (results.sinogram) {
                    sinogramData = {
                        data: decodeArray(results.sinogram.data, 'int32'),
                        shape: results.sinogram.shape,
                        max: results.sinogram.max
                    };
                    currentSinoSlice = Math.floor(sinogramData.shape[0] / 2);
                    updateSinoView();
                }

                // Process Image
                if (results.image) {
                    imageData = {
                        data: decodeArray(results.image.data, 'int32'),
                        shape: results.image.shape,
                        max: results.image.max
                    };
                    currentReconSlice = Math.floor(imageData.shape[0] / 2);
                    updateReconView();
                }

                statusBar.textContent = "Simulation Complete";
                statusBar.style.color = "#10b981";
                progressBar.style.width = "100%";
                setTimeout(() => { progressBar.style.width = "0%"; }, 1000);

                // Enable Recon
                reconBtn.disabled = false;

            } else {
                statusBar.textContent = "Error: " + result.message;
                statusBar.style.color = "#ef4444";
            }
        } catch (e) {
            console.error(e);
            statusBar.textContent = "Network Error";
            statusBar.style.color = "#ef4444";
        } finally {
            runBtn.disabled = false;
        }
    });

    // Scrolling logic
    const handleScroll = (e, type) => {
        e.preventDefault();
        const delta = Math.sign(e.deltaY) * -1; // Up is positive (next slice), Down is negative

        if (type === 'sino' && sinogramData) {
            const max = sinogramData.shape[0] - 1;
            let newSlice = currentSinoSlice + delta;
            if (newSlice < 0) newSlice = 0;
            if (newSlice > max) newSlice = max;
            currentSinoSlice = newSlice;
            updateSinoView();
        } else if (type === 'recon' && imageData) {
            const max = imageData.shape[0] - 1;
            let newSlice = currentReconSlice + delta;
            if (newSlice < 0) newSlice = 0;
            if (newSlice > max) newSlice = max;
            currentReconSlice = newSlice;
            updateReconView();
        }
    };

    sinoCanvas.parentElement.addEventListener('wheel', (e) => handleScroll(e, 'sino'));
    reconCanvas.parentElement.addEventListener('wheel', (e) => handleScroll(e, 'recon'));

    // Zoom Logic (Left-click drag)

    // Disable context menu on canvases just in case, though we are using left click now
    sinoCanvas.parentElement.addEventListener('contextmenu', e => e.preventDefault());
    reconCanvas.parentElement.addEventListener('contextmenu', e => e.preventDefault());

    const handleMouseDown = (e, target) => {
        if (e.button === 0) { // Left click
            e.preventDefault();
            isDragging = true;
            dragStartY = e.clientY;
            dragTarget = target;
        }
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
        e.preventDefault();

        const deltaY = e.clientY - dragStartY;
        dragStartY = e.clientY; // Update for incremental change

        // Drag down (positive delta) -> Zoom In
        // Drag up (negative delta) -> Zoom Out
        // Sensitivity factor
        const sensitivity = 0.01;
        const zoomFactor = 1 + (deltaY * sensitivity);

        if (dragTarget === 'sino') {
            sinoZoom *= zoomFactor;
            // Clamp zoom reasonably
            if (sinoZoom < 0.1) sinoZoom = 0.1;
            if (sinoZoom > 10) sinoZoom = 10;
            sinoCanvas.style.transform = `scale(${sinoZoom})`;
        } else if (dragTarget === 'recon') {
            reconZoom *= zoomFactor;
            if (reconZoom < 0.1) reconZoom = 0.1;
            if (reconZoom > 10) reconZoom = 10;
            reconCanvas.style.transform = `scale(${reconZoom})`;
        }
    };

    const handleMouseUp = () => {
        isDragging = false;
        dragTarget = null;
    };

    // Attach listeners
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    sinoCanvas.parentElement.addEventListener('mousedown', (e) => handleMouseDown(e, 'sino'));
    reconCanvas.parentElement.addEventListener('mousedown', (e) => handleMouseDown(e, 'recon'));

});
