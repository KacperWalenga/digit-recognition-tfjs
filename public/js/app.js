document.addEventListener('DOMContentLoaded', function() {
  const drawTab = document.getElementById('drawTab');
  const uploadTab = document.getElementById('uploadTab');
  const drawContent = document.getElementById('drawContent');
  const uploadContent = document.getElementById('uploadContent');
  
  if (drawTab && uploadTab) {
    drawTab.addEventListener('click', () => {
      drawTab.classList.add('active');
      uploadTab.classList.remove('active');
      drawContent.classList.add('active');
      uploadContent.classList.remove('active');
    });
    
    uploadTab.addEventListener('click', () => {
      uploadTab.classList.add('active');
      drawTab.classList.remove('active');
      uploadContent.classList.add('active');
      drawContent.classList.remove('active');
    });
  }
  
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const imagePreview = document.getElementById('imagePreview');
  const fileName = document.getElementById('fileName');
  
  if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }
  
  if (fileInput) {
    fileInput.addEventListener('change', handleFileSelect);
  }
  
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
      }
    });
  }
  
  function handleFileSelect() {
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];
      fileName.textContent = file.name;
      
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
      };
      reader.readAsDataURL(file);
    }
  }
  
  const canvas = document.getElementById('drawingCanvas');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const recognizeBtn = document.getElementById('recognizeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const guidelineBtn = document.getElementById('guidelineBtn');
  
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  let isDrawing = false;
  let showGrid = false;
  
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'black';
  
  if (recognizeBtn) {
    recognizeBtn.addEventListener('click', recognizeDrawing);
  }
  
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing);
  
  canvas.addEventListener('touchstart', handleTouch);
  canvas.addEventListener('touchmove', handleTouch);
  canvas.addEventListener('touchend', stopDrawing);
  
  function startDrawing(e) {
    isDrawing = true;
    draw(e);
  }
  
  function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    let x, y;
    
    if (e.type.includes('mouse')) {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    } else {
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    }
    
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  
  function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
  }
  
  function handleTouch(e) {
    e.preventDefault();
    
    if (e.type === 'touchstart') {
      startDrawing(e);
    } else if (e.type === 'touchmove') {
      draw(e);
    }
  }
  
  if (clearBtn) {
    clearBtn.addEventListener('click', clearCanvas);
  }
  
  function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (showGrid) {
      drawGrid();
    }
  }
  
  if (guidelineBtn) {
    guidelineBtn.addEventListener('click', toggleGrid);
  }
  
  function toggleGrid() {
    showGrid = !showGrid;
    
    if (showGrid) {
      guidelineBtn.textContent = 'Ukryj siatkę';
      drawGrid();
    } else {
      guidelineBtn.textContent = 'Pokaż siatkę';
      clearCanvas();
    }
  }
  
  function drawGrid() {
    ctx.save();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    const step = canvas.width / 3;
    
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(step * i, 0);
      ctx.lineTo(step * i, canvas.height);
      ctx.stroke();
    }
    
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(0, step * i);
      ctx.lineTo(canvas.width, step * i);
      ctx.stroke();
    }
    
    ctx.restore();
  }
  
  function recognizeDrawing() {
    canvas.toBlob(function(blob) {
      recognizeBtn.textContent = 'Rozpoznawanie...';
      recognizeBtn.disabled = true;
      
      const formData = new FormData();
      formData.append('image', blob, 'drawing.png');
      
      fetch('/recognize', {
        method: 'POST',
        body: formData
      })
      .then(response => response.text())
      .then(html => {
        document.open();
        document.write(html);
        document.close();
      })
      .catch(error => {
        console.error('Błąd:', error);
        recognizeBtn.textContent = 'Rozpoznaj';
        recognizeBtn.disabled = false;
        alert('Wystąpił błąd podczas rozpoznawania. Spróbuj ponownie.');
      });
    }, 'image/png');
  }
});